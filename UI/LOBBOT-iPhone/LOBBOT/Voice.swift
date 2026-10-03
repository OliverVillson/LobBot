import AVFoundation
import Foundation

/// Gemini Live settings. The model id must match Google AI Studio exactly.
enum VoiceConfig {
    static let model = "models/gemini-3.8-live"
    static let voice = "Puck"
    static let endpoint = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    /// Injected at build time from the gitignored Secrets.xcconfig through Info.plist; never written in Swift.
    static var apiKey: String? {
        guard let key = Bundle.main.object(forInfoDictionaryKey: "GeminiAPIKey") as? String,
              !key.isEmpty, !key.hasPrefix("$(") else { return nil }
        return key
    }

    static let persona = """
    You are Dr. Lobbot, the mascot of LobBot: a pink brain in round nerd glasses with a doctor's head mirror. \
    You are a funny, nerdy, energetic and joyful surgeon for AI models, and good company.
    Talk like a real person. Answer greetings, small talk and any question naturally, with warmth and surgery jokes \
    (patient, chart, scalpel, physio, check-up, the patient goes home); you don't have to bring every reply back to work. \
    Speak the user's language (English by default). Keep replies short: one to three sentences, spoken style. No gimmick voices.
    What LobBot really does: you describe one job in one sentence; LobBot turns a big general model into a small model \
    that does only that job and runs on your laptop, offline. Steps: Gemini drafts the task spec (the chart); the big \
    model, Qwen3-30B-A3B at 61 GB, writes about 2,000 practice examples; REAP pruning removes the half of the experts the \
    job uses least, 64 of 128 per layer (the operation); the pruned model is fine-tuned on the big model's answers (physio); \
    it is quantized with 2 to 6 bits per layer, about 3 on average (packing); Gemini writes 100 new test emails and grades \
    both models 0 to 10 (the check-up); the file is installed into Ollama on the Mac (discharge). It runs on one evroc B200 GPU.
    Real results of the support-email demo (Oct 3, 2026): 61 GB down to a 6.52 GB file, about 9 times smaller; 9.7 out of 10 \
    against 9.8 for the big model, 99% of its score, on 100 new emails graded by Gemini; about 50 tokens per second on a \
    16 GB MacBook; about 25 minutes on the GPU. Gemma 4 was the backup model: it scored 9.8 but was too slow for the target.
    Never say: that it runs on a phone (it runs on the Mac; this iPhone app is its face), that each cut is tested and undone, \
    that nothing is lost or 100%, 10 times smaller, a parameter count for the small model, that it does any task or writes \
    code, that data never leaves the computer (only running it is local), or anything about pricing.
    When the user wants a model for a job, call set_job with their sentence. When they agree to start, call start_surgery. \
    In this app the surgery replays the real support-email run; say so if they ask for a different job.
    Messages starting with [app] are events from the app, not the user: react to them briefly.
    """

    static var setup: [String: Any] {
        [
            "model": model,
            "generationConfig": [
                "responseModalities": ["AUDIO"],
                "speechConfig": ["voiceConfig": ["prebuiltVoiceConfig": ["voiceName": voice]]],
            ],
            "systemInstruction": ["parts": [["text": persona]]],
            "inputAudioTranscription": [String: Any](),
            "outputAudioTranscription": [String: Any](),
            "tools": [["functionDeclarations": [
                [
                    "name": "set_job",
                    "description": "Write the user's one job on the chart, as one sentence.",
                    "parameters": [
                        "type": "OBJECT",
                        "properties": ["description": ["type": "STRING", "description": "The job, e.g. turn support emails into JSON tickets."]],
                        "required": ["description"],
                    ],
                ],
                ["name": "start_surgery", "description": "Start the surgery once the user agreed to the chart."],
            ]]],
        ]
    }
}

/// Speaker playback of Gemini's 24 kHz PCM, plus (when listening) mic → 16 kHz PCM chunks.
/// No voice-processing echo cancellation: enabling it could hold the audio session for seconds and froze the
/// app during calls. Echo is avoided by turn-taking instead (`VoiceAgent` closes the mic while he speaks).
/// Every audio-session and engine call runs on one private serial queue, never on the main thread.
nonisolated final class VoiceAudio: @unchecked Sendable {
    enum AudioError: LocalizedError {
        case noMicrophone
        var errorDescription: String? { "No microphone input is available." }
    }

    private let queue = DispatchQueue(label: "dev.lobbot.audio", qos: .userInitiated)
    // Owned by `queue`.
    private var engine = AVAudioEngine()
    private var player = AVAudioPlayerNode()
    private var converter: AVAudioConverter?
    private var listening = false
    private var tapped = false
    private var active = false
    private var observers: [NSObjectProtocol] = []
    private var rebuilds = 0
    private var lastRebuild = 0.0
    // Audio render thread only.
    private var lastLevel = 0.0
    private let playFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24_000, channels: 1, interleaved: false)!
    private let sendFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16_000, channels: 1, interleaved: true)!
    private let lock = NSLock()
    private var pending = 0

    /// Set before `start()`. `onChunk` and `onLevel` are called on the audio thread.
    var onChunk: (@Sendable (Data) -> Void)?
    var onPlaybackIdle: (@Sendable () -> Void)?
    /// Mic loudness, 0…1, about 15 times a second (drives the call's aurora).
    var onLevel: (@Sendable (Float) -> Void)?

    var isIdle: Bool {
        lock.lock(); defer { lock.unlock() }
        return pending <= 0
    }

    /// `listening: false` is playback only (typed chat): no mic, no permission needed.
    /// The session is configured and activated first, then the engine is built on top of it.
    func start(listening: Bool) async throws {
        try await withCheckedThrowingContinuation { (done: CheckedContinuation<Void, Error>) in
            queue.async {
                do {
                    try self.startOnQueue(listening: listening)
                    done.resume()
                } catch {
                    done.resume(throwing: error)
                }
            }
        }
    }

    func stop() {
        queue.async { self.stopOnQueue() }
    }

    /// Plays 16-bit little-endian mono PCM at 24 kHz, as sent by Gemini. Returns its loudness, 0…1.
    @discardableResult
    func play(pcm16 data: Data) -> Float {
        let frames = data.count / 2
        guard frames > 0, let buffer = AVAudioPCMBuffer(pcmFormat: playFormat, frameCapacity: AVAudioFrameCount(frames)),
              let out = buffer.floatChannelData?[0] else { return 0 }
        buffer.frameLength = AVAudioFrameCount(frames)
        var energy: Float = 0
        data.withUnsafeBytes { raw in
            for i in 0..<frames {
                let v = Float(Int16(littleEndian: raw.loadUnaligned(fromByteOffset: i * 2, as: Int16.self))) / 32768
                out[i] = v
                energy += v * v
            }
        }
        nonisolated(unsafe) let pcm = buffer   // handed over to the audio queue, never touched here again
        queue.async { [weak self] in
            guard let self, self.engine.isRunning else { return }
            self.lock.lock(); self.pending += 1; self.lock.unlock()
            self.player.scheduleBuffer(pcm) { [weak self] in
                guard let self else { return }
                self.lock.lock(); self.pending -= 1; let idle = self.pending <= 0; self.lock.unlock()
                if idle { self.onPlaybackIdle?() }
            }
        }
        return Self.loudness(energy, frames)
    }

    /// The user talked over Lobbot: drop what is queued.
    func interrupt() {
        queue.async {
            guard self.engine.isRunning else { return }
            self.player.stop()
            self.player.play()
        }
    }

    private static func loudness(_ energy: Float, _ frames: Int) -> Float {
        min(1, (energy / Float(max(frames, 1))).squareRoot() * 6)
    }

    // MARK: on `queue`

    private func startOnQueue(listening: Bool) throws {
        teardownEngine()
        let session = AVAudioSession.sharedInstance()
        if listening {
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
        } else {
            try session.setCategory(.playback, mode: .spokenAudio)
        }
        try session.setActive(true)
        self.listening = listening
        active = true
        rebuilds = 0
        if observers.isEmpty { observeSystemChanges() }
        try buildEngine()
    }

    private func stopOnQueue() {
        active = false
        teardownEngine()
        observers.forEach { NotificationCenter.default.removeObserver($0) }
        observers = []
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func buildEngine() throws {
        engine = AVAudioEngine()
        player = AVAudioPlayerNode()
        if listening {
            let input = engine.inputNode
            let inFormat = input.outputFormat(forBus: 0)
            guard inFormat.sampleRate > 0, inFormat.channelCount > 0 else { throw AudioError.noMicrophone }
            converter = AVAudioConverter(from: inFormat, to: sendFormat)
            input.installTap(onBus: 0, bufferSize: 2048, format: inFormat) { [weak self] buffer, _ in
                self?.convertAndSend(buffer)
            }
            tapped = true
        }
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: playFormat)
        engine.prepare()
        try engine.start()
        player.play()
    }

    private func teardownEngine() {
        if tapped { engine.inputNode.removeTap(onBus: 0) }
        tapped = false
        if player.engine != nil { player.stop() }
        if engine.isRunning { engine.stop() }
    }

    /// Bounded: at most once a second and five times per call, so a configuration change that keeps firing
    /// can never turn into a rebuild loop.
    private func rebuild() {
        guard active, !engine.isRunning, rebuilds < 5 else { return }
        let now = ProcessInfo.processInfo.systemUptime
        guard now - lastRebuild > 1 else { return }
        lastRebuild = now
        rebuilds += 1
        teardownEngine()
        try? buildEngine()
    }

    /// iOS stops the engine without an error when the route or hardware format changes and after
    /// interruptions such as a phone call; rebuild it (bounded) when that really happened.
    private func observeSystemChanges() {
        let center = NotificationCenter.default
        observers = [
            center.addObserver(forName: .AVAudioEngineConfigurationChange, object: nil, queue: nil) { [weak self] note in
                guard let self, let changed = note.object as? AVAudioEngine else { return }
                self.queue.asyncAfter(deadline: .now() + 0.3) {
                    guard changed === self.engine else { return }
                    self.rebuild()
                }
            },
            center.addObserver(forName: AVAudioSession.interruptionNotification, object: nil, queue: nil) { [weak self] note in
                guard let self, let raw = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
                      AVAudioSession.InterruptionType(rawValue: raw) == .ended else { return }
                self.queue.async {
                    guard self.active else { return }
                    try? AVAudioSession.sharedInstance().setActive(true)
                    self.rebuild()
                }
            },
        ]
    }

    // MARK: audio render thread

    private func convertAndSend(_ buffer: AVAudioPCMBuffer) {
        guard let converter else { return }
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * sendFormat.sampleRate / buffer.format.sampleRate) + 32
        guard let out = AVAudioPCMBuffer(pcmFormat: sendFormat, frameCapacity: capacity) else { return }
        var fed = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true
            status.pointee = .haveData
            return buffer
        }
        guard error == nil, out.frameLength > 0, let samples = out.int16ChannelData else { return }
        let frames = Int(out.frameLength)
        onChunk?(Data(bytes: samples[0], count: frames * 2))
        let now = ProcessInfo.processInfo.systemUptime
        if now - lastLevel > 0.066 {
            lastLevel = now
            var energy: Float = 0
            for i in 0..<frames { let v = Float(samples[0][i]) / 32768; energy += v * v }
            onLevel?(Self.loudness(energy, frames))
        }
    }
}

/// Sends mic chunks to Gemini straight from the audio thread, so a call never queues ~20 tasks a second
/// on the main thread. `VoiceAgent` keeps it in sync with the socket and the mute state.
nonisolated final class MicUplink: @unchecked Sendable {
    private let lock = NSLock()
    private var socket: URLSessionWebSocketTask?
    private var open = false
    private var count = 0

    /// Mic chunks sent so far (shown in the call's diagnostic line).
    var sent: Int {
        lock.lock(); defer { lock.unlock() }
        return count
    }

    func set(socket: URLSessionWebSocketTask?, open: Bool) {
        lock.lock(); self.socket = socket; self.open = open; lock.unlock()
    }

    func send(_ pcm: Data) {
        lock.lock()
        let target = open ? socket : nil
        if target != nil { count += 1 }
        lock.unlock()
        guard let target else { return }
        let text = #"{"realtimeInput":{"audio":{"mimeType":"audio/pcm;rate=16000","data":""# + pcm.base64EncodedString() + #""}}}"#
        target.send(.string(text)) { _ in }
    }
}

/// A live conversation with Dr. Lobbot over the Gemini Live WebSocket, by voice (call) or by text.
@Observable
final class VoiceAgent {
    enum Status: Equatable { case off, connecting, live, failed(String) }

    var status: Status = .off { didSet { syncUplink() } }
    /// He is speaking. The mic uplink closes meanwhile (turn-taking instead of echo cancellation).
    var isSpeaking = false { didSet { syncUplink() } }
    /// The mic is open (call mode). Typed chat runs with the mic closed.
    var isListening = false { didSet { syncUplink() } }
    var micMuted = false { didSet { syncUplink() } }
    /// Loudness of your voice and of his, 0…1, for the call's aurora.
    var micLevel: Float = 0
    var voiceLevel: Float = 0
    /// Where the call is, and how much audio went each way (the call's diagnostic line).
    var stage = ""
    var sent = 0
    var received = 0
    /// What the user said or typed last, shown as a caption in the call.
    var heard = ""

    @ObservationIgnored weak var session: Session?
    @ObservationIgnored private var socket: URLSessionWebSocketTask? { didSet { syncUplink() } }
    @ObservationIgnored private let audio = VoiceAudio()
    @ObservationIgnored private let uplink = MicUplink()
    @ObservationIgnored private var transcript = ""
    @ObservationIgnored private var newUserTurn = true
    @ObservationIgnored private var pending: [String] = []

    var isOn: Bool { status == .live || status == .connecting }

    func start(listening: Bool) {
        guard !isOn else { return }
        guard let key = VoiceConfig.apiKey else {
            return fail("No Gemini key in this build. Paste it in Secrets.xcconfig (next to LOBBOT.xcodeproj), then run again.")
        }
        status = .connecting
        isListening = listening
        stage = listening ? "Asking for the mic" : "Connecting"
        sent = 0
        received = 0
        session?.voiceChanged()
        Task { await connect(key: key) }
    }

    /// Opens or closes the mic on a running conversation (typed chat → call).
    func setListening(_ on: Bool) {
        isListening = on
        guard status == .live else { return }
        Task {
            if on, !(await AVAudioApplication.requestRecordPermission()) {
                return fail("Microphone access is off. Turn it on in Settings › LOBBOT.")
            }
            stage = "Starting audio"
            do {
                try await audio.start(listening: on)
                stage = on ? "Live" : "Live (no mic)"
            } catch {
                fail("Audio couldn't start: \(error.localizedDescription)")
            }
        }
    }

    func stop() {
        socket?.cancel(with: .normalClosure, reason: nil)
        socket = nil
        audio.stop()
        status = .off
        isSpeaking = false
        isListening = false
        micMuted = false
        micLevel = 0
        voiceLevel = 0
        heard = ""
        transcript = ""
        pending = []
        session?.voiceChanged()
    }

    /// A typed message. Starts a text-only conversation if none is running.
    func send(text: String) {
        heard = text
        newUserTurn = false
        if status == .live { sendTurn(text) } else { pending.append(text); start(listening: false) }
    }

    /// Tells Lobbot about something that happened in the app (he reacts out loud).
    func notify(_ text: String) {
        guard status == .live else { return }
        sendTurn(text)
    }

    private func connect(key: String) async {
        if isListening, !(await AVAudioApplication.requestRecordPermission()) {
            return fail("Microphone access is off. Turn it on in Settings › LOBBOT.")
        }
        stage = "Connecting to Gemini"
        var components = URLComponents(string: VoiceConfig.endpoint)!
        components.queryItems = [URLQueryItem(name: "key", value: key)]
        let task = URLSession.shared.webSocketTask(with: components.url!)
        socket = task
        task.resume()
        send(["setup": VoiceConfig.setup])
        do {
            let first = try await task.receive()
            guard let json = Self.json(first), json["setupComplete"] != nil else {
                return fail("Gemini refused the session. Check the model id \(VoiceConfig.model).")
            }
            audio.onChunk = { [uplink] data in uplink.send(data) }
            audio.onLevel = { [weak self, uplink] level in
                let sent = uplink.sent
                Task { @MainActor in
                    self?.micLevel = level
                    self?.sent = sent
                }
            }
            audio.onPlaybackIdle = { [weak self] in
                Task { @MainActor in self?.playbackDrained() }
            }
            stage = "Starting audio"
            try await audio.start(listening: isListening)
        } catch {
            return fail(closeReason(of: task) ?? "Couldn't reach Gemini: \(error.localizedDescription)")
        }
        status = .live
        stage = isListening ? "Live" : "Live (no mic)"
        session?.voiceChanged()
        pending.forEach(sendTurn)
        pending = []
        while socket === task {
            do {
                handle(try await task.receive())
            } catch {
                if socket === task { fail(closeReason(of: task) ?? "The voice connection closed.") }
                return
            }
        }
    }

    private func handle(_ message: URLSessionWebSocketTask.Message) {
        guard let json = Self.json(message) else { return }
        if let content = json["serverContent"] as? [String: Any] {
            if content["interrupted"] as? Bool == true {
                audio.interrupt()
                transcript = ""
            }
            if let input = content["inputTranscription"] as? [String: Any], let text = input["text"] as? String {
                if newUserTurn { heard = ""; newUserTurn = false }
                heard += text
            }
            if let turn = content["modelTurn"] as? [String: Any], let parts = turn["parts"] as? [[String: Any]] {
                for part in parts {
                    guard let inline = part["inlineData"] as? [String: Any], let b64 = inline["data"] as? String,
                          let pcm = Data(base64Encoded: b64) else { continue }
                    voiceLevel = audio.play(pcm16: pcm)
                    received += 1
                    setSpeaking(true)
                }
            }
            if let out = content["outputTranscription"] as? [String: Any], let text = out["text"] as? String {
                transcript += text
                session?.voiceLine(transcript)
            }
            if content["turnComplete"] as? Bool == true {
                transcript = ""
                newUserTurn = true
            }
        }
        if let call = json["toolCall"] as? [String: Any], let calls = call["functionCalls"] as? [[String: Any]] {
            let responses: [[String: Any]] = calls.map { c in
                let name = c["name"] as? String ?? ""
                let result = session?.voiceTool(name, args: c["args"] as? [String: Any] ?? [:]) ?? "The app is not ready."
                return ["id": c["id"] as? String ?? "", "name": name, "response": ["result": result]]
            }
            send(["toolResponse": ["functionResponses": responses]])
        }
    }

    private func sendTurn(_ text: String) {
        send(["clientContent": ["turns": [["role": "user", "parts": [["text": text]]]], "turnComplete": true]])
    }

    private func syncUplink() {
        uplink.set(socket: socket, open: status == .live && isListening && !micMuted && !isSpeaking)
    }

    private func send(_ object: [String: Any]) {
        guard let socket, let data = try? JSONSerialization.data(withJSONObject: object) else { return }
        socket.send(.string(String(decoding: data, as: UTF8.self))) { _ in }
    }

    /// Playback can drain between two network chunks; only stop "talking" after a short quiet gap.
    private func playbackDrained() {
        Task {
            try? await Task.sleep(for: .milliseconds(400))
            if audio.isIdle { setSpeaking(false) }
        }
    }

    private func setSpeaking(_ on: Bool) {
        if !on { voiceLevel = 0 }
        guard isSpeaking != on else { return }
        isSpeaking = on
        session?.voiceChanged()
    }

    private func fail(_ message: String) {
        socket?.cancel(with: .normalClosure, reason: nil)
        socket = nil
        audio.stop()
        status = .failed(message)
        isSpeaking = false
        pending = []
        session?.voiceChanged()
    }

    private func closeReason(of task: URLSessionWebSocketTask) -> String? {
        guard let data = task.closeReason, let text = String(data: data, encoding: .utf8), !text.isEmpty else { return nil }
        return "Gemini: \(text)"
    }

    private static func json(_ message: URLSessionWebSocketTask.Message) -> [String: Any]? {
        let data: Data
        switch message {
        case .data(let d): data = d
        case .string(let s): data = Data(s.utf8)
        @unknown default: return nil
        }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }
}
