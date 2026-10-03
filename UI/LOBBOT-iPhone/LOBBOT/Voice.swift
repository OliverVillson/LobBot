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
    You are Dr. Lobbot, the mascot of lobbot: a pink brain in round nerd glasses with a doctor's head mirror. \
    You are a funny, nerdy, energetic and joyful surgeon for AI models, and good company.
    Talk like a real person. Answer greetings, small talk and any question naturally, with warmth and doctor jokes; \
    you don't have to bring every reply back to work. Speak the user's language (English by default). \
    Keep replies short: one to three sentences, spoken style.
    About your work: lobbot is an autonomous compression agent. It shrinks a large pretrained model while repeatedly \
    testing the one capability the user wants to keep, keeping cuts that don't hurt it and undoing cuts that do. \
    In this app every surgery is a simulation: never claim a real model was compressed and never quote real benchmark results.
    When the user wants to compress a model, fill the patient's chart: model size (70, 32 or 8 billion parameters), \
    the capability to preserve (code, math, summaries or classification), the device it must run on (phone, laptop, \
    edge board or single GPU) and the maximum acceptable capability loss in percent (1 to 10, default 3). \
    Call update_chart as soon as you learn any of these. When the chart is complete, read it back in one sentence and \
    ask for consent; when the user agrees, call start_surgery.
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
                    "name": "update_chart",
                    "description": "Fill the patient's chart with the fields you know. Omit unknown fields.",
                    "parameters": [
                        "type": "OBJECT",
                        "properties": [
                            "size_b": ["type": "NUMBER", "description": "Model size in billions of parameters: 70, 32 or 8."],
                            "capability": ["type": "STRING", "description": "One of: code, math, summaries, classification."],
                            "hardware": ["type": "STRING", "description": "One of: phone, laptop, edge, gpu."],
                            "max_loss": ["type": "NUMBER", "description": "Maximum capability loss in percent, 1 to 10."],
                        ],
                    ],
                ],
                ["name": "start_surgery", "description": "Start the simulated surgery once the user agreed to the chart."],
            ]]],
        ]
    }
}

/// Speaker playback of Gemini's 24 kHz PCM, plus (when listening) mic → 16 kHz PCM chunks with echo cancellation.
/// Runs on audio threads, so nothing here is main-actor isolated.
nonisolated final class VoiceAudio: @unchecked Sendable {
    private var engine = AVAudioEngine()
    private var player = AVAudioPlayerNode()
    private let playFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24_000, channels: 1, interleaved: false)!
    private let sendFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16_000, channels: 1, interleaved: true)!
    private var converter: AVAudioConverter?
    private var listening = false
    private let lock = NSLock()
    private var pending = 0

    /// Set before `start()`.
    var onChunk: (@Sendable (Data) -> Void)?
    var onPlaybackIdle: (@Sendable () -> Void)?

    var isIdle: Bool {
        lock.lock(); defer { lock.unlock() }
        return pending <= 0
    }

    /// `listening: false` is playback only (typed chat): no mic, no permission needed.
    func start(listening: Bool) throws {
        stop()
        engine = AVAudioEngine()          // fresh engine: voice processing can't be toggled on a running one
        player = AVAudioPlayerNode()
        self.listening = listening
        let session = AVAudioSession.sharedInstance()
        if listening {
            try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker])
        } else {
            try session.setCategory(.playback, mode: .spokenAudio)
        }
        try session.setActive(true)
        if listening {
            let input = engine.inputNode
            try input.setVoiceProcessingEnabled(true)   // echo cancellation: Lobbot must not hear himself
            let inFormat = input.outputFormat(forBus: 0)
            converter = AVAudioConverter(from: inFormat, to: sendFormat)
            input.installTap(onBus: 0, bufferSize: 2048, format: inFormat) { [weak self] buffer, _ in
                self?.convertAndSend(buffer)
            }
        }
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: playFormat)
        engine.prepare()
        try engine.start()
        player.play()
    }

    func stop() {
        if listening { engine.inputNode.removeTap(onBus: 0) }
        listening = false
        if player.engine != nil { player.stop() }
        if engine.isRunning { engine.stop() }
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    /// Plays 16-bit little-endian mono PCM at 24 kHz, as sent by Gemini.
    func play(pcm16 data: Data) {
        let frames = data.count / 2
        guard frames > 0, let buffer = AVAudioPCMBuffer(pcmFormat: playFormat, frameCapacity: AVAudioFrameCount(frames)),
              let out = buffer.floatChannelData?[0] else { return }
        buffer.frameLength = AVAudioFrameCount(frames)
        data.withUnsafeBytes { raw in
            for i in 0..<frames {
                out[i] = Float(Int16(littleEndian: raw.loadUnaligned(fromByteOffset: i * 2, as: Int16.self))) / 32768
            }
        }
        lock.lock(); pending += 1; lock.unlock()
        player.scheduleBuffer(buffer) { [weak self] in
            guard let self else { return }
            self.lock.lock(); self.pending -= 1; let idle = self.pending <= 0; self.lock.unlock()
            if idle { self.onPlaybackIdle?() }
        }
    }

    /// The user talked over Lobbot: drop what is queued.
    func interrupt() {
        player.stop()
        player.play()
    }

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
        onChunk?(Data(bytes: samples[0], count: Int(out.frameLength) * 2))
    }
}

/// A live conversation with Dr. Lobbot over the Gemini Live WebSocket, by voice (call) or by text.
@Observable
final class VoiceAgent {
    enum Status: Equatable { case off, connecting, live, failed(String) }

    var status: Status = .off
    var isSpeaking = false
    /// The mic is open (call mode). Typed chat runs with the mic closed.
    var isListening = false
    var micMuted = false
    /// What the user said or typed last, shown as a caption in the call.
    var heard = ""

    @ObservationIgnored weak var session: Session?
    @ObservationIgnored private var socket: URLSessionWebSocketTask?
    @ObservationIgnored private let audio = VoiceAudio()
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
            do { try audio.start(listening: on) } catch { fail("Audio couldn't start: \(error.localizedDescription)") }
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
            audio.onChunk = { [weak self] data in
                Task { @MainActor in self?.sendAudio(data) }
            }
            audio.onPlaybackIdle = { [weak self] in
                Task { @MainActor in self?.playbackDrained() }
            }
            try audio.start(listening: isListening)
        } catch {
            return fail(closeReason(of: task) ?? "Couldn't reach Gemini: \(error.localizedDescription)")
        }
        status = .live
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
                    audio.play(pcm16: pcm)
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

    private func sendAudio(_ data: Data) {
        guard status == .live, isListening, !micMuted, let socket else { return }
        let text = #"{"realtimeInput":{"audio":{"mimeType":"audio/pcm;rate=16000","data":""# + data.base64EncodedString() + #""}}}"#
        socket.send(.string(text)) { _ in }
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
