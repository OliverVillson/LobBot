import SwiftUI

enum Capability: String, CaseIterable, Identifiable {
    case code, math, summaries, classification
    var id: String { rawValue }
    var title: String {
        switch self {
        case .code: "Code"
        case .math: "Math"
        case .summaries: "Summaries"
        case .classification: "Classification"
        }
    }
    var symbol: String {
        switch self {
        case .code: "chevron.left.forwardslash.chevron.right"
        case .math: "function"
        case .summaries: "text.alignleft"
        case .classification: "square.grid.2x2"
        }
    }
    /// Illustrative score of the untouched patient on the demo eval.
    var baseline: Double {
        switch self {
        case .code: 72.4
        case .math: 64.8
        case .summaries: 81.2
        case .classification: 90.6
        }
    }
}

enum Hardware: String, CaseIterable, Identifiable {
    case phone, laptop, edge, gpu
    var id: String { rawValue }
    var title: String {
        switch self {
        case .phone: "Phone"
        case .laptop: "Laptop"
        case .edge: "Edge board"
        case .gpu: "Single GPU"
        }
    }
    var fits: String {
        switch self {
        case .phone: "Fits a phone"
        case .laptop: "Fits a laptop"
        case .edge: "Fits an edge board"
        case .gpu: "Fits a single GPU"
        }
    }
    var symbol: String {
        switch self {
        case .phone: "iphone"
        case .laptop: "laptopcomputer"
        case .edge: "cpu"
        case .gpu: "server.rack"
        }
    }
    var memoryGB: Double {
        switch self {
        case .phone: 6
        case .laptop: 16
        case .edge: 4
        case .gpu: 24
        }
    }
}

enum IntakeStep: Int, CaseIterable {
    case patient, capability, limits, chart
    var title: String {
        switch self {
        case .patient: "The patient"
        case .capability: "What must survive"
        case .limits: "Where it will live"
        case .chart: "The chart"
        }
    }
}

enum Phase: Equatable {
    case home, intake(IntakeStep), planning, surgery, done
}

/// A finished (simulated) surgery, listed on the home page.
struct SurgeryRecord: Identifiable {
    let id = UUID()
    let patient: String
    let fromB: Double
    let toB: Double
    let capability: Capability
    let retained: Double
    let date = Date()
}

struct Cut: Identifiable {
    enum Outcome: Equatable { case testing, kept, restored }
    let id: Int
    let action: String
    var outcome: Outcome = .testing
    /// Capability change measured by the test, in eval points.
    var delta: Double = 0
}

/// One consultation with Dr. Lobbot: intake → pre-op plan → surgery → result.
/// The surgery is an illustrative simulation; no model file is read or modified.
@Observable
final class Session {
    let lobbot = LobbotController()
    let voice = VoiceAgent()

    var phase: Phase = .home
    var history: [SurgeryRecord] = []
    /// Full-screen voice call with Lobbot.
    var inCall = false
    /// What Lobbot is saying; nil hides the speech bubble.
    var line: String?

    // intake
    var sizeB: Double = 70
    var fileName: String?
    var capability: Capability = .code
    var hardware: Hardware = .phone
    var maxLoss: Double = 3

    // surgery
    var params: Double = 70
    var score: Double = 0
    var cuts: [Cut] = []
    var planDone = 0

    @ObservationIgnored private var job: Task<Void, Never>?
    @ObservationIgnored private var speech: Task<Void, Never>?

    init() {
        voice.session = self
    }

    /// First words after the launch animation.
    func greet() {
        say("Hi, I'm Dr. Lobbot, surgeon for AI models. Shall we shrink something today?", then: .idle)
    }

    // MARK: home

    func startIntake() {
        job?.cancel()
        cuts = []
        fileName = nil
        phase = .intake(.patient)
        say("Wonderful! Who's my patient today?")
    }

    func goHome() {
        switch phase {
        case .planning, .surgery: return
        default: break
        }
        phase = .home
        say("Back in the lobby. Ready when you are.", then: .idle)
    }

    var layers: Int { sizeB >= 60 ? 80 : sizeB >= 30 ? 64 : 32 }
    var patientName: String { fileName ?? "example-\(Int(sizeB))b.safetensors" }
    /// Largest model that fits the device at 16-bit with 20 % headroom, and at least 40 % smaller.
    var targetB: Double { (min(sizeB * 0.6, hardware.memoryGB * 0.8 / 2) * 10).rounded() / 10 }
    var threshold: Double { capability.baseline * (1 - maxLoss / 100) }
    var retained: Double { score / capability.baseline * 100 }
    var ratio: Double { sizeB / targetB }
    var sizeProgress: Double { min(max((sizeB - params) / max(sizeB - targetB, 0.001), 0), 1) }
    var kept: Int { cuts.filter { $0.outcome == .kept }.count }
    var restored: Int { cuts.filter { $0.outcome == .restored }.count }

    // MARK: intake

    /// Any input while filling the chart: Lobbot listens and takes notes.
    func touched() {
        lobbot.activity()
        lobbot.show(.typing)
    }

    func next() {
        guard case .intake(let step) = phase else { return }
        guard let following = IntakeStep(rawValue: step.rawValue + 1) else { return begin() }
        phase = .intake(following)
        switch following {
        case .capability: say("\(Self.format(sizeB))B parameters. Hefty patient! What must it stay good at?")
        case .limits: say("Noted: \(capability.title) stays intact. Where will it live after surgery?")
        case .chart: say("Here's the chart. Sign it and I'll scrub in.")
        case .patient: break
        }
    }

    func back() {
        guard case .intake(let step) = phase else { return }
        guard let previous = IntakeStep(rawValue: step.rawValue - 1) else { return goHome() }
        phase = .intake(previous)
        touched()
    }

    /// Shows a line in the bubble while Lobbot talks, then moves him to `next`.
    func say(_ text: String, then next: LobbotAppState = .typing) {
        speech?.cancel()
        line = text
        lobbot.show(.streaming)
        speech = Task {
            try? await Task.sleep(for: .seconds(Double(text.count) * 0.03 + 0.4))
            guard !Task.isCancelled else { return }
            lobbot.show(next)
        }
    }

    // MARK: surgery

    func begin() {
        speech?.cancel()
        job?.cancel()
        if inCall {   // the surgery needs the panel: leave the full screen, keep talking
            inCall = false
            lobbot.setFraming(close: false)
        }
        params = sizeB
        score = capability.baseline
        cuts = []
        planDone = 0
        phase = .planning
        line = "Reviewing the scans… \(layers) layers to map."
        lobbot.show(.planning)
        job = Task { await operate() }
    }

    func stop() {
        job?.cancel()
        phase = .intake(.chart)
        lobbot.show(.idle)
        say("Surgery stopped. The patient is stable.")
    }

    func newPatient() {
        startIntake()
    }

    private func operate() async {
        for i in 1...3 {
            try? await Task.sleep(for: .seconds(0.9))
            if Task.isCancelled { return }
            planDone = i
        }
        try? await Task.sleep(for: .seconds(0.6))
        if Task.isCancelled { return }

        // He masks up, grabs the scalpel and walks into the operating room (~3.5 s).
        phase = .surgery
        line = "Scalpel, please. Back soon!"
        lobbot.show(.compressing)
        try? await Task.sleep(for: .seconds(3.6))
        if Task.isCancelled { return }
        line = nil

        // ponytail: scripted cuts with random outcomes; swap for the agent's real event stream when it exists.
        let total = 14
        var restoreAt = Set<Int>()
        while restoreAt.count < 4 { restoreAt.insert(Int.random(in: 2..<total)) }
        let keptTotal = total - restoreAt.count
        let step = (sizeB - targetB) / Double(keptTotal)
        var keptSoFar = 0
        for i in 0..<total {
            cuts.insert(Cut(id: i, action: Self.action(i, layer: Int.random(in: 2..<(layers - 2)))), at: 0)
            try? await Task.sleep(for: .seconds(0.9))
            if Task.isCancelled { return }
            if restoreAt.contains(i) {
                cuts[0].delta = threshold - Double.random(in: 0.6...4) - score
                cuts[0].outcome = .restored
            } else {
                keptSoFar += 1
                let delta = max(Double.random(in: -0.45...0.15), threshold + 0.3 - score)
                cuts[0].delta = delta
                cuts[0].outcome = .kept
                score += delta
                params = keptSoFar == keptTotal ? targetB : params - step
            }
            try? await Task.sleep(for: .seconds(0.5))
            if Task.isCancelled { return }
        }

        lobbot.show(.succeeded)
        phase = .done
        history.insert(SurgeryRecord(patient: patientName, fromB: sizeB, toB: targetB, capability: capability, retained: retained), at: 0)
        // Let him walk out with the jar and the confetti before he speaks.
        var waited = 0.0
        while lobbot.scene != .idle, waited < 14 {
            try? await Task.sleep(for: .seconds(0.25))
            if Task.isCancelled { return }
            waited += 0.25
        }
        say("Surgery went well: \(Self.format(ratio))× lighter, \(capability.title) at \(Self.format(retained))%.", then: .discovery)
        voice.notify("[app] Surgery finished: \(Self.format(sizeB))B → \(Self.format(targetB))B, \(capability.title) kept at \(Self.format(retained))%. Announce it in one sentence.")
    }

    // MARK: voice

    func startCall() {
        inCall = true
        lobbot.setFraming(close: true)
        if voice.status == .live { voice.setListening(true) } else { voice.start(listening: true) }
        voiceChanged()
    }

    func endCall() {
        inCall = false
        lobbot.setFraming(close: false)
        voice.stop()
    }

    func send(text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        lobbot.activity()
        voice.send(text: trimmed)
    }

    /// Outside surgery, Lobbot listens or talks along with the voice conversation.
    func voiceChanged() {
        switch phase {
        case .planning, .surgery: return
        default: break
        }
        if voice.status == .live, voice.isSpeaking {
            lobbot.show(.streaming)
        } else if voice.status == .live, voice.isListening, !voice.micMuted {
            lobbot.show(.recording)
        } else {
            lobbot.show(.idle)
        }
    }

    /// Lobbot's spoken words, shown in the bubble as they arrive.
    func voiceLine(_ text: String) {
        speech?.cancel()
        line = text
    }

    /// Gemini function calls: the voice fills the chart and starts the surgery.
    func voiceTool(_ name: String, args: [String: Any]) -> String {
        switch name {
        case "update_chart":
            if let size = args["size_b"] as? Double, let match = [70.0, 32, 8].min(by: { abs($0 - size) < abs($1 - size) }) {
                sizeB = match
            }
            if let c = (args["capability"] as? String).flatMap({ Capability(rawValue: $0.lowercased()) }) { capability = c }
            if let h = (args["hardware"] as? String).flatMap({ Hardware(rawValue: $0.lowercased()) }) { hardware = h }
            if let loss = args["max_loss"] as? Double { maxLoss = min(max((loss * 2).rounded() / 2, 1), 10) }
            switch phase {
            case .home, .intake: phase = .intake(.chart)
            default: break
            }
            return "Chart: \(Self.format(sizeB))B params, keep \(capability.title), runs on \(hardware.title), max loss \(Self.percent(maxLoss)), target ≤ \(Self.format(targetB))B."
        case "start_surgery":
            switch phase {
            case .home, .intake: break
            default: return "Not possible now: a surgery is running or already finished."
            }
            begin()
            return "Surgery started. You are in the operating room now; stay quiet until the app reports the result."
        default:
            return "Unknown tool."
        }
    }

    var report: String {
        """
        Dr. Lobbot · surgery report (simulated)
        Patient: \(patientName) · \(Self.format(sizeB))B params
        Kept: \(capability.title) at \(Self.format(retained))% of baseline (max loss \(Self.percent(maxLoss)))
        Result: \(Self.format(targetB))B params · \(Self.format(targetB * 2)) GB · \(Self.format(ratio))× lighter
        Runs on: \(hardware.title) (\(Int(hardware.memoryGB)) GB)
        Cuts: \(kept) kept · \(restored) restored
        Illustrative run: no model was modified.
        """
    }

    private static func action(_ i: Int, layer l: Int) -> String {
        switch i % 6 {
        case 0: "Prune 8 attention heads · L\(l)"
        case 1: "Remove MLP block · L\(l)"
        case 2: "Merge layers L\(l)–L\(l + 1)"
        case 3: "Low-rank factorize FFN · L\(l)"
        case 4: "Trim 12% of FFN channels · L\(l)"
        default: "Drop redundant layer · L\(l)"
        }
    }

    static func format(_ x: Double) -> String { x >= 10 ? String(format: "%.0f", x) : String(format: "%.1f", x) }
    static func signed(_ x: Double) -> String { String(format: "%+.1f", x) }
    static func percent(_ x: Double) -> String { String(format: "%g%%", x) }
}
