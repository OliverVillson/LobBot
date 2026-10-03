import SwiftUI

// MARK: - The real pipeline

/// The six pipeline stages, in the backend's order (`STAGES` in `lobbot/app.py`).
enum Stage: String, CaseIterable, Identifiable {
    case data, reap, heal, quantize, eval, package
    var id: String { rawValue }

    /// Dr. Lobbot's name for the stage.
    var title: String {
        switch self {
        case .data: "Practice cases"
        case .reap: "The operation"
        case .heal: "Physio"
        case .quantize: "Packing"
        case .eval: "Check-up"
        case .package: "Discharge papers"
        }
    }
    /// Shown while the stage runs.
    var working: String {
        switch self {
        case .data: "The big model is doing your job a few thousand times…"
        case .reap: "Scoring the experts on your job, removing the half it uses least…"
        case .heal: "Relearning from the big model's answers…"
        case .quantize: "Choosing the bits for each layer…"
        case .eval: "Gemini is grading both models on new emails…"
        case .package: "Writing the model file…"
        }
    }
    var symbol: String {
        switch self {
        case .data: "doc.on.doc"
        case .reap: "scissors"
        case .heal: "figure.walk"
        case .quantize: "suitcase"
        case .eval: "stethoscope"
        case .package: "shippingbox"
        }
    }
}

enum StageStatus: String { case pending, running, done, error, skipped }

struct StageRow: Identifiable {
    let stage: Stage
    var status: StageStatus = .pending
    var pct: Double = 0
    var msg = ""
    /// Time spent in the stage, in seconds of real GPU time.
    var seconds: Double = 0
    var id: String { stage.rawValue }
}

/// One progress event, the same shape as the backend's stream: `{stage, status, pct, msg}`.
struct StageEvent {
    let stage: Stage
    let status: StageStatus
    var pct: Double? = nil
    var msg: String? = nil
    var seconds: Double? = nil
}

/// The quality run of 2026-10-03 (`support-email-to-ticket`, job `quality`), as reported by
/// `lobbot results quality`. The app replays it until it is wired to the backend's live stream.
/// Numbers and wording follow the motion-ad brief's fact sheet; don't round them differently.
enum QualityRun {
    static let date = "Oct 3, 2026"
    static let jobName = "support-email-to-ticket"
    static let demoJob = "Turn support emails into JSON tickets"
    static let description = "Turn a customer support email into a structured JSON ticket for a SaaS helpdesk."
    static let input = "The free-text body of one customer email."
    static let output = "One JSON object: category, priority, summary (one sentence), customer_sentiment."
    static let grading = "Valid JSON with exactly the four keys; category and priority match the email; the summary is faithful, specific and at most 25 words; the sentiment matches the tone."
    static let seeds = 15
    static let exampleIn = "Hi, I was charged twice for my Pro subscription this month. Can you refund the extra charge? Order #48213."
    static let exampleOut = #"{"category": "billing", "priority": "high", "summary": "Customer was double-charged for the Pro subscription this month and requests a refund for order #48213.", "customer_sentiment": "negative"}"#

    static let teacher = "Qwen3-30B-A3B"
    static let teacherGB = "61 GB"
    static let modelGB = "6.52 GB"
    static let score = "9.7/10"
    static let teacherScore = "9.8/10"
    static let share = "99%"
    static let tests = 100
    static let trainExamples = "2,030"
    static let speed = "about 50 tokens/s on a 16 GB MacBook"
    static let gpu = "one evroc B200"
    static let gpuTime = "25m26s"

    /// Real stage time (seconds) and the message each stage printed.
    static let stages: [(stage: Stage, seconds: Double, msg: String)] = [
        (.data, 149, "2030 train, 100 held out"),
        (.reap, 519, "kept 64 experts per layer"),
        (.heal, 233, "healed + dense student"),
        (.quantize, 549, "lobbot-moe 6.52 GB, dense 5.3 GB"),
        (.eval, 71, "winner lobbot-moe: 9.7/10 vs teacher 9.8/10"),
        (.package, 5, "lobbot-moe ready at out/model.gguf"),
    ]

    struct Candidate: Identifiable {
        let name: String, size: String, laptop: String, judge: String
        let meetsTarget: Bool?
        let winner: Bool
        var id: String { name }
    }
    static let candidates = [
        Candidate(name: "Qwen3-30B-A3B · teacher", size: "61.1 GB", laptop: "–", judge: "9.8", meetsTarget: nil, winner: false),
        Candidate(name: "lobbot-moe · pruned", size: "6.52 GB", laptop: "49", judge: "9.7", meetsTarget: true, winner: true),
        Candidate(name: "Gemma 4 E4B · backup", size: "5.30 GB", laptop: "26", judge: "9.8", meetsTarget: false, winner: false),
    ]
    static let bits = [("2-bit", 77), ("4-bit", 12), ("3-bit", 8), ("5-bit", 2)]

    static let heroEmail = "helo... cancel subscritpion immediately!! do not charge me again! also why was i billed $49 instead of $29 last month?? wait actually keep my acount active until the 15th becos my accountant needs the reciepts then delete everything thx -Sent from my iPhone"
    static let heroTicket = #"{"category": "billing", "priority": "urgent", "summary": "Customer requests immediate subscription cancellation, questions a $49 charge instead of $29, and asks to keep account active until the 15th for invoice purposes.", "customer_sentiment": "negative"}"#
    static let saveCommand = "lobbot save quality --chat"
    static let chatCommand = "lobbot chat lobbot-support-email-to-ticket"
}

// MARK: - Consultation

enum IntakeStep: Int, CaseIterable {
    case job, chart, target
    var title: String {
        switch self {
        case .job: "The job"
        case .chart: "The chart"
        case .target: "Going home to"
        }
    }
}

enum Phase: Equatable {
    case home, intake(IntakeStep), surgery, done
}

/// A finished surgery, listed on the home page.
struct SurgeryRecord: Identifiable {
    let id = UUID()
    let job: String
    let date = Date()
}

/// One consultation with Dr. Lobbot: the job → the chart → the target → surgery → discharge.
/// Until the backend is wired, the surgery replays the real quality run (`QualityRun`).
@Observable
final class Session {
    let lobbot = LobbotController()
    let voice = VoiceAgent()

    var phase: Phase = .home { didSet { reframe() } }
    /// A voice call with Lobbot is on (mic open). Full screen from the home page, picture-in-picture
    /// during a consultation.
    var inCall = false { didSet { reframe() } }
    var history: [SurgeryRecord] = []
    /// What Lobbot is saying; nil hides the speech bubble.
    var line: String?

    /// The one job, in the user's words.
    var job = ""
    var rows: [StageRow] = Stage.allCases.map { StageRow(stage: $0) }

    @ObservationIgnored private var run: Task<Void, Never>?
    @ObservationIgnored private var speech: Task<Void, Never>?
    @ObservationIgnored private var framedClose: Bool?

    init() {
        voice.session = self
    }

    /// A consultation is on: Lobbot shrinks to a picture-in-picture tile and the panel takes the screen.
    var diagnosing: Bool { phase != .home }
    var fullScreenCall: Bool { inCall && !diagnosing }

    /// Close-up whenever he's in a small tile or a full-screen call; wide when his props need room
    /// (the home stage, and the operating room during surgery).
    private func reframe() {
        let close = phase != .surgery && (diagnosing || inCall)
        guard close != framedClose else { return }
        framedClose = close
        lobbot.setFraming(close: close)
    }

    var jobTitle: String { job.isBlank ? QualityRun.demoJob : job.trimmingCharacters(in: .whitespacesAndNewlines) }
    var gpuSeconds: Double { rows.reduce(0) { $0 + $1.seconds } }
    var progress: Double { rows.reduce(0) { $0 + ($1.status == .done ? 1 : $1.pct / 100) } / Double(rows.count) }

    /// First words after the launch animation.
    func greet() {
        say("Hi, I'm Dr. Lobbot. I shrink big AI models down to the one job you need. What can I do for you?", then: .idle)
    }

    // MARK: home and intake

    func startIntake() {
        run?.cancel()
        job = ""
        phase = .intake(.job)
        say("Wonderful! What's the one job your model should do? One sentence is plenty.")
    }

    func goHome() {
        guard phase != .surgery else { return }
        phase = .home
        say("Back in the lobby. Ready when you are.", then: .idle)
    }

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
        case .chart: say("Here's the chart Gemini drafts: what goes in, what comes out, and how we'll grade it.")
        case .target: say("Where's the patient going home? A 16 GB laptop, and it has to be quick.")
        case .job: break
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
        run?.cancel()
        rows = Stage.allCases.map { StageRow(stage: $0) }
        phase = .surgery
        line = "Practice first: the big model does your job a couple of thousand times."
        lobbot.show(.planning)
        run = Task { await replay() }
    }

    func stop() {
        run?.cancel()
        for i in rows.indices where rows[i].status == .running { rows[i].status = .error; rows[i].msg = "stopped" }
        phase = .intake(.target)
        lobbot.show(.idle)
        say("Surgery stopped. The patient is stable.")
    }

    func newPatient() {
        startIntake()
    }

    /// Feeds one backend event into the board. The live stream will call this too.
    func apply(_ e: StageEvent) {
        guard let i = rows.firstIndex(where: { $0.stage == e.stage }) else { return }
        rows[i].status = e.status
        if let pct = e.pct { rows[i].pct = pct }
        if let msg = e.msg { rows[i].msg = msg }
        if let seconds = e.seconds { rows[i].seconds = seconds }
        if e.status == .done { rows[i].pct = 100 }
    }

    // ponytail: replays the 2026-10-03 quality run, ~25 min of GPU time shown in ~21 s.
    // Swap for the backend's event stream (GET /jobs/{id}/events) feeding `apply(_:)`.
    private func replay() async {
        let shown: [Stage: Double] = [.data: 3, .reap: 5.5, .heal: 3, .quantize: 5, .eval: 3, .package: 1.2]
        for (stage, seconds, msg) in QualityRun.stages {
            if stage == .reap {
                line = "Scalpel, please. Back soon!"
                lobbot.show(.compressing)   // masks up and walks into the operating room
            }
            apply(StageEvent(stage: stage, status: .running, pct: 0, msg: stage.working, seconds: 0))
            let duration = shown[stage] ?? 2
            let ticks = Int(duration / 0.1)
            for t in 1...ticks {
                try? await Task.sleep(for: .milliseconds(100))
                if Task.isCancelled { return }
                let k = Double(t) / Double(ticks)
                apply(StageEvent(stage: stage, status: .running, pct: k * 100, seconds: seconds * k))
            }
            apply(StageEvent(stage: stage, status: .done, msg: msg, seconds: seconds))
            if stage == .reap { line = nil }
        }

        lobbot.show(.succeeded)   // he walks out with the jar and the confetti
        phase = .done
        history.insert(SurgeryRecord(job: jobTitle), at: 0)
        var waited = 0.0
        while lobbot.scene != .idle, waited < 14 {
            try? await Task.sleep(for: .seconds(0.25))
            if Task.isCancelled { return }
            waited += 0.25
        }
        say("The patient is going home: \(QualityRun.modelGB), \(QualityRun.share) of the big model's score.", then: .discovery)
        voice.notify("[app] Surgery finished: 61 GB down to 6.52 GB, scored 9.7/10 against the big model's 9.8/10 on 100 new emails graded by Gemini. Announce it in one sentence.")
    }

    var report: String {
        """
        Dr. Lobbot · surgery report (replay of the \(QualityRun.date) quality run)
        Job: \(QualityRun.description)
        Big model: \(QualityRun.teacher), \(QualityRun.teacherGB)
        LobBot model: \(QualityRun.modelGB) GGUF, about 9× smaller
        Score: \(QualityRun.score) vs \(QualityRun.teacherScore) for the big model (\(QualityRun.share)), on \(QualityRun.tests) new emails graded by Gemini
        Speed: \(QualityRun.speed), runs offline in Ollama
        Built in \(QualityRun.gpuTime) on \(QualityRun.gpu) GPU: \(QualityRun.trainExamples) practice examples, half the experts removed (64 of 128 per layer), 2 to 6 bits per layer
        Take it home: \(QualityRun.saveCommand)
        """
    }

    // MARK: voice

    func startCall() {
        inCall = true
        if voice.status == .live { voice.setListening(true) } else { voice.start(listening: true) }
        voiceChanged()
    }

    func endCall() {
        inCall = false
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
        guard phase != .surgery else { return }
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

    /// Gemini function calls: the voice writes the job down and starts the surgery.
    func voiceTool(_ name: String, args: [String: Any]) -> String {
        switch name {
        case "set_job":
            if let description = args["description"] as? String, !description.isBlank { job = description }
            switch phase {
            case .home, .intake: phase = .intake(.chart)
            default: break
            }
            return "Chart open for: \(jobTitle). In this demo the surgery replays the real support-email run."
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

    static func duration(_ s: Double) -> String {
        let total = Int(s.rounded())
        return "\(total / 60)m\(String(format: "%02d", total % 60))s"
    }
}
