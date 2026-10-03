import SwiftUI
import UIKit

// MARK: - Intake: the job, the chart, the target

struct IntakePanel: View {
    let step: IntakeStep
    @Environment(Session.self) private var session

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Step \(step.rawValue + 1) of \(IntakeStep.allCases.count)")
                        .font(.footnote.weight(.semibold))
                        .foregroundStyle(Brand.secondary)
                    Text(step.title)
                        .font(.title2.weight(.bold))
                        .foregroundStyle(Brand.ink)
                }
                Spacer()
                HStack(spacing: 4) {
                    ForEach(IntakeStep.allCases, id: \.self) { s in
                        Capsule()
                            .fill(s.rawValue <= step.rawValue ? Brand.accent : Brand.line)
                            .frame(width: s == step ? 18 : 8, height: 8)
                    }
                }
                .accessibilityHidden(true)
            }

            switch step {
            case .job: jobStep
            case .chart: chart
            case .target: target
            }

            PrimaryButton(title: step == .target ? "Start surgery" : "Continue",
                          symbol: step == .target ? "scissors" : "arrow.right") { session.next() }
        }
        .card()
    }

    @ViewBuilder private var jobStep: some View {
        @Bindable var session = session
        Field(label: "Describe one job") {
            TextField(QualityRun.demoJob, text: $session.job, axis: .vertical)
                .lineLimit(2...4)
                .font(.body)
                .foregroundStyle(Brand.ink)
                .padding(14)
                .background(Brand.tile, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Brand.line))
                .onChange(of: session.job) { session.touched() }
        }
        Button {
            session.job = QualityRun.demoJob
            session.touched()
        } label: {
            Label("Use the demo job", systemImage: "envelope.badge")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Brand.accent)
        }
        .buttonStyle(.plain)
        Text("A specialist, not a know-it-all: one model per job. In this demo Dr. Lobbot replays the real support-email run.")
            .font(.footnote)
            .foregroundStyle(Brand.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }

    /// The doctor's clipboard: the task spec Gemini drafts from the sentence (`lobbot new`).
    private var chart: some View {
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 0) {
                ChartRow(label: "Job", value: session.jobTitle)
                Divider()
                ChartRow(label: "In", value: QualityRun.input)
                Divider()
                ChartRow(label: "Out", value: QualityRun.output)
                Divider()
                ChartRow(label: "Graded on", value: QualityRun.grading)
                Divider()
                VStack(alignment: .leading, spacing: 6) {
                    Text("Example 1 of \(QualityRun.seeds)")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(Brand.secondary)
                    Text(QualityRun.exampleIn)
                        .font(.footnote)
                        .foregroundStyle(Brand.ink)
                    Text(QualityRun.exampleOut)
                        .font(.caption.monospaced())
                        .foregroundStyle(Brand.accent)
                }
                .padding(.vertical, 11)
            }
            .padding(.horizontal, 16)
            .padding(.top, 18)
            .padding(.bottom, 6)
            .background(Color.white, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Brand.line))
            .overlay(alignment: .top) {
                Capsule().fill(Brand.ink).frame(width: 76, height: 14).offset(y: -7)   // the clip
            }
            .padding(.top, 8)
            Text("Drafted by Gemini from your sentence. This demo shows the support-ticket chart.")
                .font(.caption)
                .foregroundStyle(Brand.secondary)
        }
    }

    private var target: some View {
        VStack(alignment: .leading, spacing: 14) {
            Field(label: "Runs on") {
                HStack(spacing: 12) {
                    Image(systemName: "laptopcomputer").font(.title2).foregroundStyle(Brand.accent)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("A 16 GB laptop").font(.headline).foregroundStyle(Brand.ink)
                        Text("Runs offline in Ollama on your Mac").font(.caption).foregroundStyle(Brand.secondary)
                    }
                    Spacer(minLength: 0)
                }
                .padding(14)
                .background(Brand.tile, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            }
            HStack(spacing: 10) {
                Vital(label: "File size", value: "≤ 7 GB", detail: "to fit the laptop")
                Vital(label: "Speed", value: "≥ 40", detail: "tokens per second")
            }
            Field(label: "Starting from") {
                Text("\(QualityRun.teacher), the big model: \(QualityRun.teacherGB) that knows everything.")
                    .font(.subheadline)
                    .foregroundStyle(Brand.ink)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Label("Surgery on \(QualityRun.gpu) GPU, about 25 minutes", systemImage: "cpu")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Brand.accent)
        }
    }
}

private struct ChartRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text(label)
                .font(.subheadline)
                .foregroundStyle(Brand.secondary)
                .frame(width: 80, alignment: .leading)
            Text(value)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Brand.ink)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
        .padding(.vertical, 11)
        .accessibilityElement(children: .combine)
    }
}

// MARK: - Surgery: the live board (same rows as `lobbot run`)

struct SurgeryPanel: View {
    @Environment(Session.self) private var session

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Label("In surgery", systemImage: "cross.case.fill")
                    .font(.headline)
                    .foregroundStyle(Brand.accent)
                Spacer()
                Button("Stop") { session.stop() }
                    .font(.subheadline.weight(.semibold))
                    .buttonStyle(.bordered)
                    .tint(Brand.restored)
            }
            Text(session.jobTitle)
                .font(.subheadline)
                .foregroundStyle(Brand.secondary)
                .lineLimit(2)

            ProgressView(value: session.progress)
                .tint(Brand.accent)
                .accessibilityLabel("Surgery progress")

            VStack(spacing: 0) {
                ForEach(session.rows) { row in
                    StageRowView(row: row)
                    if row.stage != Stage.allCases.last { Divider() }
                }
            }

            HStack {
                Text("GPU time \(Session.duration(session.gpuSeconds))")
                Spacer()
                Text(QualityRun.gpu)
            }
            .font(.caption.monospacedDigit())
            .foregroundStyle(Brand.secondary)

            Text("Replay of the real run of \(QualityRun.date): \(QualityRun.gpuTime) of GPU time, shown in about 20 seconds.")
                .font(.caption)
                .foregroundStyle(Brand.secondary)
        }
        .card()
    }
}

private struct StageRowView: View {
    let row: StageRow

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                icon.frame(width: 22)
                Text(row.stage.title)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(row.status == .pending ? Brand.secondary : Brand.ink)
                Text(row.stage.rawValue)
                    .font(.caption.monospaced())
                    .foregroundStyle(Brand.secondary)
                Spacer()
                if row.status != .pending {
                    Text(Session.duration(row.seconds))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(Brand.secondary)
                }
            }
            if row.status == .running || row.status == .done {
                ProgressView(value: row.pct, total: 100)
                    .tint(row.status == .done ? Brand.kept : Brand.accent)
            }
            if row.status != .pending, !row.msg.isEmpty {
                Text(row.msg)
                    .font(row.status == .done ? .caption.monospaced() : .caption)
                    .foregroundStyle(row.status == .error ? Brand.restored : Brand.secondary)
                    .lineLimit(2)
            }
        }
        .padding(.vertical, 10)
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder private var icon: some View {
        switch row.status {
        case .pending: Image(systemName: row.stage.symbol).foregroundStyle(Brand.line)
        case .running: ProgressView().tint(Brand.accent)
        case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(Brand.kept)
        case .error: Image(systemName: "xmark.circle.fill").foregroundStyle(Brand.restored)
        case .skipped: Image(systemName: "minus.circle").foregroundStyle(Brand.secondary)
        }
    }
}

private struct Vital: View {
    let label: String
    let value: String
    let detail: String

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label).font(.caption.weight(.semibold)).foregroundStyle(Brand.secondary).lineLimit(1)
            Text(value).font(.title3.weight(.bold)).monospacedDigit().foregroundStyle(Brand.ink)
            Text(detail).font(.caption2).foregroundStyle(Brand.secondary).lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Brand.tile, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .accessibilityElement(children: .combine)
    }
}

// MARK: - Result: the check-up and the discharge

struct ResultPanel: View {
    @Environment(Session.self) private var session

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Surgery complete").font(.footnote.weight(.semibold)).foregroundStyle(Brand.kept)
                Text("About 9× smaller")
                    .font(.system(size: 36, weight: .heavy))
                    .foregroundStyle(Brand.ink)
                Text("\(QualityRun.score) against \(QualityRun.teacherScore) for the big model: \(QualityRun.share) of its score on \(QualityRun.tests) new emails, graded by Gemini.")
                    .foregroundStyle(Brand.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack(spacing: 10) {
                SizeBox(title: "Big model", size: QualityRun.teacherGB, detail: QualityRun.teacher)
                Image(systemName: "arrow.right").foregroundStyle(Brand.secondary).accessibilityHidden(true)
                SizeBox(title: "LobBot", size: QualityRun.modelGB, detail: "one GGUF file", highlighted: true)
            }

            Field(label: "The check-up") {
                VStack(spacing: 2) {
                    ForEach(QualityRun.candidates) { CandidateRow(candidate: $0) }
                }
                Text("Gemma 4 scored 9.8 too, but at 26 tokens/s it misses the 40 tokens/s target. LobBot is the fastest one that's good enough.")
                    .font(.caption)
                    .foregroundStyle(Brand.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            VStack(spacing: 0) {
                ResultRow(symbol: "laptopcomputer", text: "About 50 tokens/s on a 16 GB MacBook")
                Divider()
                ResultRow(symbol: "wifi.slash", text: "Runs offline once it's on your laptop")
                Divider()
                ResultRow(symbol: "square.stack.3d.down.right", text: "Half the experts removed, 2 to 6 bits per layer")
            }

            Field(label: "A real email from the check-up") {
                VStack(alignment: .leading, spacing: 8) {
                    Text(QualityRun.heroEmail)
                        .font(.footnote)
                        .foregroundStyle(Brand.ink)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Brand.tile, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                    Image(systemName: "arrow.down")
                        .foregroundStyle(Brand.secondary)
                        .frame(maxWidth: .infinity)
                        .accessibilityHidden(true)
                    Text(QualityRun.heroTicket)
                        .font(.caption.monospaced())
                        .foregroundStyle(.white)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Brand.ink, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                }
            }

            Field(label: "Take it home, on your Mac") {
                CommandRow(command: QualityRun.saveCommand)
                Text("Downloads the 6.52 GB file, checks it and installs it in Ollama. Then chat with it offline:")
                    .font(.caption)
                    .foregroundStyle(Brand.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                CommandRow(command: QualityRun.chatCommand)
            }

            HStack(spacing: 10) {
                ShareLink(item: session.report) {
                    Label("Share report", systemImage: "square.and.arrow.up")
                        .font(.headline)
                        .frame(maxWidth: .infinity, minHeight: 50)
                }
                .buttonStyle(.bordered)
                .tint(Brand.accent)
                PrimaryButton(title: "New patient", symbol: "plus") { session.newPatient() }
            }

            Text("Replay of a real run: \(QualityRun.date), \(QualityRun.gpuTime) on \(QualityRun.gpu) GPU.")
                .font(.caption)
                .foregroundStyle(Brand.secondary)
        }
        .card()
    }
}

private struct CandidateRow: View {
    let candidate: QualityRun.Candidate

    var body: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 1) {
                Text(candidate.name)
                    .font(.footnote.weight(candidate.winner ? .bold : .medium))
                    .foregroundStyle(Brand.ink)
                Text(candidate.size).font(.caption2).foregroundStyle(Brand.secondary)
            }
            Spacer(minLength: 4)
            Text(candidate.laptop == "–" ? "–" : "\(candidate.laptop) tok/s")
                .font(.caption.monospacedDigit())
                .foregroundStyle(candidate.meetsTarget == false ? Brand.restored : Brand.secondary)
                .frame(width: 64, alignment: .trailing)
            Text(candidate.judge)
                .font(.subheadline.weight(.bold).monospacedDigit())
                .foregroundStyle(Brand.ink)
                .frame(width: 32, alignment: .trailing)
            Group {
                if let ok = candidate.meetsTarget {
                    Image(systemName: ok ? "checkmark.circle.fill" : "xmark.circle.fill")
                        .foregroundStyle(ok ? Brand.kept : Brand.restored)
                } else {
                    Color.clear
                }
            }
            .frame(width: 20)
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 8)
        .background(candidate.winner ? Brand.rose.opacity(0.35) : Color.clear, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .accessibilityElement(children: .combine)
    }
}

/// A terminal command with a copy button.
private struct CommandRow: View {
    let command: String
    @State private var copied = false

    var body: some View {
        HStack(spacing: 10) {
            Text("$ " + command)
                .font(.footnote.monospaced())
                .foregroundStyle(.white)
                .lineLimit(1)
                .minimumScaleFactor(0.75)
            Spacer(minLength: 0)
            Button {
                UIPasteboard.general.string = command
                copied = true
            } label: {
                Image(systemName: copied ? "checkmark" : "doc.on.doc").foregroundStyle(Brand.rose)
            }
            .accessibilityLabel(copied ? "Copied" : "Copy command")
        }
        .padding(12)
        .background(Brand.ink, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .task(id: copied) {
            guard copied else { return }
            try? await Task.sleep(for: .seconds(1.5))
            copied = false
        }
    }
}

private struct SizeBox: View {
    let title: String
    let size: String
    let detail: String
    var highlighted = false

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption.weight(.semibold)).opacity(0.85)
            Text(size).font(.title2.weight(.bold)).monospacedDigit()
            Text(detail).font(.caption).opacity(0.85).lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .foregroundStyle(highlighted ? Color.white : Brand.ink)
        .background(highlighted ? Brand.accent : Brand.tile, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .accessibilityElement(children: .combine)
    }
}

private struct ResultRow: View {
    let symbol: String
    let text: String

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: symbol).foregroundStyle(Brand.accent).frame(width: 24)
            Text(text).font(.subheadline.weight(.medium)).foregroundStyle(Brand.ink)
            Spacer(minLength: 0)
        }
        .padding(.vertical, 10)
        .accessibilityElement(children: .combine)
    }
}
