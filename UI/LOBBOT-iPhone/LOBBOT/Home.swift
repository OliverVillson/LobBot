import SwiftUI

/// Launch animation: the Sillon mark draws itself on the bordeaux stage, then the wordmark lands.
/// It stays up while the WebGL mascot loads behind it.
struct SplashView: View {
    @State private var drawn: CGFloat = 0
    @State private var landed = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        ZStack {
            RadialGradient(colors: [Brand.stageLight, Brand.stage, Brand.stageDeep],
                           center: .center, startRadius: 0, endRadius: 520)
                .ignoresSafeArea()
            VStack(spacing: 22) {
                SillonLine()
                    .trim(from: 0, to: drawn)
                    .stroke(Color.white, style: StrokeStyle(lineWidth: 11, lineCap: .round, lineJoin: .round))
                    .frame(width: 120, height: 120)
                    .scaleEffect(landed ? 1 : 0.9)
                VStack(spacing: 6) {
                    Text("lobbot")
                        .font(.system(size: 46, weight: .heavy))
                        .foregroundStyle(.white)
                    Text("Cut the model. Keep the capability.")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(.white.opacity(0.88))
                }
                .opacity(landed ? 1 : 0)
                .offset(y: landed ? 0 : 14)
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("lobbot. Cut the model. Keep the capability.")
        .task {
            if reduceMotion {
                drawn = 1
                landed = true
                return
            }
            withAnimation(.easeInOut(duration: 1.1)) { drawn = 1 }
            try? await Task.sleep(for: .milliseconds(800))
            withAnimation(.spring(duration: 0.6, bounce: 0.35)) { landed = true }
        }
    }
}

/// The ward board: Dr. Lobbot's patients. Bed 1 is the real run, this session's replays follow, and the
/// empty bed admits a new patient. The legend underneath explains how a patient moves through the ward.
struct HomePanel: View {
    @Environment(Session.self) private var session
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    /// The bed Lobbot is looking at on his rounds; -1 when he's done.
    @State private var round = -1

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            board
            legend
            Text("Bed 1 is the real run of \(QualityRun.date) on \(QualityRun.gpu) GPU, graded by Gemini on \(QualityRun.tests) new emails. Surgeries in this app replay it.")
                .font(.caption)
                .foregroundStyle(Brand.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 4)
        }
        .task {
            // His rounds: each bed plate lights up in turn, ending on the empty bed.
            guard !reduceMotion else { return }
            try? await Task.sleep(for: .milliseconds(500))
            for i in 0...beds.count {
                withAnimation(.easeOut(duration: 0.2)) { round = i }
                try? await Task.sleep(for: .milliseconds(380))
                if Task.isCancelled { return }
            }
            withAnimation(.easeOut(duration: 0.3)) { round = -1 }
        }
    }

    private var beds: [Bed] {
        var list = [Bed(id: "real", number: 1, job: "Support emails → JSON tickets",
                        detail: "\(QualityRun.teacherGB) → \(QualityRun.modelGB) · \(QualityRun.score) vs \(QualityRun.teacherScore)",
                        note: "About 50 tokens/s on a 16 GB MacBook, offline", real: true)]
        for (i, record) in session.history.enumerated() {
            list.append(Bed(id: record.id.uuidString, number: i + 2, job: record.job,
                            detail: "Replayed at \(record.date.formatted(date: .omitted, time: .shortened))",
                            note: nil, real: false))
        }
        return list
    }

    private var board: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .firstTextBaseline) {
                Text("Ward 64")
                    .font(.title2.weight(.heavy))
                    .foregroundStyle(.white)
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                Text("Dr. Lobbot's patients")
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.88))
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(Brand.stage)

            Text("One job in, a small model out. It runs on your laptop, offline.")
                .font(.headline)
                .foregroundStyle(Brand.ink)
                .fixedSize(horizontal: false, vertical: true)
                .padding(16)

            ForEach(Array(beds.enumerated()), id: \.element.id) { index, bed in
                Divider()
                BedRow(bed: bed, onRound: round == index) { session.openBed(real: bed.real) }
            }
            Divider()
            EmptyBedRow(number: beds.count + 1, onRound: round == beds.count) { session.startIntake() }
        }
        .background(Color.white)
        .clipShape(RoundedRectangle(cornerRadius: 22, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 22, style: .continuous).stroke(Brand.line))
    }

    private var legend: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("How a patient moves through the ward")
                .font(.headline)
                .foregroundStyle(Brand.ink)
                .accessibilityAddTraits(.isHeader)
            LegendRow(symbol: "list.clipboard", title: "Admission", code: nil, text: "You describe one job; Gemini drafts the chart.")
            ForEach(Stage.allCases) { stage in
                LegendRow(symbol: stage.symbol, title: stage.title, code: stage.rawValue, text: stage.fact)
            }
            LegendRow(symbol: "laptopcomputer", title: "Home", code: nil, text: "It runs offline on a 16 GB laptop.")
        }
        .card()
    }
}

private struct Bed: Identifiable {
    let id: String
    let number: Int
    let job: String
    let detail: String
    let note: String?
    let real: Bool
}

/// A bed plate: white digits on ink, lit in accent while Lobbot looks at it.
private struct BedPlate: View {
    let number: Int
    let lit: Bool

    var body: some View {
        Text("\(number)")
            .font(.headline.monospacedDigit())
            .foregroundStyle(.white)
            .frame(width: 36, height: 36)
            .background(lit ? Brand.accent : Brand.ink, in: RoundedRectangle(cornerRadius: 9, style: .continuous))
            .scaleEffect(lit ? 1.08 : 1)
    }
}

private struct BedRow: View {
    let bed: Bed
    let onRound: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(alignment: .top, spacing: 12) {
                BedPlate(number: bed.number, lit: onRound)
                VStack(alignment: .leading, spacing: 3) {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(bed.job)
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(Brand.ink)
                            .lineLimit(2)
                        Spacer(minLength: 4)
                        Text(bed.real ? "Discharged" : "Replay")
                            .font(.caption2.weight(.bold))
                            .foregroundStyle(bed.real ? Brand.kept : Brand.accent)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background((bed.real ? Brand.kept : Brand.accent).opacity(0.12), in: Capsule())
                    }
                    Text(bed.detail)
                        .font(.footnote.monospacedDigit())
                        .foregroundStyle(Brand.ink)
                    if let note = bed.note {
                        Text(note)
                            .font(.caption)
                            .foregroundStyle(Brand.secondary)
                    }
                }
                Image(systemName: "chevron.right")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(Brand.secondary)
                    .padding(.top, 10)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 14)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Bed \(bed.number), \(bed.job), \(bed.real ? "discharged" : "replay"). \(bed.detail)")
        .accessibilityHint("Opens the discharge report")
        .accessibilityAddTraits(.isButton)
    }
}

/// The free bed: the home page's call to action. It breathes softly until you admit someone.
private struct EmptyBedRow: View {
    let number: Int
    let onRound: Bool
    let action: () -> Void
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var breathe = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Text("\(number)")
                    .font(.headline.monospacedDigit())
                    .foregroundStyle(Brand.accent)
                    .frame(width: 36, height: 36)
                    .overlay(
                        RoundedRectangle(cornerRadius: 9, style: .continuous)
                            .strokeBorder(Brand.accent, style: StrokeStyle(lineWidth: 1.5, dash: [4, 3]))
                    )
                    .scaleEffect(onRound ? 1.08 : 1)
                VStack(alignment: .leading, spacing: 3) {
                    Text("Admit a patient")
                        .font(.headline)
                        .foregroundStyle(Brand.accent)
                    Text("Describe one job. Dr. Lobbot does the rest.")
                        .font(.caption)
                        .foregroundStyle(Brand.secondary)
                }
                Spacer(minLength: 8)
                Image(systemName: "plus.circle.fill")
                    .font(.title2)
                    .foregroundStyle(Brand.accent)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 16)
            .background(Brand.rose.opacity(breathe ? 0.32 : 0.16))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Bed \(number), empty. Admit a patient")
        .onAppear {
            guard !reduceMotion else { return }
            withAnimation(.easeInOut(duration: 1.8).repeatForever(autoreverses: true)) { breathe = true }
        }
    }
}

private struct LegendRow: View {
    let symbol: String
    let title: String
    let code: String?
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: symbol)
                .font(.body)
                .foregroundStyle(Brand.accent)
                .frame(width: 26)
            VStack(alignment: .leading, spacing: 2) {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(title).font(.subheadline.weight(.semibold)).foregroundStyle(Brand.ink)
                    if let code {
                        Text(code).font(.caption.monospaced()).foregroundStyle(Brand.secondary)
                    }
                }
                Text(text)
                    .font(.footnote)
                    .foregroundStyle(Brand.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .combine)
    }
}
