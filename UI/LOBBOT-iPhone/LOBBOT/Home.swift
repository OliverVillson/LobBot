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

/// The lobby: what lobbot does, the two ways in, and the surgeries done so far.
struct HomePanel: View {
    @Environment(Session.self) private var session

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Cut the model.\nKeep the capability.")
                    .font(.system(size: 30, weight: .heavy))
                    .foregroundStyle(Brand.ink)
                    .fixedSize(horizontal: false, vertical: true)
                Text("Dr. Lobbot shrinks a large model and tests the skill you care about after every cut. Cuts that hurt it are undone.")
                    .font(.subheadline)
                    .foregroundStyle(Brand.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.horizontal, 4)

            PrimaryButton(title: "New surgery", symbol: "scissors") { session.startIntake() }
            Button { session.startCall() } label: {
                Label("Talk to Dr. Lobbot", systemImage: "mic.fill")
                    .font(.headline)
                    .frame(maxWidth: .infinity, minHeight: 54)
                    .foregroundStyle(Brand.accent)
                    .background(Brand.surface, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Brand.line))
            }
            .buttonStyle(.plain)

            VStack(alignment: .leading, spacing: 14) {
                Field(label: "How it works") {
                    VStack(alignment: .leading, spacing: 12) {
                        HowStep(number: 1, title: "The chart", text: "Pick the model, the skill to keep and the device it must fit.")
                        HowStep(number: 2, title: "The surgery", text: "Cut a block, test the skill, keep the cut or undo it. Repeat.")
                        HowStep(number: 3, title: "The patient goes home", text: "A smaller model that still does the job.")
                    }
                }
            }
            .card()

            Field(label: "Recent surgeries") {
                if session.history.isEmpty {
                    Text("No patients yet. Your first surgery will show up here.")
                        .font(.subheadline)
                        .foregroundStyle(Brand.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else {
                    VStack(spacing: 0) {
                        ForEach(session.history) { record in
                            RecordRow(record: record)
                            if record.id != session.history.last?.id { Divider() }
                        }
                    }
                }
            }
            .card()

            Text("Simulated runs: no model is modified.")
                .font(.caption)
                .foregroundStyle(Brand.secondary)
                .padding(.horizontal, 4)
        }
    }
}

private struct HowStep: View {
    let number: Int
    let title: String
    let text: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Text("\(number)")
                .font(.subheadline.weight(.bold))
                .foregroundStyle(.white)
                .frame(width: 26, height: 26)
                .background(Brand.accent, in: Circle())
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.subheadline.weight(.semibold)).foregroundStyle(Brand.ink)
                Text(text).font(.footnote).foregroundStyle(Brand.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .combine)
    }
}

private struct RecordRow: View {
    let record: SurgeryRecord

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: record.capability.symbol)
                .foregroundStyle(Brand.accent)
                .frame(width: 24)
            VStack(alignment: .leading, spacing: 2) {
                Text(record.patient)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(Brand.ink)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text("\(Session.format(record.fromB))B → \(Session.format(record.toB))B · \(record.capability.title) \(Session.format(record.retained))%")
                    .font(.caption)
                    .foregroundStyle(Brand.secondary)
            }
            Spacer(minLength: 0)
            Text(record.date, style: .time)
                .font(.caption)
                .foregroundStyle(Brand.secondary)
        }
        .padding(.vertical, 10)
        .accessibilityElement(children: .combine)
    }
}
