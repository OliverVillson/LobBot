import SwiftUI

/// Dr. Lobbot's bordeaux stage: the persistent WebGL mascot plus load and failure states.
struct MascotStage: View {
    var cornerRadius: CGFloat = 28
    @EnvironmentObject private var lobbot: LobbotController

    var body: some View {
        ZStack {
            RadialGradient(colors: [Brand.stageLight, Brand.stage, Brand.stageDeep],
                           center: UnitPoint(x: 0.5, y: 0.45), startRadius: 0, endRadius: 360)
            LobbotView(controller: lobbot)
                .accessibilityElement(children: .ignore)
                .accessibilityLabel("Dr. Lobbot, \(lobbot.scene.title.lowercased())")
                .accessibilityHint("Tap to wake him up.")
                .accessibilityAddTraits(.isButton)
                .accessibilityAction { lobbot.activity() }
            switch lobbot.loadState {
            case .loading:
                ProgressView("Dr. Lobbot is scrubbing in…")
                    .tint(.white)
                    .foregroundStyle(.white)
            case .failed(let message):
                VStack(spacing: 12) {
                    Text(message).font(.subheadline).multilineTextAlignment(.center)
                    Button("Try again") { lobbot.retry() }
                        .buttonStyle(.bordered)
                        .tint(.white)
                        .frame(minHeight: 44)
                }
                .foregroundStyle(.white)
                .padding(20)
            case .ready:
                EmptyView()
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
    }
}

/// What Lobbot says, typed out letter by letter. The tail points up (bubble under him)
/// or down (bubble above his head, in a call).
struct SpeechBubble: View {
    let text: String
    var tailOnTop = true
    @State private var shown = 0
    @State private var typed = ""
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        ZStack(alignment: .topLeading) {
            Text(text).hidden()                 // reserves the final size so the layout doesn't jump
            Text(text.prefix(shown))
        }
        .font(.callout.weight(.medium))
        .foregroundStyle(Brand.ink)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(Brand.surface, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(alignment: tailOnTop ? .topLeading : .bottom) {
            BubbleTail()
                .fill(Brand.surface)
                .frame(width: 20, height: 11)
                .rotationEffect(.degrees(tailOnTop ? 0 : 180))
                .offset(x: tailOnTop ? 36 : 0, y: tailOnTop ? -10 : 10)
        }
        .shadow(color: Brand.ink.opacity(0.14), radius: 12, y: 4)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Dr. Lobbot says: \(text)")
        .task(id: text) {
            guard !reduceMotion else { shown = text.count; return }
            // A live transcript only grows: keep typing from where we were instead of restarting.
            let start = text.hasPrefix(typed) ? min(shown, text.count) : 0
            typed = text
            for i in start...text.count {
                shown = i
                try? await Task.sleep(for: .milliseconds(28))
                if Task.isCancelled { return }
            }
        }
    }
}

nonisolated struct BubbleTail: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.minX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.midX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
        path.closeSubpath()
        return path
    }
}
