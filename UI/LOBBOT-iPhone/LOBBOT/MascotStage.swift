import SwiftUI

/// Dr. Lobbot's bordeaux stage: the persistent WebGL mascot plus load and failure states.
struct MascotStage: View {
    var cornerRadius: CGFloat = 28
    /// During a call: northern lights behind him that follow your voice and his.
    var aurora: VoiceAgent? = nil
    @EnvironmentObject private var lobbot: LobbotController

    var body: some View {
        ZStack {
            RadialGradient(colors: [Brand.stageLight, Brand.stage, Brand.stageDeep],
                           center: UnitPoint(x: 0.5, y: 0.45), startRadius: 0, endRadius: 360)
            if let aurora { Aurora(voice: aurora) }
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

/// Northern lights behind Dr. Lobbot during a call. Three soft bands drift all the time; the rose ones swell
/// with your voice and the white one with his, so you can see at a glance that the mic hears you.
/// Gradients only, no blur, at 30 fps: the WebGL mascot keeps the GPU.
struct Aurora: View {
    let voice: VoiceAgent
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        let mic = CGFloat(voice.micLevel)
        let him = CGFloat(voice.voiceLevel)
        GeometryReader { geo in
            TimelineView(.animation(minimumInterval: 1.0 / 30, paused: reduceMotion)) { timeline in
                let t = timeline.date.timeIntervalSinceReferenceDate
                ZStack {
                    band(Brand.rose, in: geo.size, t: t, phase: 0, y: 0.30, lift: mic)
                    band(Color(hex: 0xFF8FA6), in: geo.size, t: t, phase: 2.1, y: 0.42, lift: mic * 0.8)
                    band(.white, in: geo.size, t: t, phase: 4.2, y: 0.24, lift: him * 0.9, strength: 0.32)
                }
                .animation(.easeOut(duration: 0.18), value: mic)
                .animation(.easeOut(duration: 0.18), value: him)
            }
        }
        .blendMode(.screen)
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }

    private func band(_ color: Color, in size: CGSize, t: Double, phase: Double, y: CGFloat,
                      lift: CGFloat, strength: Double = 0.5) -> some View {
        Ellipse()
            .fill(EllipticalGradient(colors: [color.opacity(strength + Double(lift) * 0.4), color.opacity(0)],
                                     center: .center, startRadiusFraction: 0, endRadiusFraction: 0.5))
            .frame(width: size.width * 1.5, height: size.height * (0.22 + lift * 0.28))
            .rotationEffect(.degrees(sin(t * 0.11 + phase) * 9 - 6))
            .offset(x: CGFloat(sin(t * 0.17 + phase)) * size.width * 0.12,
                    y: size.height * (y - 0.5) - lift * size.height * 0.08 + CGFloat(sin(t * 0.23 + phase)) * 14)
            .scaleEffect(x: 1, y: 1 + lift * 0.5)
    }
}

/// What Lobbot says, typed out letter by letter. `tail` is the side facing him: top (bubble under
/// the stage), bottom (above his head in a full-screen call) or leading (beside his picture-in-picture tile).
struct SpeechBubble: View {
    let text: String
    var tail: Edge = .top
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
        .overlay(alignment: tail == .bottom ? .bottom : .topLeading) {
            BubbleTail()
                .fill(Brand.surface)
                .frame(width: 20, height: 11)
                .rotationEffect(.degrees(tail == .bottom ? 180 : tail == .leading ? -90 : 0))
                .offset(tailOffset)
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

private extension SpeechBubble {
    var tailOffset: CGSize {
        switch tail {
        case .bottom: CGSize(width: 0, height: 10)
        case .leading: CGSize(width: -15, height: 24)
        default: CGSize(width: 36, height: -10)
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
