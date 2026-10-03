import SwiftUI

/// One screen, built around Dr. Lobbot: he sits on top and talks, the panel under him follows
/// the consultation, and the bar at the bottom lets you type to him or call him full screen.
/// The stage stays the same view in both layouts so the WebGL mascot is never reloaded.
struct ContentView: View {
    @Environment(Session.self) private var session
    @State private var splash = true

    var body: some View {
        GeometryReader { geo in
            ZStack {
                VStack(spacing: 12) {
                    if !session.inCall {
                        header
                        if case .failed(let message) = session.voice.status {
                            VoiceErrorBanner(message: message)
                        }
                    }

                    VStack(spacing: -14) {
                        MascotStage(cornerRadius: session.inCall ? 0 : 28)
                            .frame(height: session.inCall ? nil : min(geo.size.width - 32, geo.size.height * 0.42))
                            .frame(maxHeight: session.inCall ? .infinity : nil)
                            .ignoresSafeArea(edges: session.inCall ? .all : [])
                        if !session.inCall, let line = session.line {
                            SpeechBubble(text: line)
                                .padding(.horizontal, 24)
                                .transition(.opacity.combined(with: .offset(y: -8)))
                        }
                    }
                    .padding(.horizontal, session.inCall ? 0 : 16)
                    .animation(.easeOut(duration: 0.25), value: session.line)

                    if !session.inCall {
                        if session.voice.isOn, !session.voice.heard.isEmpty {
                            Text(session.voice.heard)
                                .font(.footnote)
                                .foregroundStyle(Brand.ink)
                                .lineLimit(2)
                                .padding(.horizontal, 12)
                                .padding(.vertical, 7)
                                .background(Brand.rose.opacity(0.45), in: Capsule())
                                .frame(maxWidth: .infinity, alignment: .trailing)
                                .padding(.horizontal, 20)
                                .accessibilityLabel("You: \(session.voice.heard)")
                        }
                        ScrollView {
                            panel
                                .padding(.horizontal, 16)
                                .padding(.top, 4)
                                .padding(.bottom, 24)
                                .animation(.snappy(duration: 0.3), value: session.phase)
                        }
                        .scrollIndicators(.hidden)
                    }
                }

                if session.inCall {
                    CallOverlay().transition(.opacity)
                }
            }
        }
        .background(Brand.canvas.ignoresSafeArea())
        .safeAreaInset(edge: .bottom) {
            if !session.inCall { Composer() }
        }
        .animation(.snappy(duration: 0.4), value: session.inCall)
        .overlay {
            if splash {
                SplashView().transition(.opacity.combined(with: .scale(scale: 1.08)))
            }
        }
        .task {
            // Keep the launch animation up until the mascot is ready (at least 1.9 s, at most 4 s).
            let start = Date.now
            while Date.now.timeIntervalSince(start) < 4 {
                if Date.now.timeIntervalSince(start) >= 1.9, session.lobbot.loadState == .ready { break }
                try? await Task.sleep(for: .milliseconds(100))
            }
            withAnimation(.easeInOut(duration: 0.55)) { splash = false }
            session.greet()
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
            Button { session.goHome() } label: {
                HStack(spacing: 10) {
                    SillonMark().frame(width: 28, height: 28)
                    Text("lobbot")
                        .font(.system(size: 26, weight: .heavy))
                        .foregroundStyle(Brand.ink)
                }
            }
            .buttonStyle(.plain)
            .accessibilityLabel("lobbot, home")
            .accessibilityAddTraits(.isHeader)
            Spacer()
            Text("Replay")
                .font(.caption.weight(.semibold))
                .foregroundStyle(Brand.accent)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(Brand.rose.opacity(0.45), in: Capsule())
                .accessibilityLabel("Surgeries replay a real run from October 3, 2026")
        }
        .padding(.horizontal, 20)
        .padding(.top, 4)
    }

    @ViewBuilder private var panel: some View {
        switch session.phase {
        case .home: HomePanel()
        case .intake(let step): IntakePanel(step: step)
        case .surgery: SurgeryPanel()
        case .done: ResultPanel()
        }
    }
}

#Preview {
    let session = Session()
    return ContentView()
        .environment(session)
        .environmentObject(session.lobbot)
}
