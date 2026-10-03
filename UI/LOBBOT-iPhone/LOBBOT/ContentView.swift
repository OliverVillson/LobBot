import SwiftUI

/// One screen, built around Dr. Lobbot. Two layouts of the same stage:
/// - his tile at the top left, his bubble beside him, the panel under them (the home ward board and every
///   consultation step);
/// - a call started from the home page: he fills the screen.
/// The stage is always the same view (AnyLayout keeps its identity), so the WebGL mascot never reloads.
struct ContentView: View {
    @Environment(Session.self) private var session
    @State private var splash = true

    var body: some View {
        GeometryReader { geo in
            let full = session.fullScreenCall
            let pip = !full
            let tile = min(150, geo.size.width * 0.38)
            let layout = pip ? AnyLayout(HStackLayout(alignment: .top, spacing: 14)) : AnyLayout(VStackLayout(spacing: -14))
            ZStack {
                VStack(spacing: 12) {
                    if !full {
                        header
                        if case .failed(let message) = session.voice.status {
                            VoiceErrorBanner(message: message)
                        }
                    }

                    layout {
                        MascotStage(cornerRadius: full ? 0 : 22, aurora: full ? session.voice : nil)
                            .frame(width: pip ? tile : nil,
                                   height: full ? nil : tile * 1.15)
                            .frame(maxHeight: full ? .infinity : nil)
                            .ignoresSafeArea(edges: full ? .all : [])
                        if !full, let line = session.line {
                            SpeechBubble(text: line, tail: pip ? .leading : .top)
                                .lineLimit(pip ? 7 : nil)
                                .truncationMode(.head)   // a live transcript keeps its latest words
                                .padding(.horizontal, pip ? 0 : 24)
                                .padding(.top, pip ? 6 : 0)
                                .transition(.opacity)
                        }
                    }
                    .padding(.horizontal, full ? 0 : 16)
                    .animation(.easeOut(duration: 0.25), value: session.line)

                    if !full {
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

                if full {
                    CallOverlay().transition(.opacity)
                }
            }
        }
        .background(Brand.canvas.ignoresSafeArea())
        .safeAreaInset(edge: .bottom) {
            if !session.fullScreenCall { Composer() }
        }
        .animation(.snappy(duration: 0.45), value: session.fullScreenCall)
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
