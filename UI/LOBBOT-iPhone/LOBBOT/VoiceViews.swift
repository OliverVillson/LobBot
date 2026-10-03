import SwiftUI

extension String {
    var isBlank: Bool { trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
}

/// Bottom bar of the main screen: type to Dr. Lobbot, or tap the mic to call him full screen.
struct Composer: View {
    @Environment(Session.self) private var session
    @State private var draft = ""
    @FocusState private var focused: Bool

    var body: some View {
        let voice = session.voice
        HStack(spacing: 10) {
            if case .intake(let step) = session.phase {   // back is always in reach, no scrolling
                Button { session.back() } label: {
                    Image(systemName: "chevron.left")
                        .font(.headline)
                        .frame(width: 48, height: 48)
                        .foregroundStyle(Brand.ink)
                        .background(Brand.surface, in: Circle())
                        .overlay(Circle().stroke(Brand.line))
                }
                .buttonStyle(.plain)
                .accessibilityLabel(step == .job ? "Home" : "Back")
            }
            HStack(spacing: 6) {
                TextField("Message Dr. Lobbot…", text: $draft)
                    .focused($focused)
                    .submitLabel(.send)
                    .onSubmit(send)
                    .foregroundStyle(Brand.ink)
                    .padding(.vertical, 13)
                    .padding(.leading, 16)
                if !draft.isBlank {
                    Button(action: send) {
                        Image(systemName: "arrow.up.circle.fill").font(.title2)
                    }
                    .foregroundStyle(Brand.accent)
                    .padding(.trailing, 8)
                    .accessibilityLabel("Send")
                }
            }
            .background(Brand.surface, in: Capsule())
            .overlay(Capsule().stroke(Brand.line))

            if voice.isOn {   // a typed chat is running: let him hang up
                Button { voice.stop() } label: {
                    Image(systemName: "xmark")
                        .font(.headline)
                        .frame(width: 48, height: 48)
                        .foregroundStyle(Brand.ink)
                        .background(Brand.tile, in: Circle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("End conversation")
            }
            Button {
                focused = false
                session.startCall()
            } label: {
                Image(systemName: "mic.fill")
                    .font(.title3.weight(.semibold))
                    .frame(width: 52, height: 52)
                    .foregroundStyle(.white)
                    .background(Brand.accent, in: Circle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Call Dr. Lobbot")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Brand.canvas.opacity(0.96).ignoresSafeArea(edges: .bottom))
    }

    private func send() {
        session.send(text: draft)
        draft = ""
    }
}

/// Full-screen call: Lobbot fills the page, captions and controls float over him.
struct CallOverlay: View {
    @Environment(Session.self) private var session
    @State private var typing = false
    @State private var draft = ""
    @FocusState private var focused: Bool

    var body: some View {
        let voice = session.voice
        VStack(spacing: 14) {
            HStack(spacing: 8) {
                Circle().fill(voice.status == .live ? Color.green : Brand.rose).frame(width: 8, height: 8)
                Text(status).font(.subheadline.weight(.semibold))
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(Color.black.opacity(0.18), in: Capsule())
            .padding(.top, 8)
            .accessibilityElement(children: .combine)

            if case .failed(let message) = voice.status {
                VStack(spacing: 8) {
                    Text(message).font(.footnote).multilineTextAlignment(.center)
                    Button("Try again") { voice.start(listening: true) }
                        .font(.footnote.weight(.semibold))
                        .buttonStyle(.bordered)
                        .tint(.white)
                }
                .foregroundStyle(.white)
                .padding(.horizontal, 12)
            }

            // His words float above his head so they never cover his face.
            if let line = session.line {
                SpeechBubble(text: line, tail: .bottom)
                    .lineLimit(6)
                    .truncationMode(.head)   // a long live transcript keeps its latest words visible
            }

            Spacer()

            if !voice.heard.isEmpty {
                Text(voice.heard)
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(.white)
                    .lineLimit(2)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 7)
                    .background(Color.black.opacity(0.22), in: Capsule())
                    .frame(maxWidth: .infinity, alignment: .trailing)
                    .accessibilityLabel("You: \(voice.heard)")
            }

            if typing {
                HStack(spacing: 6) {
                    TextField("Type to Dr. Lobbot…", text: $draft)
                        .focused($focused)
                        .submitLabel(.send)
                        .onSubmit(send)
                        .foregroundStyle(Brand.ink)
                        .padding(.vertical, 13)
                        .padding(.leading, 16)
                    Button(action: send) {
                        Image(systemName: "arrow.up.circle.fill").font(.title2)
                    }
                    .foregroundStyle(Brand.accent)
                    .padding(.trailing, 8)
                    .disabled(draft.isBlank)
                    .accessibilityLabel("Send")
                }
                .background(Brand.surface, in: Capsule())
            }

            HStack(spacing: 28) {
                CallButton(symbol: voice.micMuted ? "mic.slash.fill" : "mic.fill",
                           label: voice.micMuted ? "Unmute" : "Mute", active: voice.micMuted) {
                    voice.micMuted.toggle()
                    session.voiceChanged()
                }
                CallButton(symbol: "keyboard", label: "Type", active: typing) {
                    typing.toggle()
                    focused = typing
                }
                CallButton(symbol: "phone.down.fill", label: "End", tint: Brand.restored) {
                    session.endCall()
                }
            }
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 12)
    }

    private var status: String {
        let voice = session.voice
        switch voice.status {
        case .connecting: return "Calling Dr. Lobbot…"
        case .failed: return "Call failed"
        case .off: return "Call ended"
        case .live: return voice.isSpeaking ? "Dr. Lobbot is talking" : voice.micMuted ? "Mic muted" : "Listening"
        }
    }

    private func send() {
        session.send(text: draft)
        draft = ""
    }
}

private struct CallButton: View {
    let symbol: String
    let label: String
    var active = false
    var tint: Color? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 6) {
                Image(systemName: symbol)
                    .font(.title2)
                    .frame(width: 64, height: 64)
                    .foregroundStyle(active && tint == nil ? Brand.accent : Color.white)
                    .background(tint ?? (active ? Color.white : Color.white.opacity(0.2)), in: Circle())
                Text(label).font(.caption.weight(.semibold)).foregroundStyle(.white)
            }
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }
}

/// Shown under the header when the conversation fails, with the reason Gemini gave.
struct VoiceErrorBanner: View {
    let message: String
    @Environment(Session.self) private var session

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.bubble.fill").foregroundStyle(Brand.restored)
            Text(message)
                .font(.footnote)
                .foregroundStyle(Brand.ink)
                .frame(maxWidth: .infinity, alignment: .leading)
            Button("Dismiss") { session.voice.stop() }
                .font(.footnote.weight(.semibold))
                .foregroundStyle(Brand.accent)
        }
        .padding(12)
        .background(Brand.surface, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Brand.line))
        .padding(.horizontal, 16)
    }
}
