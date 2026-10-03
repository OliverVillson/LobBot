import SwiftUI

struct MascotCard: View {
    @EnvironmentObject private var lobbot: LobbotController

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("L’agent").font(.headline)
                    Text(lobbot.scene.title).font(.caption).foregroundStyle(BrandColor.secondary)
                        .accessibilityIdentifier("mascot-scene")
                }
                Spacer()
                Menu {
                    ForEach(LobbotScene.allCases) { scene in
                        Button(scene.title) { lobbot.preview(scene) }
                    }
                    Divider()
                    Button("Terminer la chirurgie") { lobbot.finish() }
                        .disabled(!lobbot.isHoldingTask)
                    Button("Revenir au repos") { lobbot.show(.idle); lobbot.stop() }
                } label: {
                    Label("Tester les scènes", systemImage: "ellipsis.circle")
                        .labelStyle(.iconOnly)
                        .frame(minWidth: 44, minHeight: 44)
                }
                .accessibilityIdentifier("mascot-scenes-menu")
                .foregroundStyle(BrandColor.accent)
            }

            ZStack {
                Color(red: 150 / 255, green: 68 / 255, blue: 90 / 255)
                LobbotView(controller: lobbot)
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel("Mascotte Lobbot, \(lobbot.scene.title)")
                    .accessibilityHint("Touchez pour réveiller l’agent.")
                    .accessibilityAddTraits(.isButton)
                    .accessibilityAction { lobbot.activity() }
                switch lobbot.loadState {
                case .loading:
                    ProgressView("Préparation de l’agent…")
                        .tint(.white).foregroundStyle(.white)
                case .failed(let message):
                    VStack(spacing: 12) {
                        Text(message).font(.subheadline).multilineTextAlignment(.center)
                        Button("Réessayer") { lobbot.retry() }
                            .buttonStyle(.bordered).frame(minHeight: 44)
                    }
                    .foregroundStyle(.white).padding(20)
                case .ready: EmptyView()
                }
            }
            .aspectRatio(4 / 3, contentMode: .fit)
            .clipShape(RoundedRectangle(cornerRadius: 22))
        }
    }
}
