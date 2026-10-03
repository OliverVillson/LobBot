import SwiftUI

@main
struct LOBBOTApp: App {
    @AppStorage("lobbot.brandMode") private var mode: BrandMode = .anatomie
    @StateObject private var lobbot = LobbotController()

    var body: some Scene {
        WindowGroup {
            ExperimentSetupView(mode: $mode)
                .environmentObject(lobbot)
                .preferredColorScheme(mode.colorScheme)
                .tint(BrandColor.accent)
        }
    }
}
