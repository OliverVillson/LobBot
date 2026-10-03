import SwiftUI

@main
struct LOBBOTApp: App {
    @State private var session = Session()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(session)
                .environmentObject(session.lobbot)
                .preferredColorScheme(.light)
                .tint(Brand.accent)
        }
    }
}
