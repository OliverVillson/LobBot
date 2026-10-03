import SwiftUI
import WebKit
import UIKit
import OSLog

enum LobbotScene: String, CaseIterable, Identifiable {
    case idle, listen, think, eureka, talk, task, bug, error, sleep
    var id: String { rawValue }
    var title: String {
        switch self {
        case .idle: "Idle"
        case .listen: "Listening"
        case .think: "Thinking"
        case .eureka: "Eureka"
        case .talk: "Talking"
        case .task: "In surgery"
        case .bug: "Chasing a bug"
        case .error: "Dizzy"
        case .sleep: "Napping"
        }
    }
}

enum LobbotAppState: Equatable {
    case idle, typing, recording, planning, requesting, streaming
    case compressing, succeeded, failed, regression, discovery, inactive
}

/// One controller owns one WebGL context for the lifetime of the app.
@MainActor
final class LobbotController: ObservableObject {
    enum LoadState: Equatable { case loading, ready, failed(String) }
    @Published private(set) var loadState: LoadState = .loading
    @Published private(set) var scene: LobbotScene = .idle
    @Published private(set) var isHoldingTask = false

    private enum Command {
        case play(LobbotScene, loop: Bool, hold: Bool)
        case finish, stop
        case framing(close: Bool)
        var javaScript: String {
            switch self {
            case let .play(scene, loop, hold):
                // Scene is a closed enum; no user-supplied JavaScript is interpolated.
                "window.lobbot.play('\(scene.rawValue)', {loop: \(loop), hold: \(hold)});"
            case .finish: "window.lobbot.finish();"
            case .stop: "window.lobbot.stop();"
            case let .framing(close): "window.lobbot.setFraming('\(close ? "close" : "wide")');"
            }
        }
    }

    private let logger = Logger(subsystem: "dev.lobbot.prototype", category: "Mascot")
    private var commands: [Command] = []
    private var isDraining = false
    private var hasLoaded = false
    private var currentState: LobbotAppState = .idle
    private var inactivityTask: Task<Void, Never>?
    private var readinessTask: Task<Void, Never>?

    private(set) lazy var webView: WKWebView = {
        let configuration = WKWebViewConfiguration()
        let view = WKWebView(frame: CGRect(x: 0, y: 0, width: 320, height: 240), configuration: configuration)
        view.isOpaque = false
        view.backgroundColor = .clear
        view.scrollView.backgroundColor = .clear
        view.scrollView.isScrollEnabled = false
        view.scrollView.contentInsetAdjustmentBehavior = .never
        view.isAccessibilityElement = false
        return view
    }()

    func attach(_ coordinator: LobbotView.Coordinator) -> WKWebView {
        webView.configuration.userContentController.removeScriptMessageHandler(forName: "lobbot")
        webView.configuration.userContentController.add(coordinator, name: "lobbot")
        webView.navigationDelegate = coordinator
        if !hasLoaded { load() }
        return webView
    }

    private func load() {
        guard let url = Bundle.main.url(forResource: "lobbot", withExtension: "html", subdirectory: "Lobbot") else {
            fail("Dr. Lobbot’s files are missing from the app.")
            return
        }
        hasLoaded = true
        loadState = .loading
        readinessTask?.cancel()
        webView.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
        readinessTask = Task { [weak self] in
            do { try await Task.sleep(for: .seconds(15)) } catch { return }
            guard let self, self.loadState == .loading else { return }
            self.fail("Dr. Lobbot couldn’t start. Try again.")
        }
    }

    func retry() { load() }

    func show(_ state: LobbotAppState) {
        guard state != currentState else { return }
        currentState = state
        switch state {
        case .idle: stop()
        case .typing, .recording: play(.listen, loop: true)
        case .planning, .requesting: play(.think, loop: true)
        case .streaming: play(.talk, loop: true)
        case .compressing: play(.task, hold: true)
        case .succeeded: finish()
        case .failed: play(.error)
        case .regression: play(.bug)
        case .discovery: play(.eureka)
        case .inactive: play(.sleep, loop: true)
        }
        scheduleInactivity()
    }

    /// Called by native input and by taps on the WebGL canvas.
    func activity() {
        if currentState == .inactive || scene == .sleep { show(.idle) }
        scheduleInactivity()
    }

    private func scheduleInactivity() {
        inactivityTask?.cancel()
        inactivityTask = Task { [weak self] in
            do { try await Task.sleep(for: .seconds(60)) } catch { return }
            guard let self else { return }
            // An active request must finish before the character can fall asleep.
            switch self.currentState {
            case .compressing, .planning, .requesting, .streaming, .recording: return
            default: self.show(.inactive)
            }
        }
    }

    func finish() {
        isHoldingTask = false
        enqueue(.finish)
    }

    func stop() {
        isHoldingTask = false
        enqueue(.stop)
    }

    /// Close-up for full-screen calls, wide when the props need room.
    func setFraming(close: Bool) {
        enqueue(.framing(close: close))
    }

    private func play(_ scene: LobbotScene, loop: Bool = false, hold: Bool = false) {
        isHoldingTask = hold
        enqueue(.play(scene, loop: loop, hold: hold))
    }

    private func enqueue(_ command: Command) {
        commands.append(command)
        drain()
    }

    private func drain() {
        guard loadState == .ready, !isDraining else { return }
        isDraining = true
        Task { [weak self] in
            guard let self else { return }
            defer { self.isDraining = false }
            while self.loadState == .ready, !self.commands.isEmpty {
                let command = self.commands.removeFirst()
                do { _ = try await self.webView.evaluateJavaScript(command.javaScript) }
                catch { self.fail("The animation stopped. Try again.") }
            }
        }
    }

    func receive(_ message: [String: Any]) {
        guard let event = message["event"] as? String else { return }
        logger.info("Bridge: \(event, privacy: .public) \((message["scene"] as? String) ?? "", privacy: .public)")
        switch event {
        case "ready":
            readinessTask?.cancel()
            loadState = .ready
            drain()
            scheduleInactivity()
        case "started":
            if let raw = message["scene"] as? String, let scene = LobbotScene(rawValue: raw) { self.scene = scene }
        case "ended": break
        case "tap":
            UIImpactFeedbackGenerator(style: .light).impactOccurred()
            activity()
        default: break
        }
    }

    func fail(_ message: String) {
        readinessTask?.cancel()
        loadState = .failed(message)
        logger.error("\(message, privacy: .public)")
    }
}

struct LobbotView: UIViewRepresentable {
    @ObservedObject var controller: LobbotController

    func makeCoordinator() -> Coordinator { Coordinator(controller: controller) }
    func makeUIView(context: Context) -> WKWebView { controller.attach(context.coordinator) }
    func updateUIView(_ uiView: WKWebView, context: Context) {}

    final class Coordinator: NSObject, WKScriptMessageHandler, WKNavigationDelegate {
        private weak var controller: LobbotController?
        init(controller: LobbotController) { self.controller = controller }

        func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
            guard let body = message.body as? [String: Any] else { return }
            controller?.receive(body)
        }
        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            controller?.fail("Dr. Lobbot couldn’t load.")
        }
        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            controller?.fail("Dr. Lobbot couldn’t load.")
        }
        func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
            controller?.fail("The animation stopped. Try again.")
        }
        func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction) async -> WKNavigationActionPolicy {
            navigationAction.request.url?.isFileURL == true ? .allow : .cancel
        }
    }
}
