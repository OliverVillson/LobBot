import Foundation

enum Capability: String, CaseIterable, Identifiable {
    case code
    case mathematics
    case classification

    var id: String { rawValue }
    var shortTitle: String {
        switch self {
        case .code: "Code"
        case .mathematics: "Maths"
        case .classification: "Classer"
        }
    }
    var title: String {
        switch self {
        case .code: "Code"
        case .mathematics: "Raisonnement mathématique"
        case .classification: "Classification"
        }
    }
    var symbol: String {
        switch self {
        case .code: "chevron.left.forwardslash.chevron.right"
        case .mathematics: "function"
        case .classification: "square.grid.2x2"
        }
    }
    var benchmarkExample: String {
        switch self {
        case .code: "Suite de code · démo"
        case .mathematics: "Suite de maths · démo"
        case .classification: "Suite de classification · démo"
        }
    }
}

struct ExperimentConfiguration {
    var modelName = "base-model.safetensors"
    var isExampleModel = true
    var capability: Capability = .code
    var memoryGB: Double = 8
    var benchmarkName = "Suite de code · démo"
}

enum ExperimentPhase: Equatable {
    case running(step: Int)
    case retained
    case restored
    case stopped

    var isRunning: Bool {
        if case .running = self { return true }
        return false
    }

    var completedSteps: Int {
        switch self {
        case .running(let step): step
        case .retained, .restored: 4
        case .stopped: 0
        }
    }

    var title: String {
        switch self {
        case .running: "Évaluation en cours"
        case .retained: "La capacité est conservée."
        case .restored: "La modification est restaurée."
        case .stopped: "L’expérience est arrêtée."
        }
    }

    var message: String {
        switch self {
        case .running: "Une modification est proposée, puis testée sur la capacité choisie."
        case .retained: "Le candidat satisfait le critère d’exemple. La réduction peut être conservée."
        case .restored: "Le candidat échoue au test d’exemple. L’état précédent du modèle est rétabli."
        case .stopped: "Relancez la simulation pour reprendre le suivi."
        }
    }
}
