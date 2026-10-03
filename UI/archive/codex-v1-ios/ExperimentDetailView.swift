import SwiftUI

struct ExperimentDetailView: View {
    @Binding var mode: BrandMode
    let configuration: ExperimentConfiguration
    @State private var phase: ExperimentPhase = .running(step: 0)
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @EnvironmentObject private var lobbot: LobbotController

    private let steps = ["Proposer une réduction", "Modifier le candidat", "Évaluer la capacité", "Conserver ou restaurer"]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                BrandModePicker(mode: $mode)
                MascotCard()
                resultHeader
                comparison
                experimentTimeline
                experimentSummary
                if !phase.isRunning {
                    Button {
                        phase = phase == .restored ? .retained : .restored
                    } label: {
                        Label(phase == .restored ? "Voir un résultat conservé" : "Voir une restauration", systemImage: "arrow.uturn.backward")
                            .font(.subheadline.weight(.semibold))
                            .frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(BrandColor.accent)
                }
                Text("Simulation visuelle. Les volumes et décisions montrés sont illustratifs ; aucun modèle ni benchmark n’est exécuté.")
                    .font(.caption)
                    .foregroundStyle(BrandColor.secondary)
            }
            .padding(.horizontal, 20)
            .padding(.top, 8)
            .padding(.bottom, 24)
        }
        .background(BrandColor.canvas)
        .foregroundStyle(BrandColor.ink)
        .navigationTitle("Expérience")
        .phoneNavigation(largeTitle: false)
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                SillonMark().frame(width: 30, height: 30)
            }
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            PrimaryAction(
                title: phase.isRunning ? "Arrêter la simulation" : "Relancer l’expérience",
                symbol: phase.isRunning ? "stop.fill" : "arrow.clockwise"
            ) {
                phase = phase.isRunning ? .stopped : .running(step: 0)
            }
            .padding(.horizontal, 20)
            .padding(.top, 16)
            .padding(.bottom, 16)
            .background(BrandColor.canvas)
        }
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: mode)
        .onChange(of: mode) { lobbot.activity() }
        .onChange(of: phase, initial: true) {
            switch phase {
            case .running(let step): lobbot.show(step == 0 ? .planning : .compressing)
            case .retained: lobbot.show(.succeeded)
            case .restored: lobbot.show(.regression)
            case .stopped: lobbot.show(.idle)
            }
        }
        .onDisappear { lobbot.show(.idle) }
        .task(id: phase.isRunning) {
            guard phase.isRunning else { return }
            do {
                for step in 1...4 {
                    try await Task.sleep(for: .seconds(2))
                    guard !Task.isCancelled, phase.isRunning else { return }
                    phase = .running(step: step)
                }
                phase = .retained
            } catch {
                // Cancellation belongs to stopping or leaving this screen.
            }
        }
    }

    private var resultHeader: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: phase.isRunning ? "waveform.path" : phase == .retained ? "checkmark.circle.fill" : "arrow.uturn.backward.circle.fill")
                    .foregroundStyle(statusColor)
                Text(phase.isRunning ? "Test en cours" : phase == .retained ? "Réduction conservée" : phase == .restored ? "État précédent rétabli" : "Simulation arrêtée")
                    .font(.subheadline.weight(.semibold))
            }
            .accessibilityElement(children: .combine)
            Text(phase.title)
                .font(.title.weight(.semibold))
                .fixedSize(horizontal: false, vertical: true)
            Text(phase.message)
                .font(.subheadline)
                .foregroundStyle(BrandColor.secondary)
        }
        .accessibilityIdentifier("experiment-result")
    }

    private var statusColor: Color {
        switch phase {
        case .retained: BrandColor.success
        case .restored: BrandColor.restoration
        case .running, .stopped: BrandColor.accent
        }
    }

    private var comparison: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Structure du modèle · exemple")
                .font(.headline)
            modelBar(title: "Initial", width: 1, accent: false)
            modelBar(title: phase == .restored ? "Restauré" : "Candidat", width: candidateWidth, accent: true)
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: configuration.capability.symbol)
                Text("Capacité à préserver : \(configuration.capability.title)")
                    .font(.subheadline)
            }
            .foregroundStyle(BrandColor.ink)
        }
        .padding(20)
        .background(BrandColor.surface, in: RoundedRectangle(cornerRadius: 16))
    }

    private var candidateWidth: CGFloat {
        switch phase {
        case .retained: 0.48
        case .restored, .stopped: 1
        case .running(let step): max(0.48, 1 - CGFloat(step) * 0.13)
        }
    }

    private func modelBar(title: String, width: CGFloat, accent: Bool) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.caption).foregroundStyle(BrandColor.secondary)
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule().fill(BrandColor.line)
                    Capsule().fill(accent ? BrandColor.accent : BrandColor.secondary)
                        .frame(width: geometry.size.width * width)
                }
            }
            .frame(height: 12)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(title), volume illustratif \(Int(width * 100)) pour cent du modèle initial")
    }

    private var experimentTimeline: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Chaque changement est un test.").font(.headline)
            ForEach(Array(steps.enumerated()), id: \.offset) { index, title in
                HStack(alignment: .top, spacing: 12) {
                    Image(systemName: index < phase.completedSteps ? "checkmark.circle.fill" : "circle")
                        .foregroundStyle(index < phase.completedSteps ? BrandColor.accent : BrandColor.secondary)
                        .font(.body)
                    Text(title).font(.subheadline)
                    Spacer(minLength: 0)
                    if case .running(let step) = phase, index == step {
                        Text("En cours").font(.caption).foregroundStyle(BrandColor.secondary)
                    }
                }
                .accessibilityElement(children: .combine)
            }
        }
    }

    private var experimentSummary: some View {
        VStack(spacing: 14) {
            summaryRow("Modèle", value: configuration.modelName)
            Divider().overlay(BrandColor.line)
            summaryRow("Test", value: configuration.benchmarkName)
            Divider().overlay(BrandColor.line)
            summaryRow("Mémoire cible", value: "\(Int(configuration.memoryGB)) Go")
        }
        .padding(18)
        .background(BrandColor.surface, in: RoundedRectangle(cornerRadius: 16))
    }

    private func summaryRow(_ title: String, value: String) -> some View {
        HStack(alignment: .top, spacing: 16) {
            Text(title).font(.caption).foregroundStyle(BrandColor.secondary)
            Spacer(minLength: 0)
            Text(value).font(.caption.weight(.semibold)).multilineTextAlignment(.trailing)
        }
        .accessibilityElement(children: .combine)
    }
}

#Preview("Expérience · Anatomie") {
    NavigationStack {
        ExperimentDetailView(mode: .constant(.anatomie), configuration: ExperimentConfiguration())
    }
    .environmentObject(LobbotController())
    .preferredColorScheme(.light).tint(BrandColor.accent)
}

#Preview("Expérience · Soufre") {
    NavigationStack {
        ExperimentDetailView(mode: .constant(.soufre), configuration: ExperimentConfiguration())
    }
    .environmentObject(LobbotController())
    .preferredColorScheme(.dark).tint(BrandColor.accent)
}
