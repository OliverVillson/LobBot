import SwiftUI
import UniformTypeIdentifiers

struct ExperimentSetupView: View {
    @Binding var mode: BrandMode
    @State private var configuration = ExperimentConfiguration()
    @State private var isImporting = false
    @State private var importError: String?
    @State private var editingBenchmark = false
    @State private var showExperiment = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.dynamicTypeSize) private var textSize
    @EnvironmentObject private var lobbot: LobbotController

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 28) {
                    BrandModePicker(mode: $mode)
                    introduction
                    modelSection
                    capabilitySection
                    memorySection
                    benchmarkSection
                }
                .padding(.horizontal, 20)
                .padding(.top, 8)
                .padding(.bottom, 24)
            }
            .background(BrandColor.canvas)
            .foregroundStyle(BrandColor.ink)
            .navigationTitle("lobbot")
            .phoneNavigation()
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    SillonMark().frame(width: 36, height: 36)
                }
            }
            .safeAreaInset(edge: .bottom, spacing: 0) {
                VStack(spacing: 8) {
                    PrimaryAction(title: "Lancer l’expérience", symbol: "arrow.right") {
                        lobbot.activity()
                        showExperiment = true
                    }
                    Text("Simulation visuelle · aucun modèle n’est modifié")
                        .font(.caption2)
                        .foregroundStyle(BrandColor.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(.horizontal, 20)
                .padding(.top, 16)
                .padding(.bottom, 12)
                .background(BrandColor.canvas)
            }
            .navigationDestination(isPresented: $showExperiment) {
                ExperimentDetailView(mode: $mode, configuration: configuration)
            }
            .fileImporter(isPresented: $isImporting, allowedContentTypes: [.data, .folder]) { result in
                switch result {
                case .success(let url):
                    lobbot.activity()
                    configuration.modelName = url.lastPathComponent
                    configuration.isExampleModel = false
                case .failure(let error):
                    lobbot.show(.failed)
                    importError = error.localizedDescription
                }
            }
            .alert("Le fichier n’a pas pu être sélectionné", isPresented: Binding(
                get: { importError != nil }, set: { if !$0 { importError = nil } }
            )) {
                Button("Réessayer") { importError = nil; isImporting = true }
                Button("Annuler", role: .cancel) { importError = nil }
            } message: {
                Text(importError ?? "Sélectionnez de nouveau votre fichier.")
            }
            .sheet(isPresented: $editingBenchmark) {
                BenchmarkEditor(name: $configuration.benchmarkName)
                    .preferredColorScheme(mode.colorScheme)
                    .tint(BrandColor.accent)
            }
            .onChange(of: configuration.capability) { oldValue, newValue in
                if configuration.benchmarkName == oldValue.benchmarkExample {
                    configuration.benchmarkName = newValue.benchmarkExample
                }
            }
        }
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: mode)
        .onChange(of: mode) { lobbot.activity() }
        .onChange(of: configuration.memoryGB) { lobbot.activity() }
        .onChange(of: configuration.capability) { lobbot.activity() }
        .onAppear { lobbot.show(.idle); lobbot.activity() }
    }

    private var introduction: some View {
        HStack(alignment: .center, spacing: 18) {
            VStack(alignment: .leading, spacing: 8) {
                Text("La capacité\nd’abord.")
                    .font(.title.weight(.semibold))
                    .fixedSize(horizontal: false, vertical: true)
                Text("Retirez la complexité.\nGardez la fonction.")
                    .font(.subheadline)
                    .foregroundStyle(BrandColor.secondary)
            }
            Spacer(minLength: 0)
            if !textSize.isAccessibilitySize {
                SillonMark().frame(width: 88, height: 88)
            }
        }
        .padding(.vertical, 6)
    }

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle("Le modèle")
            Button { isImporting = true } label: {
                HStack(spacing: 14) {
                    Image(systemName: "cube")
                        .font(.title2)
                        .foregroundStyle(BrandColor.accent)
                    VStack(alignment: .leading, spacing: 5) {
                        Text(configuration.modelName)
                            .font(.subheadline.weight(.semibold))
                            .lineLimit(2)
                            .multilineTextAlignment(.leading)
                        Text(configuration.isExampleModel ? "Modèle de démonstration" : "Fichier sélectionné · aperçu uniquement")
                            .font(.caption)
                            .foregroundStyle(BrandColor.secondary)
                    }
                    Spacer(minLength: 0)
                    Image(systemName: "chevron.right").font(.caption.weight(.semibold))
                }
                .padding(18)
                .frame(maxWidth: .infinity, alignment: .leading)
                .foregroundStyle(BrandColor.ink)
                .background(BrandColor.surface, in: RoundedRectangle(cornerRadius: 16))
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Choisir un modèle. \(configuration.modelName)")
        }
    }

    private var capabilitySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle("La capacité à préserver")
            Group {
                if textSize.isAccessibilitySize {
                    Picker("Capacité", selection: $configuration.capability) {
                        capabilityOptions
                    }.pickerStyle(.menu)
                } else {
                    Picker("Capacité", selection: $configuration.capability) {
                        capabilityOptions
                    }.pickerStyle(.segmented)
                }
            }
            .frame(minHeight: 44)
            .accessibilityIdentifier("capability-picker")
        }
    }

    private var capabilityOptions: some View {
        ForEach(Capability.allCases) { item in
            Text(item.shortTitle).tag(item)
        }
    }

    private var memorySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                sectionTitle("Mémoire cible")
                Spacer()
                Text("\(Int(configuration.memoryGB)) Go")
                    .font(.headline.monospacedDigit())
            }
            Slider(value: $configuration.memoryGB, in: 2...16, step: 1) {
                Text("Mémoire cible")
            }
            .frame(minHeight: 44)
            .accessibilityValue("\(Int(configuration.memoryGB)) gigaoctets")
            Text("La contrainte guide la recherche de réduction.")
                .font(.caption)
                .foregroundStyle(BrandColor.secondary)
        }
    }

    private var benchmarkSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle("Le test qui décide")
            Button { editingBenchmark = true } label: {
                HStack(spacing: 14) {
                    Image(systemName: "checkmark.shield").font(.title2).foregroundStyle(BrandColor.accent)
                    VStack(alignment: .leading, spacing: 4) {
                        Text(configuration.benchmarkName).font(.subheadline.weight(.semibold))
                        Text("Tester après chaque modification")
                            .font(.caption).foregroundStyle(BrandColor.secondary)
                    }
                    Spacer(minLength: 0)
                    Image(systemName: "slider.horizontal.3").font(.subheadline)
                }
                .padding(18)
                .frame(maxWidth: .infinity, alignment: .leading)
                .foregroundStyle(BrandColor.ink)
                .background(BrandColor.surface, in: RoundedRectangle(cornerRadius: 16))
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Configurer le benchmark. \(configuration.benchmarkName)")
        }
    }

    private func sectionTitle(_ text: String) -> some View {
        Text(text).font(.headline).foregroundStyle(BrandColor.ink)
    }
}

private struct BenchmarkEditor: View {
    @Binding var name: String
    @State private var draft = ""
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var lobbot: LobbotController

    var body: some View {
        NavigationStack {
            Form {
                Section("Nom de la suite") {
                    TextField("Ex. Suite de code", text: $draft)
                        .submitLabel(.done)
                }
                Section {
                    Text("Ce prototype enregistre le nom du test. Il n’exécute pas de benchmark.")
                        .font(.subheadline)
                }
            }
            .navigationTitle("Votre benchmark")
            .phoneNavigation(largeTitle: false)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Annuler") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Enregistrer") {
                        name = draft.trimmingCharacters(in: .whitespacesAndNewlines)
                        dismiss()
                    }
                    .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
            .onAppear { draft = name }
            .onChange(of: draft) { lobbot.activity(); lobbot.show(.typing) }
            .onDisappear { lobbot.show(.idle) }
        }
    }
}

#Preview("A · Anatomie") {
    ExperimentSetupView(mode: .constant(.anatomie))
        .environmentObject(LobbotController())
        .preferredColorScheme(.light).tint(BrandColor.accent)
}

#Preview("B · Soufre") {
    ExperimentSetupView(mode: .constant(.soufre))
        .environmentObject(LobbotController())
        .preferredColorScheme(.dark).tint(BrandColor.accent)
}
