import SwiftUI

// MARK: - Intake: the patient's chart, one question at a time

struct IntakePanel: View {
    let step: IntakeStep
    @Environment(Session.self) private var session
    @State private var importing = false

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Step \(step.rawValue + 1) of \(IntakeStep.allCases.count)")
                        .font(.footnote.weight(.semibold))
                        .foregroundStyle(Brand.secondary)
                    Text(step.title)
                        .font(.title2.weight(.bold))
                        .foregroundStyle(Brand.ink)
                }
                Spacer()
                HStack(spacing: 4) {
                    ForEach(IntakeStep.allCases, id: \.self) { s in
                        Capsule()
                            .fill(s.rawValue <= step.rawValue ? Brand.accent : Brand.line)
                            .frame(width: s == step ? 18 : 8, height: 8)
                    }
                }
                .accessibilityHidden(true)
            }

            switch step {
            case .patient: patient
            case .capability: capabilities
            case .limits: limits
            case .chart: chart
            }

            HStack(spacing: 10) {
                Button { session.back() } label: {
                    Image(systemName: "chevron.left")
                        .font(.headline)
                        .frame(width: 54, height: 54)
                        .foregroundStyle(Brand.ink)
                        .background(Brand.tile, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Brand.line))
                }
                .buttonStyle(.plain)
                .accessibilityLabel(step == .patient ? "Home" : "Back")
                PrimaryButton(title: step == .chart ? "Start surgery" : "Continue",
                              symbol: step == .chart ? "scissors" : "arrow.right") { session.next() }
            }
        }
        .card()
        .fileImporter(isPresented: $importing, allowedContentTypes: [.item]) { result in
            if case .success(let url) = result {
                session.fileName = url.lastPathComponent
                session.touched()
            }
        }
    }

    @ViewBuilder private var patient: some View {
        Field(label: "Model size") {
            HStack(spacing: 10) {
                ForEach([70.0, 32, 8], id: \.self) { size in
                    OptionTile(title: "\(Int(size))B", detail: "\(Int(size * 2)) GB", selected: session.sizeB == size) {
                        session.sizeB = size
                        session.touched()
                    }
                }
            }
        }
        Field(label: "Weights") {
            Button { importing = true } label: {
                HStack(spacing: 12) {
                    Image(systemName: "doc.badge.plus").font(.title3).foregroundStyle(Brand.accent)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(session.patientName)
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(Brand.ink)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Text(session.fileName == nil ? "Example file. Tap to choose yours." : "Only the file name is read.")
                            .font(.caption)
                            .foregroundStyle(Brand.secondary)
                    }
                    Spacer(minLength: 0)
                    Image(systemName: "chevron.right").font(.footnote.weight(.semibold)).foregroundStyle(Brand.secondary)
                }
                .padding(14)
                .background(Brand.tile, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            }
            .buttonStyle(.plain)
        }
    }

    private var capabilities: some View {
        Field(label: "Capability to preserve") {
            LazyVGrid(columns: [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)], spacing: 10) {
                ForEach(Capability.allCases) { c in
                    OptionTile(title: c.title, detail: "Demo eval · \(Session.format(c.baseline))",
                               symbol: c.symbol, selected: session.capability == c) {
                        session.capability = c
                        session.touched()
                    }
                }
            }
        }
    }

    @ViewBuilder private var limits: some View {
        @Bindable var session = session
        Field(label: "Runs on") {
            LazyVGrid(columns: [GridItem(.flexible(), spacing: 10), GridItem(.flexible(), spacing: 10)], spacing: 10) {
                ForEach(Hardware.allCases) { h in
                    OptionTile(title: h.title, detail: "\(Int(h.memoryGB)) GB memory",
                               symbol: h.symbol, selected: session.hardware == h) {
                        session.hardware = h
                        session.touched()
                    }
                }
            }
        }
        Field(label: "Max capability loss") {
            HStack(spacing: 12) {
                Slider(value: $session.maxLoss, in: 1...10, step: 0.5) {
                    Text("Max capability loss")
                } onEditingChanged: { editing in
                    if editing { session.touched() }
                }
                Text(Session.percent(session.maxLoss))
                    .font(.headline)
                    .monospacedDigit()
                    .foregroundStyle(Brand.ink)
                    .frame(width: 52, alignment: .trailing)
            }
        }
        Label("Target: ≤ \(Session.format(session.targetB))B params · \(Session.format(session.targetB * 2)) GB",
              systemImage: "scope")
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(Brand.accent)
    }

    /// The doctor's clipboard: a summary to sign before surgery.
    private var chart: some View {
        VStack(alignment: .leading, spacing: 0) {
            ChartRow(label: "Patient", value: session.patientName)
            Divider()
            ChartRow(label: "Size", value: "\(Session.format(session.sizeB))B params · \(Session.format(session.sizeB * 2)) GB")
            Divider()
            ChartRow(label: "Preserve", value: "\(session.capability.title), at most −\(Session.percent(session.maxLoss))")
            Divider()
            ChartRow(label: "Runs on", value: "\(session.hardware.title) · \(Int(session.hardware.memoryGB)) GB")
            Divider()
            ChartRow(label: "Target", value: "≤ \(Session.format(session.targetB))B params")
        }
        .padding(.horizontal, 16)
        .padding(.top, 18)
        .padding(.bottom, 6)
        .background(Color.white, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(Brand.line))
        .overlay(alignment: .top) {
            Capsule().fill(Brand.ink).frame(width: 76, height: 14).offset(y: -7)   // the clip
        }
        .padding(.top, 8)
    }
}

private struct ChartRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            Text(label)
                .font(.subheadline)
                .foregroundStyle(Brand.secondary)
                .frame(width: 76, alignment: .leading)
            Text(value)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Brand.ink)
                .lineLimit(2)
                .truncationMode(.middle)
            Spacer(minLength: 0)
        }
        .padding(.vertical, 11)
        .accessibilityElement(children: .combine)
    }
}

// MARK: - Planning

struct PlanningPanel: View {
    @Environment(Session.self) private var session

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Pre-op plan").font(.title2.weight(.bold)).foregroundStyle(Brand.ink)
            ForEach(Array(steps.enumerated()), id: \.offset) { i, text in
                HStack(spacing: 12) {
                    Group {
                        if session.planDone > i {
                            Image(systemName: "checkmark.circle.fill").foregroundStyle(Brand.kept)
                        } else if session.planDone == i {
                            ProgressView().tint(Brand.accent)
                        } else {
                            Image(systemName: "circle").foregroundStyle(Brand.line)
                        }
                    }
                    .frame(width: 24)
                    Text(text)
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(session.planDone >= i ? Brand.ink : Brand.secondary)
                }
                .accessibilityElement(children: .combine)
            }
        }
        .card()
    }

    private var steps: [String] {
        ["Map \(session.layers) layers of \(session.patientName)",
         "Score each block's impact on \(session.capability.title)",
         "Plan cuts down to \(Session.format(session.targetB))B params"]
    }
}

// MARK: - Surgery: vitals and the live log of cuts

struct SurgeryPanel: View {
    @Environment(Session.self) private var session

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Label("In surgery", systemImage: "cross.case.fill")
                    .font(.headline)
                    .foregroundStyle(Brand.accent)
                Spacer()
                Button("Stop") { session.stop() }
                    .font(.subheadline.weight(.semibold))
                    .buttonStyle(.bordered)
                    .tint(Brand.restored)
            }

            HStack(spacing: 10) {
                Vital(label: "Size", value: "\(Session.format(session.params))B",
                      detail: "target \(Session.format(session.targetB))B")
                Vital(label: session.capability.title, value: "\(Session.format(session.retained))%",
                      detail: "floor \(Session.format(100 - session.maxLoss))%",
                      alert: session.retained < 100 - session.maxLoss * 0.7)
                Vital(label: "Cuts", value: "\(session.kept)",
                      detail: "\(session.restored) restored")
            }

            ProgressView(value: session.sizeProgress)
                .tint(Brand.accent)
                .accessibilityLabel("Size reduction progress")

            if session.cuts.isEmpty {
                HStack(spacing: 10) {
                    ProgressView().tint(Brand.accent)
                    Text("Scrubbing in…").font(.subheadline).foregroundStyle(Brand.secondary)
                }
            } else {
                VStack(spacing: 0) {
                    ForEach(session.cuts) { cut in
                        CutRow(cut: cut)
                            .transition(.move(edge: .top).combined(with: .opacity))
                        if cut.id != session.cuts.last?.id { Divider() }
                    }
                }
                .animation(.snappy(duration: 0.3), value: session.cuts.count)
            }

            Text("Every cut is tested on \(session.capability.title). Damaging cuts are undone. Simulated run: no model is modified.")
                .font(.caption)
                .foregroundStyle(Brand.secondary)
        }
        .card()
    }
}

private struct Vital: View {
    let label: String
    let value: String
    let detail: String
    var alert = false

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label).font(.caption.weight(.semibold)).foregroundStyle(Brand.secondary).lineLimit(1)
            Text(value)
                .font(.title3.weight(.bold))
                .monospacedDigit()
                .foregroundStyle(alert ? Brand.restored : Brand.ink)
                .contentTransition(.numericText())
                .animation(.snappy, value: value)
            Text(detail).font(.caption2).foregroundStyle(Brand.secondary).lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Brand.tile, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .accessibilityElement(children: .combine)
    }
}

private struct CutRow: View {
    let cut: Cut

    var body: some View {
        HStack(spacing: 12) {
            icon.frame(width: 24)
            VStack(alignment: .leading, spacing: 2) {
                Text(cut.action).font(.subheadline.weight(.medium)).foregroundStyle(Brand.ink)
                Text(status).font(.caption).foregroundStyle(color)
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 10)
        .accessibilityElement(children: .combine)
    }

    @ViewBuilder private var icon: some View {
        switch cut.outcome {
        case .testing: ProgressView().tint(Brand.accent)
        case .kept: Image(systemName: "checkmark.circle.fill").foregroundStyle(Brand.kept)
        case .restored: Image(systemName: "arrow.uturn.backward.circle.fill").foregroundStyle(Brand.restored)
        }
    }

    private var status: String {
        switch cut.outcome {
        case .testing: "Testing the capability…"
        case .kept: "Kept · \(Session.signed(cut.delta)) pts"
        case .restored: "Undone · the test fell \(Session.signed(cut.delta)) pts"
        }
    }

    private var color: Color {
        switch cut.outcome {
        case .testing: Brand.secondary
        case .kept: Brand.kept
        case .restored: Brand.restored
        }
    }
}

// MARK: - Result

struct ResultPanel: View {
    @Environment(Session.self) private var session

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Surgery complete").font(.footnote.weight(.semibold)).foregroundStyle(Brand.kept)
                Text("\(Session.format(session.ratio))× lighter")
                    .font(.system(size: 40, weight: .heavy))
                    .foregroundStyle(Brand.ink)
                Text("\(session.capability.title) kept at \(Session.format(session.retained))% of the original.")
                    .foregroundStyle(Brand.secondary)
            }

            HStack(spacing: 10) {
                SizeBox(title: "Before", params: session.sizeB)
                Image(systemName: "arrow.right").foregroundStyle(Brand.secondary).accessibilityHidden(true)
                SizeBox(title: "After", params: session.targetB, highlighted: true)
            }

            VStack(spacing: 0) {
                ResultRow(symbol: session.hardware.symbol,
                          text: "\(session.hardware.fits): \(Session.format(session.targetB * 2)) of \(Int(session.hardware.memoryGB)) GB")
                Divider()
                ResultRow(symbol: "scissors", text: "\(session.kept) cuts kept, \(session.restored) undone")
            }

            HStack(spacing: 10) {
                ShareLink(item: session.report) {
                    Label("Share report", systemImage: "square.and.arrow.up")
                        .font(.headline)
                        .frame(maxWidth: .infinity, minHeight: 50)
                }
                .buttonStyle(.bordered)
                .tint(Brand.accent)
                PrimaryButton(title: "New patient", symbol: "plus") { session.newPatient() }
            }

            Text("Simulated run: no model was modified.")
                .font(.caption)
                .foregroundStyle(Brand.secondary)
        }
        .card()
    }
}

private struct SizeBox: View {
    let title: String
    let params: Double
    var highlighted = false

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption.weight(.semibold)).opacity(0.85)
            Text("\(Session.format(params))B").font(.title2.weight(.bold)).monospacedDigit()
            Text("\(Session.format(params * 2)) GB").font(.caption).opacity(0.85)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .foregroundStyle(highlighted ? Color.white : Brand.ink)
        .background(highlighted ? Brand.accent : Brand.tile, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .accessibilityElement(children: .combine)
    }
}

private struct ResultRow: View {
    let symbol: String
    let text: String

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: symbol).foregroundStyle(Brand.accent).frame(width: 24)
            Text(text).font(.subheadline.weight(.medium)).foregroundStyle(Brand.ink)
            Spacer(minLength: 0)
        }
        .padding(.vertical, 10)
        .accessibilityElement(children: .combine)
    }
}
