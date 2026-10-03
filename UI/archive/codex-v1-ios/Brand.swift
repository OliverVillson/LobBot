import SwiftUI

enum BrandMode: String, CaseIterable, Identifiable {
    case anatomie
    case soufre

    var id: String { rawValue }
    var title: String { self == .anatomie ? "A · Anatomie" : "B · Soufre" }
    var colorScheme: ColorScheme { self == .anatomie ? .light : .dark }
}

/// Asset colors provide the approved Anatomie (light) and Soufre (dark) roles.
enum BrandColor {
    static var canvas: Color { Color("Canvas") }
    static var surface: Color { Color("Surface") }
    static var ink: Color { Color("Ink") }
    static var secondary: Color { Color("Secondary") }
    static var line: Color { Color("Line") }
    static var accent: Color { Color("AccentColor") }
    static var onAccent: Color { Color("OnAccent") }
    static var success: Color { Color("Success") }
    static var restoration: Color { Color("Restoration") }
}

struct BrandModePicker: View {
    @Binding var mode: BrandMode
    @Environment(\.dynamicTypeSize) private var textSize

    var body: some View {
        Group {
            if textSize.isAccessibilitySize {
                Picker("Palette", selection: $mode) {
                    options
                }
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                Picker("Palette", selection: $mode) {
                    options
                }
                .pickerStyle(.segmented)
            }
        }
        .frame(minHeight: 44)
        .accessibilityIdentifier("brand-mode-picker")
    }

    private var options: some View {
        ForEach(BrandMode.allCases) { item in
            Text(item.title).tag(item)
        }
    }
}

/// The exact open contour and cortical strokes of the approved M / Sillon SVG.
nonisolated struct SillonShape: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: 53, y: 15))
        curve(&path, to: (21, 34), first: (35, 9), second: (18, 18))
        curve(&path, to: (20, 64), first: (6, 37), second: (5, 59))
        curve(&path, to: (34, 92), first: (8, 77), second: (18, 94))
        curve(&path, to: (65, 106), first: (33, 109), second: (53, 120))
        curve(&path, to: (97, 92), first: (81, 119), second: (100, 106))
        curve(&path, to: (113, 65), first: (112, 95), second: (125, 76))
        curve(&path, to: (102, 35), first: (125, 53), second: (116, 32))
        curve(&path, to: (74, 15), first: (104, 20), second: (89, 9))
        path.move(to: CGPoint(x: 42, y: 37))
        curve(&path, to: (51, 56), first: (61, 34), second: (63, 49))
        curve(&path, to: (53, 86), first: (40, 63), second: (39, 80))
        path.move(to: CGPoint(x: 84, y: 39))
        curve(&path, to: (84, 62), first: (73, 41), second: (72, 56))
        curve(&path, to: (81, 86), first: (97, 68), second: (94, 83))
        let transform = CGAffineTransform(scaleX: rect.width / 128, y: rect.height / 128)
            .concatenating(CGAffineTransform(translationX: rect.minX, y: rect.minY))
        return path.applying(transform).strokedPath(
            StrokeStyle(lineWidth: min(rect.width, rect.height) * 12 / 128, lineCap: .round, lineJoin: .round)
        )
    }

    private func curve(_ path: inout Path, to: (CGFloat, CGFloat), first: (CGFloat, CGFloat), second: (CGFloat, CGFloat)) {
        path.addCurve(
            to: CGPoint(x: to.0, y: to.1),
            control1: CGPoint(x: first.0, y: first.1),
            control2: CGPoint(x: second.0, y: second.1)
        )
    }
}

struct SillonMark: View {
    var body: some View {
        SillonShape()
            .fill(BrandColor.accent)
            .aspectRatio(1, contentMode: .fit)
            .accessibilityHidden(true)
    }
}

struct PrimaryAction: View {
    let title: String
    let symbol: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Text(title)
                Image(systemName: symbol)
            }
            .font(.headline)
            .frame(maxWidth: .infinity, minHeight: 54)
            .padding(.horizontal, 16)
            .foregroundStyle(BrandColor.onAccent)
            .background(BrandColor.accent, in: RoundedRectangle(cornerRadius: 14))
        }
        .buttonStyle(.plain)
    }
}

extension View {
    @ViewBuilder
    func phoneNavigation(largeTitle: Bool = true) -> some View {
        #if os(iOS)
        self.navigationBarTitleDisplayMode(largeTitle ? .large : .inline)
            .toolbarBackground(BrandColor.canvas, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
        #else
        self
        #endif
    }
}
