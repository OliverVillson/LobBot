import SwiftUI

/// Anatomie palette: bordeaux and rose, light only.
enum Brand {
    static let canvas = Color(hex: 0xF3E7E6)
    static let surface = Color(hex: 0xFCF4F2)
    static let ink = Color(hex: 0x592C3B)
    static let secondary = Color(hex: 0x805764)
    static let line = Color(hex: 0xD4B5BB)
    static let accent = Color(hex: 0xA13D58)
    static let rose = Color(hex: 0xF4B8C4)
    static let stageLight = Color(hex: 0xAD5A70)
    static let stage = Color(hex: 0x96445A)
    static let stageDeep = Color(hex: 0x7A3247)
    static let kept = Color(hex: 0x245C3D)
    static let restored = Color(hex: 0x853E32)
    static let tile = Color(hex: 0xF3E7E6).opacity(0.7)
}

extension Color {
    init(hex: UInt32) {
        self.init(red: Double((hex >> 16) & 0xFF) / 255, green: Double((hex >> 8) & 0xFF) / 255, blue: Double(hex & 0xFF) / 255)
    }
}

/// The approved M / Sillon mark, filled (an outline of the strokes).
nonisolated struct SillonShape: Shape {
    func path(in rect: CGRect) -> Path {
        SillonLine().path(in: rect).strokedPath(
            StrokeStyle(lineWidth: min(rect.width, rect.height) * 12 / 128, lineCap: .round, lineJoin: .round)
        )
    }
}

/// The centre line of the Sillon SVG (open contour + two cortical strokes), so it can be drawn with `trim`.
nonisolated struct SillonLine: Shape {
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
        return path.applying(transform)
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
    var color: Color = Brand.accent
    var body: some View {
        SillonShape()
            .fill(color)
            .aspectRatio(1, contentMode: .fit)
            .accessibilityHidden(true)
    }
}

struct PrimaryButton: View {
    let title: String
    let symbol: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Text(title)
                Image(systemName: symbol)
            }
            .font(.headline)
            .frame(maxWidth: .infinity, minHeight: 54)
            .padding(.horizontal, 16)
            .foregroundStyle(.white)
            .background(Brand.accent, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        }
        .buttonStyle(.plain)
    }
}

/// A labelled group inside a card.
struct Field<Content: View>: View {
    let label: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(label.uppercased())
                .font(.caption.weight(.semibold))
                .tracking(0.6)
                .foregroundStyle(Brand.secondary)
            content
        }
    }
}

/// A selectable tile (model size, capability, hardware).
struct OptionTile: View {
    let title: String
    var detail: String? = nil
    var symbol: String? = nil
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 6) {
                if let symbol { Image(systemName: symbol).font(.title3) }
                Text(title).font(.headline)
                if let detail { Text(detail).font(.caption).opacity(0.85) }
            }
            .frame(maxWidth: .infinity, minHeight: 64, alignment: .leading)
            .padding(12)
            .foregroundStyle(selected ? Color.white : Brand.ink)
            .background(selected ? Brand.accent : Brand.tile, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 14, style: .continuous).stroke(selected ? Color.clear : Brand.line))
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

extension View {
    /// The rose paper card every panel sits on.
    func card() -> some View {
        padding(18)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Brand.surface, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 22, style: .continuous).stroke(Brand.line, lineWidth: 1))
    }
}
