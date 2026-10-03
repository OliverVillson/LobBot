"""Create actual outlined SVG identities and self-hosted webfonts."""
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.varLib.instancer import instantiateVariableFont
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parent / 'dist' / 'assets'
FONTS = ROOT / 'fonts'
LOGOS = ROOT / 'logos'
LOGOS.mkdir(parents=True, exist_ok=True)

for source in FONTS.glob('*.ttf'):
    font = TTFont(source)
    font.flavor = 'woff2'
    font.save(source.with_suffix('.woff2'))

symbols = {
    'incision': '''<defs><clipPath id="top"><path d="M0 0H144V52L0 88Z"/></clipPath><clipPath id="bottom"><path d="M0 101L144 65V144H0Z"/></clipPath></defs><g fill-rule="evenodd"><path clip-path="url(#top)" d="M16 9H70C97 9 111 23 111 44C111 56 104 64 97 68C111 74 117 86 117 100C117 123 99 135 72 135H16Z M42 32V58H65C80 58 87 53 87 45C87 36 79 32 65 32Z M42 81V112H69C85 112 93 107 93 97C93 88 84 81 69 81Z"/><path clip-path="url(#bottom)" d="M16 9H70C97 9 111 23 111 44C111 56 104 64 97 68C111 74 117 86 117 100C117 123 99 135 72 135H16Z M42 32V58H65C80 58 87 53 87 45C87 36 79 32 65 32Z M42 81V112H69C85 112 93 107 93 97C93 88 84 81 69 81Z"/></g>''',
    'core': '<path d="M124 20H20V124H124V88H108V108H36V36H124Z"/><path d="M94 50H50V94H94V79H79V79H65V65H94Z"/><path d="M106 50H124V68H106Z"/>',
    'signal': '<path d="M12 31H24V59H12ZM12 85H24V113H12ZM35 21H47V59H35ZM35 85H47V123H35ZM58 12H70V59H58ZM58 85H70V132H58ZM81 31H93V59H81ZM81 85H93V113H81ZM104 43H116V59H104ZM104 85H116V101H104ZM12 66H132V78H12Z"/>'
}
# A 14 px core at the center remains independent of the removable layers.
symbols['core'] += '<path d="M76 64H90V78H76Z"/>'

configs = {
    'incision': ('Archivo.ttf', {'wght': 800, 'wdth': 100}, '#191B18', '#F04B32'),
    'core': ('Manrope.ttf', {'wght': 750}, '#293326', '#D6E95A'),
    'signal': ('BarlowSemiCondensed.ttf', {}, '#212321', '#FF894C'),
}

def write_svg(path, content, box, title):
    width, height = box
    data = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.3f} {height:.3f}" role="img" aria-labelledby="title"><title id="title">{title}</title>{content}</svg>'
    ElementTree.fromstring(data)
    path.write_text(data)

for key, (file, axes, ink, accent) in configs.items():
    font = TTFont(FONTS / file)
    if axes:
        font = instantiateVariableFont(font, axes)
    glyphs = font.getGlyphSet()
    cmap = font.getBestCmap()
    font_size = 130
    scale = font_size / font['head'].unitsPerEm
    cap = font['OS/2'].sCapHeight * scale
    x = 0
    word_paths = []
    for letter in 'LOBBOT':
        name = cmap[ord(letter)]
        pen = SVGPathPen(glyphs)
        glyphs[name].draw(pen)
        word_paths.append(f'<path transform="translate({x:.3f},{cap:.3f}) scale({scale:.6f},-{scale:.6f})" d="{pen.getCommands()}"/>')
        x += (glyphs[name].width * scale) - 2
    word = ''.join(word_paths)
    for variant, color in [('ink', ink), ('white', '#FFFFFF'), ('accent', accent)]:
        write_svg(LOGOS / f'{key}-symbol-{variant}.svg', f'<g fill="{color}">{symbols[key]}</g>', (144, 144), f'LOBBOT — {key} — symbol')
        write_svg(LOGOS / f'{key}-wordmark-{variant}.svg', f'<g fill="{color}">{word}</g>', (x + 2, cap + 2), f'LOBBOT — {key} — outlined wordmark')
        symbol_scale = (cap + 2) / 144
        offset = cap + 26
        lockup = f'<g fill="{color}"><g transform="scale({symbol_scale:.6f})">{symbols[key]}</g><g transform="translate({offset:.3f},0)">{word}</g></g>'
        write_svg(LOGOS / f'{key}-lockup-{variant}.svg', lockup, (offset + x + 2, cap + 2), f'LOBBOT — {key} — complete logo')

write_svg(LOGOS / 'favicon.svg', '<rect width="144" height="144" rx="20" fill="#F04B32"/><g transform="translate(14,14) scale(.8)" fill="#191B18">'+symbols['incision']+'</g>', (144,144), 'LOBBOT')
write_svg(LOGOS / 'signal-small.svg', '<path fill="#212321" d="M16 22H35V56H16ZM55 12H74V56H55ZM94 34H113V56H94ZM16 68H128V86H16ZM16 98H35V132H16ZM55 98H74V142H55ZM94 98H113V120H94Z"/>', (144,144), 'LOBBOT — Signal — simplified small-size mark')

print(f'Created {len(list(LOGOS.glob("*.svg")))} validated SVG files and {len(list(FONTS.glob("*.woff2")))} webfonts.')
