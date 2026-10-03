"""Local, deterministic vector exploration. No network or image generation."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools.pens.svgPathPen import SVGPathPen

ROOT = Path(__file__).resolve().parent
FONTS = ROOT.parent / 'brand-studio/dist/assets/fonts'
INK = '#191B18'
FONT = instantiateVariableFont(TTFont(FONTS / 'Archivo.ttf'), {'wght': 650, 'wdth': 100})
GLYPHS = FONT.getGlyphSet()
CMAP = FONT.getBestCmap()


def lettering(text, x, y, size, color=INK, tracking=0):
    scale = size / FONT['head'].unitsPerEm
    cursor = x
    paths = []
    for char in text:
        glyph = GLYPHS[CMAP[ord(char)]]
        pen = SVGPathPen(GLYPHS)
        glyph.draw(pen)
        paths.append(f'<path fill="{color}" transform="translate({cursor:.3f} {y}) scale({scale:.6f} {-scale:.6f})" d="{pen.getCommands()}"/>')
        cursor += glyph.width * scale + tracking
    return ''.join(paths), cursor - x - tracking


def wordmark(x, y, size, color=INK):
    return lettering('LOBBOT', x, y, size, color, .45)[0]


CONCEPTS = [
    ('D', 'ligature', 'LIGATURE',
     '<path fill-rule="evenodd" d="M8 12H32V92H56V12H88L116 36V52L102 66L120 84V100L98 120H8Z M80 34V56H90L98 48V42L90 34Z M80 78V98H90L100 88L90 78Z"/>'),
    ('E', 'confluence', 'CONFLUENCE',
     '<path d="M8 8H31L73 50H120V74H63L21 32H8Z M8 52H80V76H8Z M8 96H21L63 54H120V78H73L31 120H8Z"/>'),
    ('F', 'decoupe', 'DECOUPE',
     '<path d="M12 12H116V46H80V80H46V116H12Z M84 84H116V116H84Z"/>'),
    ('G', 'circuit', 'CIRCUIT',
     '<path d="M20 12V96Q20 116 40 116H86Q108 116 108 94Q108 72 86 72H60Q38 72 38 50Q38 28 60 28H108" fill="none" stroke="currentColor" stroke-width="22" stroke-linecap="butt" stroke-linejoin="round"/>'),
    ('H', 'strates', 'STRATES',
     '<path d="M10 30L82 8L118 26L46 48Z M10 64L82 42L118 60L46 82Z M10 98L48 86L84 104L46 116Z"/>'),
    ('I', 'fenetre', 'FENETRE',
     '<path d="M60 12H32C18 12 10 22 10 36V92C10 106 18 116 32 116H60V92H38V36H60Z M68 12H96C110 12 118 22 118 36V92C118 106 110 116 96 116H68V92H90V36H68Z"/>'),
]


def mark(shape, x=0, y=0, size=128, color=INK):
    return f'<g transform="translate({x} {y}) scale({size / 128})" fill="{color}" color="{color}">{shape}</g>'


def svg(content, width, height, title):
    result = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img"><title>{title}</title>{content}</svg>'
    ET.fromstring(result)
    return result


for letter, key, label, shape in CONCEPTS:
    for variant, color in [('ink', INK), ('white', '#FFFFFF')]:
        (ROOT / f'{letter.lower()}-{key}-symbol-{variant}.svg').write_text(svg(mark(shape, color=color), 128, 128, f'LOBBOT — {label} — symbol'))
        (ROOT / f'{letter.lower()}-{key}-logo-{variant}.svg').write_text(svg(mark(shape, 0, 0, 96, color) + wordmark(116, 67, 56, color), 359, 96, f'LOBBOT — {label} — logo'))


parts = ['<rect width="1500" height="1110" fill="#F6F6F2"/>']
parts.append(lettering('LOBBOT', 48, 60, 30)[0])
parts.append(lettering('D — I', 1360, 58, 19)[0])
parts.append('<path d="M48 92H1452M500 120V1062M1000 120V1062M48 578H1452" stroke="#D1D3CB" stroke-width="1"/>')
for i, (letter, key, label, shape) in enumerate(CONCEPTS):
    col, row = i % 3, i // 3
    x, y = col * 500, row * 485 + 110
    parts.append(lettering(f'{letter} / {label}', x + 48, y + 35, 15, '#63675E', .9)[0])
    parts.append(mark(shape, x + 153, y + 97, 194))
    _, width = lettering('LOBBOT', 0, 0, 56, tracking=.45)
    parts.append(wordmark(x + (500 - width) / 2, y + 386, 56))
    parts.append(mark(shape, x + 424, y + 414, 26))
    parts.append('<g transform="translate(' + str(x + 48) + ' ' + str(y + 414) + ')"><rect width="34" height="26" rx="2" fill="#191B18"/>')
    parts.append(mark(shape, 5, 1, 24, '#F6F6F2') + '</g>')
(ROOT / 'lobbot-logos-d-i.svg').write_text(svg(''.join(parts), 1500, 1110, 'LOBBOT — Six additional logo directions, D to I'))

# The outlines are derived from the local, SIL Open Font License Archivo typeface.
(ROOT / 'Archivo-OFL.txt').write_text((FONTS / 'Archivo-OFL.txt').read_text())
with ZipFile(ROOT / 'lobbot-six-more-logos.zip', 'w', ZIP_DEFLATED) as archive:
    for file in sorted(ROOT.glob('*.svg')):
        archive.write(file, file.name)
    archive.write(ROOT / 'Archivo-OFL.txt', 'Archivo-OFL.txt')
print('Created six concepts, 24 transparent SVGs, one comparison board, and ZIP.')
