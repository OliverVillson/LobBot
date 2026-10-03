"""Organic, lobotomy-inspired SVG marks. Local fonts and geometry only."""
from pathlib import Path
from itertools import count
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools.pens.svgPathPen import SVGPathPen

ROOT = Path(__file__).resolve().parent / 'organic'
ROOT.mkdir(exist_ok=True)
FONTS = ROOT.parent.parent / 'brand-studio/dist/assets/fonts'
INK = '#191B18'
FONT = instantiateVariableFont(TTFont(FONTS / 'Manrope.ttf'), {'wght': 650})
GLYPHS, CMAP = FONT.getGlyphSet(), FONT.getBestCmap()
IDS = count()


def lettering(text, x, y, size, color=INK, tracking=0):
    scale = size / FONT['head'].unitsPerEm
    cursor, paths = x, []
    for char in text:
        glyph = GLYPHS[CMAP[ord(char)]]
        pen = SVGPathPen(GLYPHS)
        glyph.draw(pen)
        paths.append(f'<path fill="{color}" transform="translate({cursor:.3f} {y}) scale({scale:.6f} {-scale:.6f})" d="{pen.getCommands()}"/>')
        cursor += glyph.width * scale + tracking
    return ''.join(paths), cursor - x - tracking


CONCEPTS = [
    ('J', 'lobotomie', 'LOBOTOMIE',
     '<path d="M58 14C49 8 35 12 30 24C22 24 16 28 14 35L58 25Z M58 36L12 46C5 55 8 69 16 76C12 87 19 98 30 101C34 114 48 119 58 112Z M70 14C79 8 93 12 98 24C109 24 116 32 115 45C124 54 121 69 112 76C116 87 109 98 98 101C94 114 80 119 70 112Z"/>'),
    ('K', 'lobule', 'LOBULE',
     '<path d="M88 12C50 -1 12 20 10 60C8 98 36 123 76 116C87 114 98 108 107 98C80 102 60 91 55 73C50 54 61 35 81 32C89 31 97 34 104 40C100 27 96 17 88 12Z M91 44C103 47 111 57 109 70C107 83 96 92 83 89C70 85 65 70 70 58C74 48 82 42 91 44Z"/>'),
    ('L', 'fissure', 'FISSURE',
     '<defs><mask id="UID" maskUnits="userSpaceOnUse" x="0" y="0" width="128" height="128"><rect width="128" height="128" fill="white"/><path d="M75 6C60 26 53 50 65 67C77 84 81 99 68 119" fill="none" stroke="black" stroke-width="11"/></mask></defs><path mask="url(#UID)" d="M16 82C3 71 6 51 20 44C18 28 31 16 47 19C57 5 78 7 85 22C104 17 119 31 117 47C128 61 118 79 104 83C97 101 80 111 61 109C47 115 30 104 27 92C23 91 19 88 16 82Z"/>'),
    ('M', 'sillon', 'SILLON',
     '<g fill="none" stroke="currentColor" stroke-width="12" stroke-linecap="round" stroke-linejoin="round"><path d="M53 15C35 9 18 18 21 34C6 37 5 59 20 64C8 77 18 94 34 92C33 109 53 120 65 106C81 119 100 106 97 92C112 95 125 76 113 65C125 53 116 32 102 35C104 20 89 9 74 15"/><path d="M42 37C61 34 63 49 51 56C40 63 39 80 53 86M84 39C73 41 72 56 84 62C97 68 94 83 81 86"/></g>'),
    ('N', 'coupes', 'COUPES',
     '<path d="M13 48C12 29 30 13 54 12C80 8 108 18 116 36C92 49 51 53 13 48Z M12 59C43 67 83 59 121 47C127 64 112 84 92 88C69 95 37 91 15 81C11 75 10 67 12 59Z M20 92C42 103 72 103 92 99C82 118 58 126 37 115C29 111 23 102 20 92Z"/>'),
    ('O', 'b-lobe', 'B / LOBE',
     '<path fill-rule="evenodd" d="M18 14C18 5 38 5 38 14V48C48 31 65 23 86 31C111 39 122 65 117 88C112 111 92 123 67 118C38 123 18 107 18 80Z M54 56C60 45 75 43 82 53C97 54 102 68 94 79C96 90 84 98 74 92C60 100 48 87 53 77C45 69 46 61 54 56Z"/>'),
]


def mark(shape, x=0, y=0, size=128, color=INK):
    shape = shape.replace('UID', f'mark-{next(IDS)}')
    return f'<g transform="translate({x} {y}) scale({size / 128})" fill="{color}" color="{color}">{shape}</g>'


def svg(content, width, height, title):
    result = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img"><title>{title}</title>{content}</svg>'
    ET.fromstring(result)
    return result


_, word_width = lettering('lobbot', 0, 0, 60)
for letter, key, label, shape in CONCEPTS:
    for variant, color in [('ink', INK), ('white', '#FFFFFF')]:
        (ROOT / f'{letter.lower()}-{key}-symbol-{variant}.svg').write_text(svg(mark(shape, color=color), 128, 128, f'LOBBOT — {label} — symbol'))
        logo = mark(shape, 0, 0, 96, color) + lettering('lobbot', 116, 68, 60, color)[0]
        (ROOT / f'{letter.lower()}-{key}-logo-{variant}.svg').write_text(svg(logo, round(116 + word_width + 12), 96, f'LOBBOT — {label} — logo'))

parts = ['<rect width="1500" height="1110" fill="#F6F6F2"/>', lettering('LOBBOT', 48, 60, 29)[0], lettering('J — O', 1357, 58, 19)[0], '<path d="M48 92H1452M500 120V1062M1000 120V1062M48 578H1452" stroke="#D1D3CB" stroke-width="1"/>']
for i, (letter, key, label, shape) in enumerate(CONCEPTS):
    x, y = (i % 3) * 500, (i // 3) * 485 + 110
    parts.append(lettering(f'{letter} / {label}', x + 48, y + 35, 15, '#63675E', .7)[0])
    parts.append(mark(shape, x + 153, y + 97, 194))
    parts.append(lettering('lobbot', x + (500 - word_width) / 2, y + 386, 60)[0])
    parts.append(mark(shape, x + 424, y + 414, 26))
    parts.append(f'<circle cx="{x + 63}" cy="{y + 426}" r="18" fill="{INK}"/>' + mark(shape, x + 51, y + 414, 24, '#F6F6F2'))
(ROOT / 'lobbot-organic-logos-j-o.svg').write_text(svg(''.join(parts), 1500, 1110, 'LOBBOT — Six organic, lobotomy-inspired logo directions'))

(ROOT / 'Manrope-OFL.txt').write_text((FONTS / 'Manrope-OFL.txt').read_text())
with ZipFile(ROOT / 'lobbot-organic-logos.zip', 'w', ZIP_DEFLATED) as archive:
    for file in sorted(ROOT.glob('*.svg')):
        archive.write(file, file.name)
    archive.write(ROOT / 'Manrope-OFL.txt', 'Manrope-OFL.txt')
print('Six organic concepts; 24 transparent SVGs; comparison board; ZIP.')
