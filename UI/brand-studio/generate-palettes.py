"""Palette applications for the approved Sillon logo; local vector assets only."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
import xml.etree.ElementTree as ET
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools.pens.svgPathPen import SVGPathPen

BASE = Path(__file__).resolve().parent
OUT = BASE / 'dist/assets/palettes'
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = BASE.parent / 'logo-explorations/organic'
FONT_DIR = BASE / 'dist/assets/fonts'
FONT = instantiateVariableFont(TTFont(FONT_DIR / 'Manrope.ttf'), {'wght': 650})
GLYPHS, CMAP = FONT.getGlyphSet(), FONT.getBestCmap()

PALETTES = [
    dict(id='incision', name='Incision', note='Corail chirurgical · précis et provocant', canvas='#F6F0E7', surface='#FFFAF3', ink='#2B231F', secondary='#685A51', line='#CDC0B3', accent='#EE6047', onaccent='#2B231F'),
    dict(id='sauge', name='Sauge', note='Vert végétal · calme et expérimental', canvas='#EEF2E8', surface='#F8FAF2', ink='#233C30', secondary='#526453', line='#BCC7B6', accent='#B9CFA6', onaccent='#233C30'),
    dict(id='anatomie', name='Anatomie', note='Bordeaux et rose · organique et singulier', canvas='#F3E7E6', surface='#FCF4F2', ink='#592C3B', secondary='#805764', line='#D4B5BB', accent='#A13D58', onaccent='#FFFFFF'),
    dict(id='cobalt', name='Cobalt', note='Bleu franc · instrumentation et rigueur', canvas='#EDF1F2', surface='#F8FBFB', ink='#173446', secondary='#526773', line='#C0D0D7', accent='#305AC5', onaccent='#FFFFFF'),
    dict(id='soufre', name='Soufre', note='Jaune acide sur graphite · recherche et énergie', canvas='#20261F', surface='#2B3228', ink='#F1F2DB', secondary='#BBC2AD', line='#555E4E', accent='#DDDE72', onaccent='#20261F'),
    dict(id='archive', name='Archive', note='Charbon et papier · discret et intemporel', canvas='#ECEDE7', surface='#F8F8F0', ink='#292D26', secondary='#626858', line='#C4C8BB', accent='#292D26', onaccent='#F8F8F0'),
]


def text_path(text, x, y, size, color, tracking=0):
    scale = size / FONT['head'].unitsPerEm
    cursor, paths = x, []
    for char in text:
        glyph = GLYPHS[CMAP[ord(char)]]
        pen = SVGPathPen(GLYPHS)
        glyph.draw(pen)
        paths.append(f'<path fill="{color}" transform="translate({cursor:.3f} {y}) scale({scale:.6f} {-scale:.6f})" d="{pen.getCommands()}"/>')
        cursor += glyph.width * scale + tracking
    return ''.join(paths)


symbol = ET.parse(SOURCE / 'm-sillon-symbol-ink.svg').getroot()
logo = (SOURCE / 'm-sillon-logo-ink.svg').read_text()
group = next(child for child in symbol if child.tag.endswith('g'))
SHAPE = ''.join(ET.tostring(child, encoding='unicode') for child in group)


def mark(x, y, size, color):
    return f'<g transform="translate({x} {y}) scale({size / 128})" color="{color}">{SHAPE}</g>'


def luminance(color):
    channels = [int(color[i:i+2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in channels]
    return sum(a * b for a, b in zip(linear, (.2126, .7152, .0722)))


def contrast(a, b):
    hi, lo = sorted([luminance(a), luminance(b)], reverse=True)
    return (hi + .05) / (lo + .05)


report = {}
parts = ['<rect width="1800" height="1510" fill="#F8F7F3"/>', text_path('lobbot', 52, 75, 43, '#292D26'), text_path('Sillon / 6 palettes en action', 1040, 71, 26, '#51574C')]
for i, p in enumerate(PALETTES):
    checks = {
        'text/canvas': contrast(p['ink'], p['canvas']),
        'secondary/canvas': contrast(p['secondary'], p['canvas']),
        'text/surface': contrast(p['ink'], p['surface']),
        'secondary/surface': contrast(p['secondary'], p['surface']),
        'action/accent': contrast(p['onaccent'], p['accent']),
    }
    assert all(ratio >= 4.5 for ratio in checks.values()), (p['name'], checks)
    report[p['id']] = {name: round(value, 2) for name, value in checks.items()}
    (OUT / f"tokens-{p['id']}.json").write_text(json.dumps(p, ensure_ascii=False, indent=2))
    for variant, color in [('ink', p['ink']), ('accent', p['accent']), ('reverse', p['onaccent'])]:
        (OUT / f"sillon-{p['id']}-{variant}.svg").write_text(logo.replace('#191B18', color))
    x, y = 48 + (i % 2) * 864, 118 + (i // 2) * 452
    w, h = 840, 428
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{p["canvas"]}"/>')
    parts.append(mark(x + 24, y + 23, 40, p['ink']))
    parts.append(text_path('lobbot', x + 75, y + 56, 35, p['ink']))
    parts.append(text_path(p['name'], x + 610, y + 54, 24, p['ink']))
    parts.append(text_path('Coupez le modèle.', x + 28, y + 130, 37, p['ink']))
    parts.append(text_path('Gardez la capacité.', x + 28, y + 177, 37, p['ink']))
    parts.append(text_path('Une intelligence réduite à sa mission.', x + 28, y + 218, 18, p['secondary']))
    parts.append(f'<rect x="{x + 28}" y="{y + 244}" width="224" height="48" rx="8" fill="{p["accent"]}" stroke="{p["ink"]}"/>')
    parts.append(text_path('Lancer une expérience', x + 43, y + 275, 16, p['onaccent']))
    parts.append(f'<rect x="{x + 534}" y="{y + 89}" width="278" height="212" rx="13" fill="{p["accent"]}"/>')
    parts.append(mark(x + 597, y + 114, 156, p['onaccent']))
    parts.append(f'<path d="M{x + 28} {y + 323}H{x + 812}" stroke="{p["line"]}"/>')
    parts.append(text_path('Capacité conservée', x + 28, y + 355, 17, p['ink']))
    parts.append(text_path('état illustratif', x + 212, y + 355, 13, p['secondary']))
    for j, token in enumerate(['canvas', 'ink', 'accent']):
        sx = x + 437 + j * 125
        parts.append(f'<circle cx="{sx}" cy="{y + 350}" r="10" fill="{p[token]}" stroke="{p["ink"]}" stroke-width=".7"/>')
        parts.append(text_path(p[token], sx + 17, y + 355, 13, p['ink']))
    parts.append(text_path(p['note'], x + 28, y + 398, 16, p['secondary']))

board = '<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1510" viewBox="0 0 1800 1510" role="img"><title>LOBBOT — Sillon, six palettes applied to logo and interface</title>' + ''.join(parts) + '</svg>'
ET.fromstring(board)
(OUT / 'sillon-palettes.svg').write_text(board)
(OUT / 'palettes.json').write_text(json.dumps(PALETTES, ensure_ascii=False, indent=2))
(OUT / 'contrast.json').write_text(json.dumps(report, indent=2))
(OUT / 'Manrope-OFL.txt').write_text((FONT_DIR / 'Manrope-OFL.txt').read_text())
(OUT / 'README.md').write_text('''# LOBBOT / Sillon

Logo M selected by the user. Its geometry is unchanged.
Six exploratory palettes, with no final color selection yet.

`palettes.json`: canvas, surface, ink, secondary, line, accent, onaccent.
`contrast.json`: sRGB relative luminance checks; all text pairs are at least 4.5:1.
The interface and states are illustrative; no model or benchmark is running.
Fonts: self-hosted Manrope and Martian Mono, under SIL Open Font License.
Board: local SVG geometry and outlined Manrope lettering; rasterized locally.
''')
css = []
for p in PALETTES:
    css.append('[data-palette="' + p['id'] + '"] {\n' + '\n'.join(f'  --{key}: {p[key]};' for key in ['canvas', 'surface', 'ink', 'secondary', 'line', 'accent', 'onaccent']) + '\n}')
(OUT / 'palette-tokens.css').write_text('\n\n'.join(css))
with ZipFile(OUT / 'lobbot-sillon-palettes.zip', 'w', ZIP_DEFLATED) as archive:
    for file in sorted(OUT.iterdir()):
        if file.suffix in ['.svg', '.png', '.json', '.md', '.css', '.txt']:
            archive.write(file, file.name)
print('Six palettes, 18 colored logo SVGs, outlined board, tokens, contrast checks, ZIP.')
