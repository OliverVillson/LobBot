from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools.pens.svgPathPen import SVGPathPen
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parent / 'dist' / 'assets'
font = instantiateVariableFont(TTFont(root/'fonts/Archivo.ttf'), {'wght':600,'wdth':100})
glyphs = font.getGlyphSet()
cmap = font.getBestCmap()
def lettering(text, x, y, size, color='#191B18'):
    scale = size / font['head'].unitsPerEm
    paths=[]
    cursor=x
    for ch in text:
        g=glyphs[cmap[ord(ch)]]
        pen=SVGPathPen(glyphs);g.draw(pen)
        paths.append(f'<path fill="{color}" transform="translate({cursor},{y}) scale({scale},-{scale})" d="{pen.getCommands()}"/>')
        cursor+=g.width*scale
    return ''.join(paths)
def embed(file,x,y,w,h):
    svg=ET.parse(root/'logos'/file).getroot()
    children=''.join(ET.tostring(child,encoding='unicode') for child in svg if child.tag.split('}')[-1]!='title')
    return f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewBox="{svg.attrib["viewBox"]}">{children}</svg>'

parts=['<rect width="1240" height="820" fill="#F5F5F1"/>', embed('incision-wordmark-ink.svg',40,40,220,48),lettering('Identity exploration / October 2026',775,73,18),'<path d="M40 114H1200" stroke="#C9CDC5"/>']
data=[('incision','A / INCISION','#F04B32','#191B18','#F5F5F1','#C9CDC5','Cut the model.','Keep the capability.','Archivo + Martian Mono'),('core','B / NOYAU','#D6E95A','#293326','#F3F5ED','#BAC4AE','Less model.','Same mission.','Manrope + Martian Mono'),('signal','C / SIGNAL','#FF894C','#212321','#EFF1EF','#BFC5BF','Intelligence,','reduced to purpose.','Barlow + Martian Mono')]
for i,(key,label,accent,ink,paper,muted,t1,t2,face) in enumerate(data):
    x=40+i*395
    parts.append(f'<rect x="{x}" y="150" width="370" height="370" fill="{accent}"/>')
    parts.append(embed(f'{key}-symbol-ink.svg',x+93,195,184,184))
    parts.append(embed(f'{key}-wordmark-ink.svg',x+34,432,302,55))
    parts.append(lettering(label,x,567,24,ink))
    parts.append(lettering(t1,x,605,21,ink));parts.append(lettering(t2,x,632,21,ink))
    for j,color in enumerate([accent,ink,paper,muted]):
        parts.append(f'<rect x="{x+j*92.5}" y="665" width="92.5" height="43" fill="{color}"/>')
        parts.append(lettering(color,x+j*92.5+5,738,12,ink))
    parts.append(lettering(face,x,772,15,ink))
board='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1240 820" width="1240" height="820" role="img"><title>LOBBOT — Three identity directions</title>'+''.join(parts)+'</svg>'
ET.fromstring(board)
(root/'brand-directions.svg').write_text(board)
print('Outlined brand comparison board created.')
