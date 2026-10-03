# Builds embed/lobbot.html (for the iOS app's WKWebView) from index.html: canvas only, transparent, local three.js,
# then copies the embed into the app bundle folder. Edit index.html only; run this after every change.
import pathlib
src = pathlib.Path(__file__).with_name('index.html').read_text()
css = """<style>
html, body { height: 100%; margin: 0; background: transparent !important; overflow: hidden; -webkit-user-select: none; -webkit-tap-highlight-color: transparent; }
.top, .note, .hint, .monitor { display: none !important; }
.app { max-width: none; padding: 0; height: 100%; gap: 0; }
.stage { height: 100% !important; border-radius: 0; background: transparent; }
</style>"""
src = src.replace('<script src="https://cdn.jsdelivr.net/npm/three@0.149.0/build/three.min.js"></script>',
                  '<script>window.LOBBOT_EMBED = true;</script>\n<script src="three.min.js"></script>')
assert 'three.min.js"></script>' in src and 'LOBBOT_EMBED = true' in src
head = '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover">\n'
out = head + src.replace('</style>', '</style>\n' + css, 1).replace('\n<div class="app">', '\n</head><body>\n<div class="app">', 1) + '\n</body></html>\n'
here = pathlib.Path(__file__).parent
(here / 'embed' / 'lobbot.html').write_text(out)
print('embed/lobbot.html written')
app = here.parent / 'LOBBOT-iPhone' / 'LOBBOT' / 'Resources' / 'Lobbot'
if app.is_dir():
    for name in ['lobbot.html', 'three.min.js']:
        (app / name).write_bytes((here / 'embed' / name).read_bytes())
    print('copied into', app)
