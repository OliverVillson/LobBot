# UI · lobbot

Design, mascot and iOS app for lobbot, by Nolann. **Agents and teammates: read `CLAUDE.md` and `BRAND.md` first.**

| Folder | What it is |
|---|---|
| `LOBBOT-iPhone/` | SwiftUI app (iOS 17+). Open `LOBBOT.xcodeproj` in Xcode 26 and run. See its README. |
| `mascot/` | Dr. Lobbot, the 3D mascot (Three.js). `index.html` is the source and a standalone demo; `make_embed.py` builds the app copy. |
| `brand-studio/`, `logo-explorations/` | Logo (Sillon), palettes and their generators. |
| `PRODUCT.md` | Product brief. |
| `archive/` | Superseded explorations. Don't use them as a reference. |

## Run the app
1. Copy `LOBBOT-iPhone/Secrets.example.xcconfig` to `LOBBOT-iPhone/Secrets.xcconfig` (the project script does it
   too) and put the Gemini key after `GEMINI_API_KEY =`. That file is gitignored.
2. Open `LOBBOT-iPhone/LOBBOT.xcodeproj` and run on an iPhone. The mic works best on a real device.

## Change the mascot
Edit `mascot/index.html` (open it in a browser to preview), then run `python3 mascot/make_embed.py`.
