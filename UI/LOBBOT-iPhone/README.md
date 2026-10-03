# LOBBOT / iPhone

Design rules: `../BRAND.md` and `../CLAUDE.md` (read them before editing).

Native SwiftUI app built around Dr. Lobbot, the 3D mascot. One screen: Lobbot on his
bordeaux stage, a speech bubble under him, and a panel that follows the consultation.

0. **Launch**: the Sillon mark draws itself on bordeaux, then the wordmark lands; it stays up while the
   mascot loads (1.9 s min, 4 s max), then Lobbot greets you.
   **Home**: tagline, *New surgery*, *Talk to Dr. Lobbot*, how it works, recent surgeries. Tap the
   logo to come back (not during surgery).
1. **Intake** (Lobbot listens and takes notes): model size and weights file, capability
   to preserve, target hardware and max capability loss, then the signed chart.
2. **Pre-op plan** (Lobbot thinks).
3. **Surgery**: Lobbot masks up and walks into the operating room (`task` held). The panel
   shows vitals (size, capability %, cuts) and a live log: every cut is tested, damaging
   cuts are undone.
4. **Result**: he walks out with the distilled model in a jar. Before/after, fit on the
   device, share report, new patient.

Open `LOBBOT.xcodeproj` in Xcode 26, pick an iPhone simulator, run. iOS 17+, portrait, light.

## Voice (Gemini Live)
Put the key in `Secrets.xcconfig` (next to the .xcodeproj): `GEMINI_API_KEY = AIza...` (no quotes).
That file is gitignored; `Secrets.example.xcconfig` is the committed template. At build time the key
flows xcconfig → `LOBBOT/Info.plist` → `VoiceConfig.apiKey`; it never appears in Swift source.
Bottom bar: type a message (he answers out loud, words in the bubble, no mic needed) or tap the
mic to call him: the stage fills the screen (close-up framing), with Mute, Type and End buttons and
captions of what you said. He chats about anything; when you talk about a model he fills the chart
through `update_chart` and starts the surgery with `start_surgery` (the call then folds back so the
surgery log is visible, and he keeps listening).
Model id, voice and persona: `VoiceConfig` in `Voice.swift`.
Any key shipped inside an app binary can be extracted: fine for the hackathon; for a release,
keep the key on a server that hands the app short-lived Gemini tokens.

## Files
- `Session.swift`: the consultation state and the surgery simulation (`operate()`).
  Replace the scripted cuts there with the agent's real event stream.
- `LobbotController.swift`: one persistent `WKWebView` running `Resources/Lobbot/lobbot.html`
  and the JS bridge (`play`, `finish`, `stop`, events `ready/started/ended/tap`).
- `ContentView.swift`, `Home.swift` (splash + home), `Panels.swift`, `MascotStage.swift`, `VoiceViews.swift`, `Brand.swift`: UI.
- `Resources/Lobbot/`: generated copies of `../mascot/embed/`. Edit `../mascot/index.html`, then run `python3 ../mascot/make_embed.py` (it rebuilds and copies).

Adding a Swift file: run `python3 create-project.py` to regenerate the project.
Codex's earlier palette-comparison UI is archived in `../archive/codex-v1-ios/`.

**Simulation only:** no model weights are read or modified; all numbers are illustrative.
