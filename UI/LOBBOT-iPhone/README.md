# LOBBOT / iPhone

Design rules: `../BRAND.md` and `../CLAUDE.md` (read them before editing).

Native SwiftUI app built around Dr. Lobbot, the 3D mascot. One screen: Lobbot on his
bordeaux stage, a speech bubble under him, and a panel that follows the consultation.

0. **Launch**: the Sillon mark draws itself on bordeaux, then the wordmark lands; it stays up while the
   mascot loads (1.9 s min, 4 s max), then Lobbot greets you.
   **Home**: tagline, *New surgery*, *Talk to Dr. Lobbot*, how it works, recent surgeries. Tap the
   logo to come back (not during surgery).
1. **The job**: describe one job in a sentence (Lobbot listens and takes notes).
2. **The chart**: the task spec Gemini drafts (input, output, grading, an example).
3. **Going home to**: a 16 GB laptop, ≤ 7 GB, ≥ 40 tokens/s; starting from Qwen3-30B-A3B (61 GB).
4. **Surgery**: the six real stages (data, reap, heal, quantize, eval, package) on a live board; Lobbot is in the
   operating room. Until the backend is wired, it replays the real run of Oct 3, 2026 in about 20 s.
5. **Discharge**: 61 GB → 6.52 GB, 9.7/10 vs 9.8/10, the three candidates, a real email → JSON ticket, and the
   `lobbot save` command for the Mac.

Open `LOBBOT.xcodeproj` in Xcode 26, pick an iPhone simulator, run. iOS 17+, portrait, light.

## Voice (Gemini Live)
Put the key in `Secrets.xcconfig` (next to the .xcodeproj): `GEMINI_API_KEY = AIza...` (no quotes).
That file is gitignored; `Secrets.example.xcconfig` is the committed template. At build time the key
flows xcconfig → `LOBBOT/Info.plist` → `VoiceConfig.apiKey`; it never appears in Swift source.
Bottom bar: type a message (he answers out loud, words in the bubble, no mic needed) or tap the
mic to call him: the stage fills the screen (close-up framing), with Mute, Type and End buttons and
captions of what you said. He chats about anything; when you ask for a model he writes your job on the chart
through `set_job` and starts the surgery with `start_surgery` (the call then folds back so the
surgery log is visible, and he keeps listening).
Model id, voice and persona: `VoiceConfig` in `Voice.swift`.
Any key shipped inside an app binary can be extracted: fine for the hackathon; for a release,
keep the key on a server that hands the app short-lived Gemini tokens.

## Files
- `Session.swift`: the consultation state, the stage board (`apply(_:)` takes the backend's events) and
  `QualityRun`, the real results. `replay()` stands in for the live stream until it is wired.
- `LobbotController.swift`: one persistent `WKWebView` running `Resources/Lobbot/lobbot.html`
  and the JS bridge (`play`, `finish`, `stop`, events `ready/started/ended/tap`).
- `ContentView.swift`, `Home.swift` (splash + home), `Panels.swift`, `MascotStage.swift`, `VoiceViews.swift`, `Brand.swift`: UI.
- `Resources/Lobbot/`: generated copies of `../mascot/embed/`. Edit `../mascot/index.html`, then run `python3 ../mascot/make_embed.py` (it rebuilds and copies).

Adding a Swift file: run `python3 create-project.py` to regenerate the project.
Codex's earlier palette-comparison UI is archived in `../archive/codex-v1-ios/`.

**Replay:** the app doesn't run a job yet. It replays the real Oct 3, 2026 run; every number comes from that run.
