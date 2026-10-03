# UI/ is owned by Nolann (design)

Everything under `UI/` is the product's design: the iOS app, Dr. Lobbot (the 3D mascot) and the brand.
**Read `BRAND.md` before touching anything here.** It is the source of truth; `archive/` is outdated.

## Rules for AI agents

1. **Don't change the design unless Nolann asks.** That covers colours, fonts, sizes, spacing, radii, layout,
   animations, icons, copy tone and the mascot. No "cleanup", no "modernising", no reformatting, no dark mode,
   no UI library.
2. **Wire the backend through these points only:**
   - `LOBBOT-iPhone/LOBBOT/Session.swift`
     - `apply(_:)`: feed it the job's progress events (`GET /jobs/{id}/events`, same shape: stage, status,
       pct, msg) and drop `replay()`, which replays the Oct 3 quality run until then.
     - `QualityRun`: the real results shown on screen. Replace with the job's results once wired; never invent numbers.
     - `voiceTool(_:args:)`: what the voice can do in the app.
     - Add data fields there if needed. Views keep reading them as they are.
   - `LOBBOT-iPhone/LOBBOT/Voice.swift`: `VoiceConfig` (Gemini model id, persona, tools).
   - The mascot's state goes through `LobbotController.show(_:)`. Don't call the web view's JavaScript directly.
3. **If a feature needs new UI,** build it from the existing components (`PrimaryButton`, `OptionTile`, `Field`,
   `.card()`, `SpeechBubble`) with `Brand.*` colours only. Never use hex literals in views. Keep it minimal, and
   say in the commit message that it needs Nolann's review.
4. **The mascot:** edit only `mascot/index.html`, then run `python3 mascot/make_embed.py`, which regenerates the
   embed and copies it into the app. Never hand-edit `LOBBOT-iPhone/LOBBOT/Resources/Lobbot/*`. Never change his look.
5. **Xcode project:** after adding or removing a Swift file, run `python3 LOBBOT-iPhone/create-project.py`.
   Never hand-edit `project.pbxproj`.
6. **Secrets:** the Gemini key lives only in `LOBBOT-iPhone/Secrets.xcconfig`, which is gitignored. Never
   commit it, never paste it in code, logs or docs.
7. **Honesty:** words and numbers follow Oliver's `motion-ad-brief.md` fact sheet. Keep the "Replay" labels while
   the surgery is a replay. Never invent numbers, never claim it runs on the phone.
8. **When you commit changes under `UI/`,** list the UI files you touched in the commit message.
