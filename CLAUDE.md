# LobBot

## UI/ (design, owned by Nolann)

`UI/` holds the iOS app, Dr. Lobbot (the 3D mascot) and the brand.
**Before editing anything under `UI/`, read `UI/CLAUDE.md` and `UI/BRAND.md`.**

- Don't change the app's look, the mascot or the brand. Wire backend features only through the
  points `UI/CLAUDE.md` lists: `Session.operate()`, `voiceTool(_:args:)` and `VoiceConfig`.
- Never commit `UI/LOBBOT-iPhone/Secrets.xcconfig`, which holds the Gemini key (it is gitignored).
