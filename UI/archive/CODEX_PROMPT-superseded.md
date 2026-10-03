Task: embed the Lobbot 3D mascot in our iOS app. Do NOT rebuild it natively: it is a self-contained WebGL page. Host it in a WKWebView and drive it from the agent's state. If the app turns out to be React Native, use react-native-webview with the same API (events arrive via onMessage as JSON).

## Files (read-only, do not edit)
- `~/Coding/lobbot/embed/lobbot.html`: canvas only, transparent background, no UI
- `~/Coding/lobbot/embed/three.min.js`: three.js r149, loaded by relative path

Copy both into the app as a folder reference (e.g. `Resources/Lobbot/`) so the relative path holds. If they ever need refreshing: `python3 ~/Coding/lobbot/make_embed.py`, then copy again.

## JS API (`window.lobbot`)
- `play(scene, opts?)`: scenes are `idle listen think eureka talk task bug error sleep`
  - `{loop: true}`: repeats until the next `play()`. Use it for listen/think/talk/sleep.
  - `{hold: true}` with `task`: he goes into surgery and stays there until `finish()`.
  - Without opts, a scene plays once, then he returns to idle on his own.
- `finish()`: releases a held task. He comes out with the distilled model in a jar, confetti falls, then he goes back to idle.
- `stop()`: back to idle.
- Events posted to `webkit.messageHandlers.lobbot`: `{event:"ready"}` (once), `{event:"started"|"ended", scene}`, `{event:"tap"}`.
- Never call `play()` before `ready`. Queue the commands and flush them on `ready`.

## Build
1. `LobbotView`: a `UIViewRepresentable` around `WKWebView`.
   - `config.userContentController.add(coordinator, name: "lobbot")`
   - `isOpaque = false`, `backgroundColor = .clear`, `scrollView.isScrollEnabled = false`, `scrollView.contentInsetAdjustmentBehavior = .never`
   - `loadFileURL(htmlURL, allowingReadAccessTo: folderURL)`
   - commands go through `evaluateJavaScript("lobbot.play('think',{loop:true})")`
2. `LobbotController` (ObservableObject, one shared instance). Keep the web view alive across screens rather than recreating it, because WebGL takes about 1 s to start. Expose `show(_ state:)`, which maps app state to the JS calls below.
3. Mapping (adapt it to our real agent states):
   - user typing / recording → `listen` loop
   - request sent / agent planning → `think` loop
   - streaming the answer → `talk` loop
   - compression / distillation job running → `task` hold; job succeeded → `finish()`
   - job failed / API error → `error`
   - eval regression / failing test → `bug`
   - result better than expected → `eureka`
   - no activity for more than 60 s → `sleep` loop; any input → `idle`
   - `tap` event → light haptic (`UIImpactFeedbackGenerator(style: .light)`)
4. Layout: put it on the agent screen inside a bordeaux container (`#96445A`, corner radius 22). The page is transparent and its lighting is tuned for that background. Use an aspect between square and 4:3, at least 280 pt wide. The camera adapts to the aspect on its own.

## Constraints
- Do not modify `lobbot.html`: all of its behaviour goes through the API above.
- Fonts (Manrope, Caveat) load from Google Fonts. Offline they fall back to system fonts, which is fine. Everything else is local.
- The page already respects Reduce Motion (`prefers-reduced-motion`).

## Done when
The app runs in the simulator, Lobbot shows up in the container, and every mapped state triggers its scene. If the real states aren't wired yet, add a temporary debug menu that fires all 9 scenes, plus task hold → `finish()`.
