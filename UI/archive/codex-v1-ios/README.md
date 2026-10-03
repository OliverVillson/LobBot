# LOBBOT / iPhone

Native SwiftUI prototype for comparing the approved Sillon logo in two palettes:

- **A · Anatomie**: bordeaux, blush, light surfaces.
- **B · Soufre**: acid yellow, graphite, dark surfaces.

Open `LOBBOT.xcodeproj` in Xcode 26 or newer, select an iPhone simulator, then run.
The default `LOBBOT` scheme remembers the in-app selection. Two shared schemes,
`LOBBOT-A-Anatomie` and `LOBBOT-B-Soufre`, launch the corresponding palette directly.

The same screens and data are used for both modes. The switch is available on
the configuration screen and on the experiment screen. Four named SwiftUI
previews cover both screens in both palettes.

Included interactions: native model-file picker (filename only), capability
selection, memory slider, benchmark-name editor, illustrative experiment,
cancellation, restart, and preserved/restored result states.

The experiment screen embeds the supplied 3D mascot in a persistent `WKWebView`,
on the requested bordeaux background `#96445A`, with a 4:3 frame and 22pt corners.
`LobbotController` queues commands until the page sends `ready`, then dispatches
them in order. The app keeps one controller and one WebGL context across navigation.
Planning plays `think`, compression holds `task`, success calls `finish`, regression
plays `bug`, and an import error plays `error`. Typing, recording, streaming,
discovery, and inactivity are also mapped for future integration; 60 seconds of
inactivity puts the agent to sleep after active work has finished. Canvas taps wake
the agent and provide a light haptic.

The **ellipsis menu beside “L’agent”** previews all nine scenes. Select “Chirurgie”,
wait until the surgery is underway, then “Terminer la chirurgie” to see the jar and
confetti. Listen, think, talk, and sleep loop until another state is selected.
The simulation runs for eight seconds; it is illustrative and can be stopped.

`LOBBOT/Resources/Lobbot/` is a folder reference containing unmodified copies of
`~/Coding/lobbot/embed/lobbot.html` and `three.min.js`. Both are bundled locally,
without rebuilding the character in Swift. Google Fonts in the supplied page are
optional; the offline system fallback is preserved. The page handles Reduce Motion.
The executable is named `LobbotApp` to avoid a case-insensitive filesystem collision
between the `Lobbot` resource folder and the `LOBBOT` target name.

**Prototype only:** no model weights are read or modified, and no compression,
benchmark, network request, or measured performance result is produced.

iPhone, portrait, iOS 17+. Dynamic Type, VoiceOver labels, native navigation
and sheets, reduced motion, and increased-contrast asset variants are included.
The requested palettes determine appearance: A uses light, B uses dark.

The logo path is transcribed from `logo-explorations/organic/m-sillon-symbol-ink.svg`.
Icons are SF Symbols; interface text uses Dynamic Type with San Francisco.
App-icon PNGs are rasterized locally from the approved vector geometry.

`create-project.py` regenerates the Xcode project and color assets deterministically.
No external package, CocoaPods, Swift package resolution, or account is required
for simulator development. Running on a physical iPhone requires a signing team
chosen in Xcode.
