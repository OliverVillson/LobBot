---
version: 1
slug: "lobbot-iphone-lobbot-home-swift"
primary_target: "LOBBOT-iPhone/LOBBOT/Home.swift"
related_targets: ["LOBBOT-iPhone/LOBBOT/ContentView.swift"]
---

## Scope
Home page of the LobBot iOS app (`HomePanel` + the home layout in `ContentView`). Visitor mode: Persuade inside an app. Audience: hackathon jury and developers, equally: understand what LobBot does and see proof in seconds, then admit a patient right away.

## Direction contract
THESIS: The home is Dr. Lobbot's ward board, not a hero with explanations under it. Each job is a bed: the real patient is bed 1, this session's replays follow, an empty bed is the call to action. Refuses the big-mascot hero with stacked info cards.
OWN-WORLD: Anatomie palette (BRAND.md). A white whiteboard card ruled in `line`, a bordeaux "WARD 64" strip, numbered bed plates in tabular digits, status chips (Discharged in `kept`, Replay in rose, Empty as a dashed outline), SF Pro only.
STORY: In one look: one job in, a small model out that runs on your laptop offline; bed 1 proves it (61 GB → 6.52 GB, 9.7/10 vs 9.8); the legend explains the six stations; the empty bed admits your patient.
FIRST VIEWPORT: Lobbot's live tile top left (same tile as consultations), his bubble beside him. Under it the board: title strip, the one-line promise, bed 1 with its vitals, the empty bed "Admit a patient" in accent, the legend starting below the fold. Composer and mic stay docked at the bottom.
FORM: ward whiteboard / bed board, #3 on the ordered list; seed key 60cb09c1. Signature interaction: tapping a bed makes Lobbot comment on it in his bubble before opening it; rows arrive one by one like a nurse filling the board; the empty bed breathes.
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

## Constraints
Native SwiftUI, iOS 17. No local rendering on Nolann's Mac: he verifies on device and sends screenshots. Numbers only from `QualityRun` (Oliver's fact sheet). The stage view keeps its identity (AnyLayout).
