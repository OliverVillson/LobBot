# lobbot · Brand chart

**Owner: Nolann Petri (design). This is the approved identity and the source of truth for everything in `UI/`.**
Anything that contradicts it (including `archive/`) is outdated. Changes to this chart go through Nolann.

lobbot is an autonomous compression agent: it shrinks a large model while testing, after every cut, the one
capability the user wants to keep. The brand tells that story as **awake brain surgery**, embodied by
**Dr. Lobbot**, a funny, nerdy, energetic and joyful surgeon for AI models.

---

## 1. Logo

| Asset | File |
|---|---|
| Symbol "Sillon" (brain contour with an open incision) | `brand-studio/dist/assets/palettes/sillon-anatomie-accent.svg`, `-ink.svg`, `-reverse.svg` |
| Master geometry (128 × 128, 12 px round strokes) | `logo-explorations/organic/m-sillon-symbol-ink.svg` |
| Wordmark | `lobbot`, always lowercase, heavy weight (SF Pro Heavy in the app, Manrope 800 on the web) |
| In code | `SillonShape` (filled) and `SillonLine` (centre line, for the draw-on animation) in `LOBBOT-iPhone/LOBBOT/Brand.swift` |

- Colours: bordeaux `#A13D58` on light surfaces, white on the bordeaux stage. Nothing else.
- Keep the geometry exactly: no redraw, no outline change, no rotation, no stretching, no gradient, no shadow.
- Clear space: at least half the symbol's width on every side. Minimum size: 20 pt.
- Symbol and wordmark sit on one baseline with a 10 pt gap (see the app header).

## 2. Colour

One palette, **Anatomie: bordeaux and rose**, light only. No dark mode, no other accent.

| Token (`Brand.*` in Swift) | Hex | Use |
|---|---|---|
| `canvas` | `#F3E7E6` | Page background |
| `surface` | `#FCF4F2` | Cards, speech bubble, text fields |
| `tile` | `#F3E7E6` at 70 % | Unselected tiles and boxes inside cards |
| `line` | `#D4B5BB` | 1 pt borders and dividers |
| `ink` | `#592C3B` | Text and icons |
| `secondary` | `#805764` | Supporting text, labels |
| `accent` | `#A13D58` | Primary buttons, selection, progress, logo |
| `rose` | `#F4B8C4` | Soft highlights (pills, captions); never for text on light backgrounds |
| `stageLight` → `stage` → `stageDeep` | `#AD5A70` → `#96445A` → `#7A3247` | Dr. Lobbot's stage: a radial gradient, always this one |
| `kept` | `#245C3D` | A cut that was kept, success |
| `restored` | `#853E32` | A cut that was undone, stop, errors |

- Views use `Brand.*` only. Never write hex values in views.
- White text only on `accent`, the stage or `restored`.
- The web mascot page (`mascot/index.html`) uses eyedropped equivalents of these values for its own chrome
  (`--page #efe5e4`, `--card #fcf3f2`, `--plum #4a2633`, `--wine #96445a`). They are approved: don't "fix" them.

## 3. Typography

- **App:** San Francisco (system font) with Dynamic Type styles: `.system(size: 26–46, weight: .heavy)` for the
  wordmark and hero numbers, `.title2.bold` for panel titles, `.headline` for buttons, `.subheadline` and
  `.footnote` for content, `.caption` for labels. Nothing below 11 pt (`.caption2` is the floor).
- **Web:** Manrope 500 to 800 only.
- **Caveat** is reserved for Dr. Lobbot's handwriting on his clipboard (inside the 3D scene). Nowhere else.
- Uppercase only for small section labels (`Field` in `Brand.swift`, 0.6 tracking).

## 4. Shape and layout

- Radii: stage 28 (0 in a full-screen call), cards 22, speech bubble 18, buttons, tiles and boxes 14, pills
  and the composer are capsules. Use the continuous corner style.
- Card = `surface` fill, 1 pt `line` border, 18 pt padding (`.card()` modifier). Don't nest cards in cards.
- Side gutter 16 pt, gaps 10 to 18 pt. Minimum touch target 44 pt; primary buttons are 54 pt high.
- **One screen, built around the mascot:** header (logo + "Simulation" pill), Dr. Lobbot's stage with his
  speech bubble under it, the panel for the current phase (home → intake → plan → surgery → result), and the
  composer docked at the bottom (text field + mic). The mic opens a full-screen call where the stage fills the page.
- Components to reuse: `PrimaryButton`, `OptionTile`, `Field`, `.card()` (`Brand.swift`), `SpeechBubble`,
  `MascotStage` (`MascotStage.swift`).

## 5. Dr. Lobbot (the mascot)

- **Look:** a pink brain (`#FF8A9C`, soft clearcoat), round white nerd glasses, a doctor's head mirror on a white
  band, expressive eyes, brows and mouth drawn on his face. He always stands on the bordeaux stage.
- **Personality:** a funny doctor, nerdy, energetic, joyful. Surgery humour: patient, chart, scalpel, "cuts kept / undone".
- **Scenes** (`window.lobbot.play(id, {loop, hold})` from `LobbotController`):

| Scene | Meaning |
|---|---|
| `idle` | Breathing, blinking, follows the finger |
| `listen` | Takes notes on his wooden clipboard while you talk |
| `think` | Thought bubble with code |
| `talk` | Speaking (voice or bubble) |
| `task` (`hold`) | Masks up, grabs the scalpel, goes into the operating room; `finish()` brings him out with the jar and confetti |
| `eureka` | Light bulb, jump |
| `bug` | A beetle climbs his head, he shakes it off |
| `error` | Dizzy, spiral eyes |
| `sleep` | Naps, wakes with a start |

- He is **never** redesigned, recoloured, replaced by a generic robot or avatar, or hidden behind other UI.
- Source of truth: `mascot/index.html`. The app's copy (`LOBBOT-iPhone/LOBBOT/Resources/Lobbot/`) is generated
  by `python3 mascot/make_embed.py`. Never edit the generated copy.

## 6. Motion

- Short, damped, springy. No motion blur, no velocity skew, no scroll hijacking.
- Launch: the Sillon line draws itself on the stage (1.1 s), then the wordmark lands with a light spring;
  the splash stays until the mascot is ready (1.9 s minimum, 4 s maximum).
- Panels change with a short snappy animation. The bubble types its text at about 28 ms per character.
- Respect Reduce Motion everywhere (the splash and the bubble already do).

## 7. Voice and copy

- English, short and plain. Name things the way a user would ("New surgery", "Talk to Dr. Lobbot", "Stop").
- Medical metaphors are the brand: patient, chart, pre-op plan, surgery, cuts kept / undone, the patient goes home.
- **Honesty:** until the real agent is wired, every number is illustrative. Keep the "Simulation" pill and the
  "Simulated run: no model is modified." lines. Never quote real benchmark results or customer claims.

## 8. Don't

- Purple or blue AI gradients, neon, glowing robot brains, generic neural-network art, chatbot clichés.
- New fonts, new colours, dark mode, or swapping in a UI kit.
- Moving the mic and composer away from the bottom, or the mascot away from the top.
- "Cleaning up", "modernising" or reformatting the design without Nolann asking.
