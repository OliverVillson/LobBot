> **Superseded.** This documents the early identity exploration (Identity Lab). The approved design is in `../BRAND.md`. Do not apply these colors or fonts.

---
name: LOBBOT Identity Lab
description: An exploratory industrial specimen catalogue of three capability-guided identities.
colors:
  accent: "#f04b32"
  ink: "#191b18"
  paper: "#f5f5f1"
  muted: "#c9cdc5"
  secondary: "#5a5f55"
  rule: "#d5d8d0"
  white: "#fff"
  core-accent: "#d6e95a"
  core-ink: "#293326"
  core-paper: "#f3f5ed"
  core-muted: "#bac4ae"
  core-secondary: "#56634a"
  core-rule: "#cfd5c4"
  signal-accent: "#ff894c"
  signal-ink: "#212321"
  signal-paper: "#eff1ef"
  signal-muted: "#bfc5bf"
  signal-secondary: "#555c55"
  signal-rule: "#cdd1cd"
  model-structure: "#60685c"
  code-comment: "#c4ccbd"
  code-value: "#e4e9dd"
typography:
  display:
    fontFamily: "Archivo, sans-serif"
    fontSize: "clamp(50px, 5.3vw, 80px)"
    fontWeight: 700
    lineHeight: 1.02
    letterSpacing: "-.035em"
  display-core:
    fontFamily: "Manrope, sans-serif"
    fontSize: "clamp(50px, 5.3vw, 80px)"
    fontWeight: 700
    lineHeight: 1.02
    letterSpacing: "-.035em"
  display-signal:
    fontFamily: "Barlow Semi Condensed, sans-serif"
    fontSize: "clamp(50px, 5.3vw, 80px)"
    fontWeight: 600
    lineHeight: 1.02
    letterSpacing: "-.035em"
  headline:
    fontFamily: "Archivo, sans-serif"
    fontSize: "clamp(34px, 3.25vw, 48px)"
    fontWeight: 700
    lineHeight: 1.02
    letterSpacing: "-.035em"
  title:
    fontFamily: "Archivo, sans-serif"
    fontSize: "18px"
    fontWeight: 650
    lineHeight: 1.55
    letterSpacing: "-.015em"
  body:
    fontFamily: "Archivo, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "Archivo, sans-serif"
    fontSize: "13px"
    fontWeight: 600
    lineHeight: 1.3
  measurement:
    fontFamily: "Martian Mono, monospace"
    fontSize: "11px"
    fontWeight: 500
    lineHeight: 1.5
  smallprint:
    fontFamily: "Archivo, sans-serif"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.7
rounded:
  square: "0"
spacing:
  mobile-gutter: "20px"
  compact-gutter: "26px"
  tablet-gutter: "40px"
  desktop-gutter: "64px"
  control-gap: "18px"
  panel-padding: "24px"
components:
  button-solid:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    typography: "{typography.label}"
    rounded: "{rounded.square}"
    padding: "15px 20px"
  button-solid-hover:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.ink}"
  button-accent:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.square}"
    padding: "15px 20px"
  button-accent-hover:
    backgroundColor: "{colors.paper}"
  button-outline:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.square}"
    padding: "15px 20px"
  button-outline-hover:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
  branch-pass:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.ink}"
    rounded: "{rounded.square}"
    padding: "6px 9px"
  branch-restore:
    backgroundColor: "{colors.muted}"
    textColor: "{colors.ink}"
    rounded: "{rounded.square}"
    padding: "6px 9px"
  experiment:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    rounded: "{rounded.square}"
    padding: "24px"
---

# Design System: LOBBOT Identity Lab

## Overview

**Creative North Star: "Industrial Specimen Catalogue"**

An industrial specimen catalogue makes reduction visible through flat material fields, structural gaps and fixed references. Cold paper, dark ink and bold local type support precise, experimental identities for LOBBOT’s future developer audience. The marks are editable vectors; the catalogue treats their geometry as the material, rather than decorating it with photographic or synthetic imagery.

This documents the shipped exploration, not a final approved identity. Incision is the default; Noyau (`core`) and Signal remain selectable alternatives. Their accent, ink, paper, secondary text, rules, display face, marks and explanatory language change together. Archivo body text and Martian Mono notation stay common. The illustrative experiment explains reversible reduction without supplying real performance evidence.

**Key Characteristics:**
- Flat paper and ink surfaces with broad symbol-bearing accent fields.
- Bold display type paired with local tabular technical notation.
- Coordinated theme selection across marks, color, display type and links.
- Fixed protected geometry with explicit remove, test, keep and restore states.

## Colors

Each selected identity uses a single intervention accent against paper and ink. The frontmatter’s unprefixed roles describe default Incision; `core-` and `signal-` are complete alternatives, not secondary accents to mix into it.

### Primary

- **Incision / vermilion:** default symbol field, action and protected signal.
- **Cœur vif / acid yellow:** Noyau’s selected accent and protected center.
- **Trace / electric orange:** Signal’s selected accent and protected trace.

### Neutral

- **Carbone, Olive dense, Graphite:** selected ink for reading, monochrome marks and dark sections.
- **Feuille:** each direction’s paper background and reversed dark-field text.
- **Acier, Membrane, Bruit:** selected muted structural surfaces and restore labels.
- **Secondary ink:** each direction’s supporting text.
- **Fine rule:** each direction’s dividers and container outlines.
- **White:** unchanged wordmark field, selected selector surface and research note.
- **Model structure, code comment, code value:** shared dark-field structural and code colors, independent of identity.

**The The Selected Specimen Rule.** Apply the chosen palette and display face as one system. Keep the other identities visible only where they are being compared.

**The The Reading Ink Rule.** Noyau and Signal landing emphasis stays in ink with an accent underline; their title punctuation also stays in ink.

## Typography

**Display Font:** Archivo for Incision; Manrope for Noyau; Barlow Semi Condensed for Signal, each with sans-serif fallback.

**Body Font:** Archivo, sans-serif, across all themes.

**Label/Mono Font:** Martian Mono, monospace, with tabular numbers on code.

All four families are loaded locally with `font-display: swap`. Heavy compact headings contrast with clear utilitarian prose; changing the display voice does not change the body font.

### Hierarchy

- **Display:** the catalogue uses the frontmatter display roles. The developer hero uses `clamp(54px, 5.5vw, 84px)`; at 800px it becomes 70px, then 57px at 550px. The mobile catalogue title is 53px with 1.04 leading.
- **Headline:** section titles use the headline ramp, selected display face and 700 weight, except Signal uses 600. Purpose-specific process and closing titles vary within the same compact rhythm.
- **Title:** general subheadings use the title role; method headings use selected display type at 22px, 20px at 800px and 25px at 550px.
- **Body:** the default is 16px with 1.55 leading and a 70ch maximum paragraph width. Actual explanatory prose ranges from 13–17px; the desktop introduction is 21px at 1.4.
- **Label:** controls use the label role, reaching 11px in compact contexts; navigation uses 14px.
- **Measurement:** specimen identifiers, experiment status and phase digits use mono at 11px. Phase digits accompany their heading on one baseline.
- **Smallprint:** simulation disclaimers, hero footnotes, introduction supporting text, naming caveats and code caveats resolve to 12px through the final legibility override.

**The The Measured Notation Rule.** Reserve mono for code, measurement, specimen identity and meaningful phase indexes. Keep phase digits inside the associated heading.

## Layout

The shared container is centered with a 1440px maximum width. Its horizontal padding steps through the frontmatter gutter values: desktop, 1100px, 800px and 550px. The header has its own 40px margins, then 30px, 24px and 20px; at 1600px it shares the centered container geometry.

The catalogue specimen pairs a 43% symbol field with a 57% wordmark field, with 410px minimum height. At 1100px the split is 42/58; at 550px it stacks. The introduction pairs unequal columns; system explanations use paired grids. The three-mark gallery becomes compact horizontal rows on phones. Ordinary section gaps range from 30–80px and vertical section space from roughly 43–100px; there is no universal spacing scale beyond the observed gutter and control rhythm.

The landing hero is a 1.05/.95 grid with a 56px gap; it stacks at 800px. The four method columns become separated rows at 550px. Main navigation hides at 800px while identity and cross-surface links remain. Developer copy and code stack at 800px; constraints stack at 550px. Text reflows without a horizontal code overflow.

## Elevation & Depth

Depth comes from paper, white, ink and accent fields, with fine one-pixel rules. Dark experiment and code panels use a paper-color border mixed to 25% opacity; dark supporting text uses paper at 75–90%. No box shadows or backdrop blur are present.

**The The Flat Instrument Rule.** Use tonal fields and fine rules for separation; the shipped system has no box shadows.

## Shapes

Controls and panels have square corners. Fine rectangular boundaries and segmented bars echo the vector marks. Incision uses an oblique gap in a B; Noyau uses opened squared layers around an independent center; Signal uses a bar matrix crossed by a continuous trace. These are separate exploratory silhouettes, not interchangeable generic network symbols. Interface icons use inline SVG with square caps, miter joins and a 1.6 stroke.

## Components

### Buttons

Flat square controls use 15px by 20px padding, a 50px minimum height, 18px icon gap and a transparent one-pixel border. Solid controls reverse paper on ink and switch to ink on accent on hover. Accent controls use ink on accent and paper on hover. The outline variant is defined in the shipped stylesheet, though not used in the two current pages; it reverses on hover. Transitions last 150ms. Focus uses a 3px outline offset by 5px; dark experiment buttons use accent. Disabled experiment controls use 0.6 opacity and a wait cursor. Compact experiment actions reach 43px minimum height.

### Chips

The method’s pass and restore chips are noninteractive rectangular annotations with 6px by 9px padding and 11px text. Pass uses selected accent; restore uses selected muted. They describe branches and do not simulate selectable filters.

### Cards / Containers

The signature specimen is a contiguous accent symbol field plus white wordmark field. There is no rounded card shell. Gallery specimens are flat accent tiles; hover scales the mark to 1.05 over 240ms. The experiment is a fine outlined dark panel with 24px padding; the landing uses 28px, reducing to 19px on phones. Code uses the same dark outline vocabulary, while research notes use white flat fields.

### Navigation

A slim ruled header pairs the Archivo LOBBOT identity with 14px links and a compact cross-surface action. Links underline on hover; supporting color changes on ordinary navigation. Keyboard focus remains visible. The three-way identity selector has divided square cells, a white pressed state, an ink identifier square and a 3px accent underline. `aria-pressed` communicates selection; the query parameter carries the direction between surfaces.

### Protected-capability experiment

Twenty-four fixed-position bars surround protected indices 9–14. Only peripheral bars change; the rejected group is 8 and 15. Removal uses a -22px translation and 0.14 opacity, accepted removal a 0.045 vertical scale and 0.22 opacity; rejected bars briefly move -14px at 0.5 opacity before restoration. Movement and opacity take 320ms; gallery easing and bar transforms share `cubic-bezier(.16,1,.3,1)`. Stages pause for 820ms for reading, or 120ms with reduced motion. Reduced motion removes transitions and the transient translations. Status is announced politely; reset cancels the pending sequence. Every instance carries an explicit illustrative disclaimer.

## Do's and Don'ts

### Do:

- **Do** change palette, display face and vector assets together when selecting an identity.
- **Do** keep the protected capability fixed during the illustrative experiment and restore rejected peripheral components to their original positions.
- **Do** preserve visible focus: ink on light fields and accent on dark experiment controls.
- **Do** keep functional labels at least 11px and the shipped body smallprint at least 12px.
- **Do** use the included local fonts and editable SVG geometry.

### Don't:

- **Don’t** present any of the three exploratory marks as the approved final LOBBOT identity.
- **Don’t** add generic purple/blue AI gradients, glowing brains, chatbot imagery or artificial logo volume.
- **Don’t** use the animation as evidence of an actual compression result or benchmark.
- **Don’t** turn the specimen’s practical identifiers into decorative kickers.
