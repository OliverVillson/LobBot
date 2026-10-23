"""Black-hole palette: the terminal's black, ember reds, disk orange, and a little
white for the hottest light. Everything that draws a color takes it from here.
Names match the architecture doc's PALETTE (void, ember, flare, redshift, photon)."""

from textual.theme import Theme

VOID = "#000000"       # the event horizon: background
ASH = "#4a0d02"        # coolest lit gas: quiet borders, empty bar cells
CRIMSON = "#9e1b07"    # outer disk: frames
REDSHIFT = "#d7261e"   # errors, failed gates
EMBER = "#ff6a00"      # primary: the disk's hot orange
FLARE = "#ffae42"      # highlights, done marks
PHOTON = "#f5efe6"     # the bit of white: titles, the photon ring
DIM = "#8a5a44"        # secondary text
TEXT = "#f0dccb"       # body text, warm off-white

# accretion-disk steps from coolest to hottest (index 0 is unlit = VOID)
HEAT = [ASH, CRIMSON, REDSHIFT, EMBER, FLARE, PHOTON]
BANNER_GRADIENT = [CRIMSON, REDSHIFT, REDSHIFT, EMBER, EMBER, FLARE, FLARE, PHOTON]
SHADOW_EDGE = ASH

THEME = Theme(
    name="blackhole",
    primary=EMBER,
    secondary=REDSHIFT,
    accent=FLARE,
    foreground=TEXT,
    background=VOID,
    surface=VOID,
    panel=VOID,
    boost=VOID,
    success=FLARE,
    warning=EMBER,
    error=REDSHIFT,
    dark=True,
)

CSS = f"""
Screen {{ background: {VOID}; color: {TEXT}; }}
#banner {{ padding: 0 1; height: auto; }}
#main {{ border: round {CRIMSON}; border-title-align: right; border-title-color: {FLARE};
        border-title-style: bold; border-subtitle-align: left; border-subtitle-color: {DIM};
        height: auto; padding: 0 1; }}
#left {{ width: 52; height: auto; }}
#art {{ height: auto; }}
#who {{ height: auto; margin-top: 1; }}
#right {{ width: 1fr; height: auto; padding-left: 2; }}
#right Static {{ height: auto; margin-bottom: 1; }}
#log {{ height: 1fr; min-height: 3; background: {VOID}; padding: 0 1; scrollbar-size: 0 0; }}
#status {{ height: 1; padding: 0 1; background: {EMBER}; color: {VOID}; text-style: bold; }}
#prompt {{ border: none; height: 1; padding: 0 1; background: {VOID}; color: {TEXT}; }}
#prompt:focus {{ border: none; background: {VOID}; }}
#promptrow {{ height: 3; border-top: solid {CRIMSON}; border-bottom: solid {CRIMSON}; }}
#caret {{ width: 3; padding-left: 1; text-style: bold; color: {EMBER}; }}

ModalScreen {{ background: {VOID} 80%; align: center middle; }}
.box {{ border: round {EMBER}; border-title-color: {FLARE}; border-title-style: bold;
        border-subtitle-color: {DIM}; background: {VOID}; padding: 0 1; }}
#models {{ width: 124; max-width: 98%; height: auto; max-height: 95%; }}
#any_row {{ height: 1; margin: 1 0; }}
#any_model {{ border: none; height: 1; background: {VOID}; padding: 0; }}
#any_model:focus {{ border: none; }}
#detected {{ height: auto; border: round {ASH}; padding: 0 1; margin-bottom: 1; }}
#market_label {{ color: {EMBER}; text-style: bold; }}
#models DataTable {{ max-height: 14; }}
#schemas {{ width: 142; max-width: 98%; height: 30; max-height: 95%; }}
#schema_body {{ height: 1fr; }}
#schema_table {{ width: 72; height: 1fr; }}
#schema_detail {{ width: 1fr; height: 1fr; padding: 0 1 0 2; border-left: solid {ASH}; }}
#describe_row {{ height: 3; border-top: solid {ASH}; }}
#describe {{ border: none; height: 1; background: {VOID}; }}
#describe:focus {{ border: none; }}
#target_row {{ height: auto; padding-top: 1; }}
#target_row Label {{ color: {DIM}; }}

#logs {{ border: round {CRIMSON}; border-title-color: {FLARE}; border-title-style: bold;
         border-subtitle-color: {DIM}; height: 1fr; padding: 0 1; }}
#logs RichLog {{ background: {VOID}; scrollbar-color: {CRIMSON}; scrollbar-background: {VOID}; }}

Button {{ background: {VOID}; color: {TEXT}; border: tall {CRIMSON}; }}
Button:hover, Button:focus {{ background: {EMBER}; color: {VOID}; text-style: bold; }}
Input {{ background: {VOID}; border: tall {ASH}; }}
Input:focus {{ border: tall {EMBER}; }}
DataTable {{ background: {VOID}; height: auto; scrollbar-color: {CRIMSON}; scrollbar-background: {VOID}; }}
DataTable > .datatable--header {{ background: {VOID}; color: {FLARE}; text-style: bold; }}
DataTable > .datatable--cursor {{ background: {EMBER}; color: {VOID}; text-style: bold; }}
DataTable > .datatable--hover {{ background: {ASH}; }}
"""
