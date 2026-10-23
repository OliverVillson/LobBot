"""Drive the demo through each screen and save SVG screenshots to shots/."""
import asyncio, os, shutil, sys
from pathlib import Path

HOME = Path.home() / ".blackhole"
shutil.rmtree(HOME, ignore_errors=True)
os.environ["BLACKHOLE_HOME"] = str(HOME)
os.environ["BLACKHOLE_DEMO_SPEED"] = "1"
from blackhole_tui.app import BlackHole, LogsScreen

OUT = Path(__file__).parent / "shots"
SIZE = (150, 52)


async def typed(pilot, text):
    await pilot.press(*[c if c != " " else "space" for c in text])


async def main():
    app = BlackHole(demo=True)
    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause(0.3)
        app.save_screenshot("01-welcome.svg", str(OUT))
        await pilot.press("f3"); await pilot.pause(0.3)
        await typed(pilot, "mistralai/Mixtral-8x7B-Instruct-v0.1"); await pilot.press("enter")
        await pilot.pause(0.6)
        app.save_screenshot("02-detect.svg", str(OUT))
        await pilot.press("escape"); await pilot.pause(0.2)
        await pilot.press("f3"); await pilot.pause(0.3)
        await pilot.press("down", "down", "down"); await pilot.pause(0.2)
        app.save_screenshot("02-models.svg", str(OUT))
        await pilot.press("enter"); await pilot.pause(0.2)
        await pilot.press("f4"); await pilot.pause(0.3)
        await pilot.press("down", "down"); await pilot.pause(0.2)
        app.save_screenshot("03-schemas.svg", str(OUT))
        await pilot.press("enter"); await pilot.pause(0.2)
        await pilot.press("f4"); await pilot.pause(0.3)
        await pilot.press("tab"); await pilot.pause(0.1)
        await typed(pilot, "a tiny rust tutor from my docs that runs on my 16 GB laptop")
        await pilot.press("enter"); await pilot.pause(0.3)
        app.save_screenshot("04-describe.svg", str(OUT))
        await pilot.press("escape"); await pilot.pause(0.2)
        await typed(pilot, "/target mac32"); await pilot.press("enter")
        await typed(pilot, "/purpose \"Debugging Python and Rust\""); await pilot.press("enter")
        await typed(pilot, "/schema pocket-coder"); await pilot.press("enter")
        await pilot.press("f5")
        await pilot.pause(7)
        app.save_screenshot("05-running.svg", str(OUT))
        await pilot.press("f6"); await pilot.pause(0.6)
        app.save_screenshot("06-logs.svg", str(OUT))
        await pilot.press("escape")
    # a fresh TUI, as after an SSH drop, picks the job back up
    app2 = BlackHole(demo=True)
    async with app2.run_test(size=SIZE) as pilot:
        await pilot.pause(1.0)
        app2.save_screenshot("07-reattach.svg", str(OUT))
        while app2.state.running:
            await pilot.pause(0.5)
        await pilot.pause(0.5)
        app2.save_screenshot("08-done.svg", str(OUT))
    app3 = BlackHole(demo=True)
    async with app3.run_test(size=(160, 68)) as pilot:
        await pilot.pause(0.6)
        app3.save_screenshot("09-large.svg", str(OUT))

asyncio.run(main())
print("ok")
