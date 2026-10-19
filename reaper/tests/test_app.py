import asyncio

from reaper_tui.app import Reaper


def test_setup_market_and_reap():
    async def run():
        app = Reaper()
        async with app.run_test(size=(150, 52)) as pilot:
            for cmd in ["/key nebius nb-1234567890abcd", "/compute evroc ev-1234567890abcd 2",
                        "/model gpt-oss 20b", "/prune 25"]:
                await pilot.click("#prompt")
                for ch in cmd:
                    await pilot.press(ch if ch != " " else "space")
                await pilot.press("enter")
                await pilot.pause(0.8)
            s = app.state
            assert s.inference.id == "nebius" and s.compute.id == "evroc" and s.gpu == "2x H100 80GB"
            assert s.model.id == "openai/gpt-oss-20b" and s.ratio == 0.25
            app.action_reap()
            while s.running or not s.stages:
                await pilot.pause(0.2)
            assert all(e["status"] == "done" for e in s.stages.values())

    asyncio.run(run())
