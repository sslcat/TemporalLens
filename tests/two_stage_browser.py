"""Live Edge/Chromium verification with a private, validation-derived video."""

import os
import sys
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

root = Path(__file__).resolve().parents[1]
ui = root / "work/two-stage/ui"
base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18731"
with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path=os.environ.get("TL_BROWSER"), headless=True
    )
    for width, height in ((1280, 850), (390, 844)):
        page = browser.new_page(viewport={"width": width, "height": height})
        errors = []
        page.on("pageerror", lambda error, seen=errors: seen.append(str(error)))
        page.goto(base)
        expect(page.locator(".row")).to_have_count(2)
        page.select_option("#mode", "two-stage")
        expect(page.locator("#scope")).to_contain_text("false alarms")
        expect(page.locator("#model-name")).to_contain_text("Target detector")
        page.locator("#file").set_input_files(ui / "gesture.mp4")
        event = page.locator(".block").last
        expect(event).to_be_visible(timeout=60000)
        expect(page.locator("#candidates")).to_be_visible()
        page.locator("#candidates summary").click()
        candidate = page.locator("#candidate-list button").last
        candidate.click()
        video = page.locator("video")
        page.wait_for_function("!document.querySelector('video').seeking")
        event.click()
        page.wait_for_function(
            "document.querySelector('video').currentTime>0 && !document.querySelector('video').seeking"
        )
        start = video.evaluate("v=>v.currentTime")
        video.evaluate("v=>v.play()")
        page.wait_for_function(
            "t=>document.querySelector('video').currentTime>t+.2", arg=start
        )
        video.evaluate("v=>v.pause()")
        page.locator(".row").last.click()
        page.wait_for_function(
            "t=>Math.abs(document.querySelector('video').currentTime-t)<.05", arg=start
        )
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=ui / f"verified-{width}.png", full_page=True)
        page.locator("#file").set_input_files(
            {"name": "invalid.mp4", "mimeType": "video/mp4", "buffer": b"damaged"}
        )
        expect(page.locator(".error")).to_be_visible()
        page.locator("#file").set_input_files(root / "work/ipn/ui/still.mp4")
        expect(page.locator("#events")).to_contain_text("No gesture met")
        expect(page.locator(".block")).to_have_count(0)
        expect(page.locator("#candidates summary")).to_contain_text("(0)")
        page.select_option("#mode", "gesture")
        expect(page.locator("#model-name")).to_contain_text("DINOv2")
        expect(page.locator("#candidates")).to_be_hidden()
        page.select_option("#mode", "marker")
        expect(page.locator(".row")).to_have_count(2)
        assert not errors, errors
        print(
            "Verified two-stage upload, playback, candidate/event seeking, errors, empty state and mode switching at",
            width,
        )
        page.close()
    browser.close()
