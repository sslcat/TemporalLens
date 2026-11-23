"""Exercise both real-video layouts using private local QA fixtures."""

from pathlib import Path
import os
import sys

from playwright.sync_api import expect, sync_playwright

root = Path(__file__).resolve().parents[1]
ui = root / "work/ipn/ui"
base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18731"
with sync_playwright() as pw:
    browser = pw.chromium.launch(
        executable_path=os.environ.get("TL_BROWSER"), headless=True
    )
    for width, height in ((1280, 850), (390, 844)):
        page = browser.new_page(viewport={"width": width, "height": height})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(base)
        expect(page.locator(".row")).to_have_count(2)
        page.select_option("#mode", "gesture")
        expect(page.locator("#model-name")).to_contain_text("DINOv2")
        page.locator("#file").set_input_files(ui / "gesture.mp4")
        expect(page.locator(".row").first).to_be_visible(timeout=60000)
        target = page.locator(".block").last
        target.click()
        page.wait_for_function(
            "document.querySelector('video').currentTime>0 && !document.querySelector('video').seeking"
        )
        assert page.locator("video").evaluate("v=>v.videoWidth===320")
        video = page.locator("video")
        before = video.evaluate("v=>v.currentTime")
        video.evaluate("v=>v.play()")
        page.wait_for_function(
            "t=>document.querySelector(\u0027video\u0027).currentTime>t+0.2", arg=before
        )
        video.evaluate("v=>v.pause()")
        page.locator(".row").last.click()
        page.wait_for_function(
            "t=>Math.abs(document.querySelector('video').currentTime-t)<0.05",
            arg=before,
        )
        assert page.evaluate(
            "document.documentElement.scrollWidth <= window.innerWidth"
        )
        page.screenshot(path=ui / f"gesture-{width}.png", full_page=True)
        page.locator("#file").set_input_files(
            {"name": "broken.mp4", "mimeType": "video/mp4", "buffer": b"invalid"}
        )
        expect(page.locator(".error")).to_be_visible()
        page.locator("#file").set_input_files(ui / "still.mp4")
        expect(page.locator("#events")).to_contain_text("No gesture met")
        expect(page.locator(".block")).to_have_count(0)
        page.select_option("#mode", "marker")
        expect(page.locator(".row")).to_have_count(2)
        assert not errors, errors
        print(
            f"PASS {width}px: marker, gesture upload, playback preview, timeline, error, abstention and switch back"
        )
        page.close()
    browser.close()
