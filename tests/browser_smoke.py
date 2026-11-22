"""Run against a started local app. Set TL_BROWSER to a Chromium executable if needed."""

import os
from pathlib import Path
import sys

from playwright.sync_api import expect, sync_playwright

root = Path(__file__).resolve().parents[1]
url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18731"
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        executable_path=os.getenv("TL_BROWSER"), headless=True
    )
    page = browser.new_page(viewport={"width": 1280, "height": 850})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(url)
    expect(page.locator(".row")).to_have_count(2)
    expect(page.locator("#status")).to_contain_text("ONLINE")
    page.locator(".block").last.click()
    page.wait_for_function(
        "document.querySelector('video').currentTime > 1 && !document.querySelector('video').seeking"
    )
    assert page.locator("video").evaluate("v => v.videoWidth > 0")
    video = page.locator("video")
    video.evaluate("v => v.play()")
    page.wait_for_function("document.querySelector('video').currentTime > 4.4")
    video.evaluate("v => v.pause()")
    (root / "work").mkdir(exist_ok=True)
    page.screenshot(path=root / "work/workspace-desktop.png", full_page=True)
    page.locator("#file").set_input_files(root / "web/example.mp4")
    expect(page.locator("#source")).to_have_text("UPLOAD")
    expect(page.locator(".row")).to_have_count(2)
    page.locator("#file").set_input_files(
        {"name": "invalid.mp4", "mimeType": "video/mp4", "buffer": b"invalid"}
    )
    expect(page.locator(".error")).to_be_visible()
    page.set_viewport_size({"width": 390, "height": 844})
    page.locator("#demo").click()
    expect(page.locator(".row")).to_have_count(2)
    page.locator(".row").first.click()
    page.wait_for_function("document.querySelector('video').currentTime > 0")
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.screenshot(path=root / "work/workspace-mobile.png", full_page=True)
    assert not errors, errors
    browser.close()
    print(
        "PASS: demo, preview decode, upload, error recovery, event/timeline seeking, mobile layout"
    )
