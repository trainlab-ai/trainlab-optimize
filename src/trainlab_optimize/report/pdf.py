"""PDF export through a locally installed Chromium-based browser (no extra Python dependency)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

CANDIDATES = [
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "msedge",
    "microsoft-edge", "brave-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def find_browser() -> str | None:
    env = os.environ.get("TRAINLAB_BROWSER")
    if env:
        return env
    for c in CANDIDATES:
        path = shutil.which(c) or (c if os.path.isabs(c) and os.path.exists(c) else None)
        if path:
            return path
    # Playwright-installed Chromium, if present
    bases = [Path.home() / ".cache/ms-playwright", Path.home() / "Library/Caches/ms-playwright"]
    if os.environ.get("PLAYWRIGHT_BROWSERS_PATH"):
        bases.insert(0, Path(os.environ["PLAYWRIGHT_BROWSERS_PATH"]))
    for base in bases:
        for exe in sorted(base.glob("chromium-*/chrome-linux*/chrome")) + sorted(
                base.glob("chromium-*/chrome-mac*/Chromium.app/Contents/MacOS/Chromium")):
            return str(exe)
    return None


class PdfError(RuntimeError):
    pass


def html_to_pdf(html_path: Path, pdf_path: Path, timeout: int = 120) -> Path:
    browser = find_browser()
    if not browser:
        raise PdfError("No Chrome/Edge/Chromium found for PDF export. Open the HTML report and use "
                       "Print > Save as PDF, or set TRAINLAB_BROWSER to a Chromium-based browser.")
    args = [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()]
    if sys.platform.startswith("linux") and os.geteuid() == 0:
        args.insert(1, "--no-sandbox")
    try:
        subprocess.run(args, check=True, capture_output=True, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        raise PdfError(f"PDF export failed with {browser}: {exc}") from exc
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise PdfError(f"{browser} did not produce a PDF")
    return pdf_path
