"""Report rendering (HTML, JSON)."""
from __future__ import annotations

import json
from pathlib import Path

from ..analyzer.engine import Analysis
from .html import render_html


def write_reports(analysis: Analysis, out_dir: str | Path, formats=("html", "json")) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    if "json" in formats:
        p = out / "trainlab-report.json"
        p.write_text(json.dumps(analysis.to_dict(), indent=1))
        written.append(p)
    if "html" in formats:
        p = out / "trainlab-report.html"
        p.write_text(render_html(analysis), encoding="utf-8")
        written.append(p)
    return written
