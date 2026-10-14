"""Report rendering (HTML, JSON, Markdown, PDF)."""
from __future__ import annotations

import json
from pathlib import Path

warnings: list[str] = []   # non-fatal problems from the last write_reports call (e.g. no browser for PDF)

from ..analyzer.engine import Analysis
from .html import render_html
from .markdown import render_markdown

FORMATS = ("html", "json", "md", "pdf")


def write_reports(analysis: Analysis, out_dir: str | Path, formats=("html", "json")) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings.clear()
    unknown = set(formats) - set(FORMATS)
    if unknown:
        raise ValueError(f"unknown report format(s): {', '.join(sorted(unknown))}; use {', '.join(FORMATS)}")
    written = []
    if "json" in formats:
        p = out / "trainlab-report.json"
        p.write_text(json.dumps(analysis.to_dict(), indent=1))
        written.append(p)
    if "md" in formats:
        p = out / "trainlab-report.md"
        p.write_text(render_markdown(analysis), encoding="utf-8")
        written.append(p)
    if "html" in formats or "pdf" in formats:
        p = out / "trainlab-report.html"
        p.write_text(render_html(analysis), encoding="utf-8")
        if "html" in formats:
            written.append(p)
        if "pdf" in formats:
            from .pdf import PdfError, html_to_pdf
            try:
                written.append(html_to_pdf(p, out / "trainlab-report.pdf"))
            except PdfError as exc:
                warnings.append(str(exc))
    return written
