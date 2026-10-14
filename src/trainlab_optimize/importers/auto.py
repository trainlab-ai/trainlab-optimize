"""Detect what kind of usage file this is and read it with the right importer.

Used by `trainlab import <file>` and the browser UI, so non-developers never have to know
whether they hold a billing CSV, a usage-API dump, a Langfuse export or OpenTelemetry traces.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..schema import UsageRecord

KINDS = ("auto", "csv", "aws", "azure", "gcp", "langfuse", "otel", "openai-file", "anthropic-file",
         "openrouter-file", "trainlab")


def detect_kind(path: str | Path) -> str:
    p = Path(path)
    if p.is_dir():
        return "trainlab"
    suffix = p.suffix.lower()
    if suffix not in (".json", ".jsonl", ".ndjson"):
        import csv
        from .cloud_csv import detect
        try:
            with open(p, newline="", encoding="utf-8-sig") as fh:
                header = next(csv.reader(fh), [])
        except (UnicodeDecodeError, csv.Error):
            header = []
        return detect(header) or "csv"
    with open(p, "rb") as fh:
        head = fh.read(20000).decode("utf-8", "replace")
    from .runtime import is_events_file
    if is_events_file(head):
        return "trainlab"
    if "resourceSpans" in head or "gen_ai." in head:
        return "otel"
    if '"organization.usage' in head or '"num_model_requests"' in head:
        return "openai-file"
    if "uncached_input_tokens" in head or '"cache_read_input_tokens"' in head:
        return "anthropic-file"
    if '"model_permaslug"' in head or ('"provider_name"' in head and '"prompt_tokens"' in head):
        return "openrouter-file"
    return "langfuse"


def read_any(path: str | Path, kind: str = "auto", default_provider: str | None = None,
             column_map: dict[str, str] | None = None) -> tuple[list[UsageRecord], str]:
    """Return (records, detected kind). Raises ValueError with a readable message on failure."""
    from .cloud_csv import read_cloud_csv
    from .csv_generic import read_csv
    from .traces import read_langfuse, read_otel

    p = Path(path)
    if kind not in KINDS:
        raise ValueError(f"Unknown file type '{kind}'. Use one of: {', '.join(KINDS)}")
    if kind == "auto":
        kind = "csv" if column_map else detect_kind(p)
    if kind == "csv":
        records = read_csv(p, default_provider=default_provider, column_map=column_map)
    elif kind in ("aws", "azure", "gcp"):
        records = read_cloud_csv(p, kind)
    elif kind == "langfuse":
        records = read_langfuse(p)
    elif kind == "otel":
        records = read_otel(p)
    elif kind == "trainlab":
        from .runtime import read_events
        records = read_events(p)
    elif kind == "openrouter-file":
        from .openrouter import parse_activity
        try:
            data = json.loads(p.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p.name} is not valid JSON ({exc})") from exc
        records = parse_activity(data, source=f"openrouter_usage_file:{p.name}")
    else:
        from . import anthropic_usage, openai_usage
        try:
            pages = json.loads(p.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p.name} is not valid JSON ({exc})") from exc
        pages = pages if isinstance(pages, list) else [pages]
        if kind == "openai-file":
            emb = any("embeddings.result" in json.dumps(pg)[:20000] for pg in pages)
            records = openai_usage.parse_pages(pages, kind="embeddings" if emb else "completions")
        else:
            records = anthropic_usage.parse_pages(pages)
        label = "openai_usage_file" if kind == "openai-file" else "anthropic_usage_file"
        for r in records:
            r.source = f"{label}:{p.name}"
    return records, kind
