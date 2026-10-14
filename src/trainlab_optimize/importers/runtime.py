"""Reads what `trainlab_optimize.instrument()` records (.trainlab/traces/events-*.jsonl).

Each line is one LLM call with exact tokens and the file:line that made it. Calls are
aggregated per day, provider, model, code location, service and tags; sampled calls are
scaled back up by their weight.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..schema import UsageRecord

SOURCE = "trainlab"


def is_events_file(head: str) -> bool:
    return '"sdk":' in head and '"v":' in head and '"provider":' in head


def event_files(path: str | Path) -> list[Path]:
    p = Path(path)
    if p.is_dir():
        files = sorted(p.rglob("events-*.jsonl")) or sorted(p.rglob("*.jsonl"))
        return [f for f in files if f.is_file()]
    return [p]


def read_events(path: str | Path) -> list[UsageRecord]:
    rows: dict[tuple, dict] = defaultdict(lambda: {"in": 0.0, "out": 0.0, "cached": 0.0, "cw": 0.0, "req": 0.0,
                                                    "est": 0})
    files = event_files(path)
    bad = 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    bad += 1                       # a line cut off by a crash; skip it
                    continue
                if not isinstance(e, dict) or not e.get("provider"):
                    continue
                if e.get("ok") is False and not (e.get("input_tokens") or e.get("output_tokens")):
                    continue                       # failed before anything was billed (4xx/5xx, client-side error)
                model = e.get("model") or "unknown"
                w = float(e.get("weight") or 1.0)
                tags = e.get("tags") or {}
                key = ((e.get("ts") or "")[:10] or "unknown", e["provider"], model, e.get("path"), e.get("line"),
                       e.get("service"), json.dumps(tags, sort_keys=True) if tags else "", bool(e.get("batch")))
                r = rows[key]
                r["in"] += (e.get("input_tokens") or 0) * w
                r["out"] += (e.get("output_tokens") or 0) * w
                r["cached"] += (e.get("cached_input_tokens") or 0) * w
                r["cw"] += (e.get("cache_write_tokens") or 0) * w
                r["req"] += w
                r["est"] += 1 if e.get("output_tokens_estimated") else 0
    if bad and not rows:
        raise ValueError(f"{Path(path).name}: no readable trainlab events ({bad} malformed lines)")
    name = Path(path).name or "traces"
    out = []
    for (day, provider, model, cpath, cline, service, tags, batch), r in rows.items():
        out.append(UsageRecord(
            date=day, provider=provider, model=model, input_tokens=round(r["in"]), output_tokens=round(r["out"]),
            cached_input_tokens=round(r["cached"]), cache_write_tokens=round(r["cw"]), requests=round(r["req"]),
            batch=batch or None, project=service, source=f"{SOURCE}:{name}", code_path=cpath, code_line=cline,
            tags=json.loads(tags) if tags else {}))
    return out
