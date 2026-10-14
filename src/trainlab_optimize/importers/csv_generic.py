"""Generic CSV usage importer: any billing export with recognisable column names.

Works for provider dashboard exports, AWS/Azure cost exports reshaped to one row per
model per day, or a hand-made sheet. Required: model + (tokens or cost).
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..schema import UsageRecord

ALIASES = {
    "date": ["date", "day", "usage_date", "start_time", "bucket", "period", "starting_at"],
    "provider": ["provider", "vendor", "service"],
    "model": ["model", "model_name", "model_id", "deployment", "snapshot_id"],
    "input_tokens": ["input_tokens", "prompt_tokens", "n_context_tokens_total", "tokens_in", "input"],
    "output_tokens": ["output_tokens", "completion_tokens", "n_generated_tokens_total", "tokens_out", "output"],
    "cached_input_tokens": ["cached_input_tokens", "input_cached_tokens", "cache_read_input_tokens", "cached_tokens"],
    "requests": ["requests", "num_requests", "num_model_requests", "n_requests", "calls"],
    "cost_usd": ["cost_usd", "cost", "amount", "amount_usd", "spend", "usd"],
    "project": ["project", "project_id", "workspace", "workspace_id", "team", "api_key_id", "api_key_name"],
    "batch": ["batch", "is_batch"],
}


def _num(v: str | None) -> float:
    if v is None:
        return 0.0
    v = v.strip().replace(",", "").replace("$", "")
    try:
        return float(v) if v else 0.0
    except ValueError:
        return 0.0


def _infer_provider(model: str) -> str:
    m = model.lower()
    if m.startswith("claude"):
        return "anthropic"
    if "anthropic." in m or m.startswith(("meta.", "amazon.", "cohere.", "mistral.")):
        return "bedrock"
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith("azure/"):
        return "azure"
    return "openai"


def read_csv(path: str | Path, default_provider: str | None = None) -> list[UsageRecord]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"{path}: empty CSV")
        from .cloud_csv import detect, read_cloud_csv
        kind = detect(list(reader.fieldnames))
        if kind:
            fh.close()
            return read_cloud_csv(path, kind)
        cols = {h.lower().strip(): h for h in reader.fieldnames}
        mapping = {}
        for field, names in ALIASES.items():
            for n in names:
                if n in cols:
                    mapping[field] = cols[n]
                    break
        if "model" not in mapping:
            raise ValueError(f"{path}: no model column (looked for {ALIASES['model']})")
        if not ({"input_tokens", "output_tokens", "cost_usd"} & mapping.keys()):
            raise ValueError(f"{path}: need token columns or a cost column")

        out: list[UsageRecord] = []
        for row in reader:
            model = (row.get(mapping["model"]) or "").strip()
            if not model:
                continue
            get = lambda f: row.get(mapping[f]) if f in mapping else None  # noqa: E731
            provider = (get("provider") or default_provider or _infer_provider(model)).strip().lower()
            cost = get("cost_usd")
            batch_raw = (get("batch") or "").strip().lower()
            out.append(UsageRecord(
                date=(get("date") or "").strip()[:10] or "unknown",
                provider=provider, model=model,
                input_tokens=int(_num(get("input_tokens"))),
                output_tokens=int(_num(get("output_tokens"))),
                cached_input_tokens=int(_num(get("cached_input_tokens"))),
                requests=int(_num(get("requests"))),
                cost_usd=_num(cost) if cost not in (None, "") else None,
                batch=batch_raw in ("true", "1", "yes") if batch_raw else None,
                project=(get("project") or None),
                source=f"csv:{path.name}",
            ))
        return out
