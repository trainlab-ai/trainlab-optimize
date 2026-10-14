"""Generic CSV usage importer: any billing or usage export with recognisable columns.

Works for provider dashboard exports (OpenAI, Anthropic, OpenRouter, DeepSeek, Mistral,
DashScope/Model Studio, ...), gateway logs (LiteLLM, Portkey, Helicone), cloud cost exports
(AWS/Azure/GCP, routed to their own parser) or a hand-made sheet.

Required: a model column plus token columns or a cost column. Column names are matched
loosely (case, spaces, dashes and brackets ignored). When an export uses names we don't
know, pass a mapping: ``read_csv(path, column_map={"model": "Billing Item", "cost_usd": "Amount"})``
(CLI: ``--map model="Billing Item",cost_usd=Amount``).

Rows are aggregated per day x provider x model x project x API key, so per-request logs with
millions of rows stay small.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from ..scanner.providers import maker_of, provider_from_model_prefix
from ..schema import UsageRecord

ALIASES = {
    "date": ["date", "day", "usage_date", "billing_date", "start_time", "bucket", "period", "starting_at",
             "created_at", "timestamp", "time", "request_time", "datetime", "billing_cycle"],
    "provider": ["provider", "vendor", "service", "provider_name", "llm_provider", "custom_llm_provider"],
    "model": ["model", "model_name", "model_id", "deployment", "snapshot_id", "model_permaslug", "llm",
              "engine", "model_slug", "model_group"],
    "input_tokens": ["input_tokens", "prompt_tokens", "n_context_tokens_total", "tokens_in", "input",
                     "tokens_prompt", "native_tokens_prompt", "inputtokens", "prompttokens", "input_token"],
    "output_tokens": ["output_tokens", "completion_tokens", "n_generated_tokens_total", "tokens_out", "output",
                      "tokens_completion", "native_tokens_completion", "outputtokens", "completiontokens",
                      "output_token"],
    "cached_input_tokens": ["cached_input_tokens", "input_cached_tokens", "cache_read_input_tokens",
                            "cached_tokens", "tokens_cached", "cache_hit_tokens", "prompt_cache_hit_tokens"],
    "requests": ["requests", "num_requests", "num_model_requests", "n_requests", "calls", "request_count"],
    "cost_usd": ["cost_usd", "cost", "amount", "amount_usd", "spend", "usd", "total_cost", "cost_total",
                 "total_cost_usd", "response_cost", "charges", "price", "billed_amount"],
    "project": ["project", "project_id", "project_name", "workspace", "workspace_id", "workspace_name", "team",
                "team_id", "app", "app_name"],
    "api_key": ["api_key", "api_key_id", "api_key_name", "api_key_hash", "key", "key_name", "key_id",
                "apikey", "api_key_alias", "key_alias"],
    "batch": ["batch", "is_batch"],
}
_OPENROUTER_COLS = {"model_permaslug", "provider_name", "tokens_prompt", "generation_id", "cost_total"}


def _norm(h: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", h.strip().lower()).strip("_")


def _num(v: str | None) -> float:
    if v is None:
        return 0.0
    v = v.strip().replace(",", "").replace("$", "").replace("¥", "").replace("€", "")
    try:
        return float(v) if v else 0.0
    except ValueError:
        return 0.0


def _infer_provider(model: str) -> str:
    m = model.lower()
    if m.startswith("ft:"):
        return "openai"                     # OpenAI fine-tune: ft:gpt-4o-mini:org::id
    if "anthropic." in m or m.startswith(("meta.", "amazon.", "cohere.", "mistral.")):
        return "bedrock"
    return provider_from_model_prefix(model) or maker_of(model) or "unknown"


def _date(v: str | None) -> str:
    v = (v or "").strip()
    m = re.match(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", v)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", v)               # MM/DD/YYYY
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    if re.fullmatch(r"\d{10}(\.\d+)?", v):                         # unix seconds
        from datetime import datetime, timezone
        return datetime.fromtimestamp(float(v), tz=timezone.utc).strftime("%Y-%m-%d")
    return v[:10] or "unknown"


_ITEM_SUFFIX = re.compile(r"^\s*([\w.\-:/]+)[\s_-]+[(\[]?\s*(?:input|output|prompt|completion|cached|cache)"
                          r"(?:[\s_-]*tokens?)?\b.*$", re.I)


def _clean_model(v: str) -> str:
    """Bill line items name the model plus the meter: 'qwen-max input_tokens' -> 'qwen-max'."""
    m = _ITEM_SUFFIX.match(v)
    return m.group(1) if m else v


def parse_map(spec: str | None) -> dict[str, str]:
    """'model=Billing Item,cost_usd=Amount' -> {"model": "Billing Item", "cost_usd": "Amount"}"""
    out = {}
    for part in (spec or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    unknown = set(out) - set(ALIASES)
    if unknown:
        raise ValueError(f"unknown field(s) in --map: {', '.join(sorted(unknown))}; use {', '.join(ALIASES)}")
    return out


def read_csv(path: str | Path, default_provider: str | None = None,
             column_map: dict[str, str] | None = None) -> list[UsageRecord]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"{path.name}: empty CSV")
        if not column_map:
            from .cloud_csv import detect, read_cloud_csv
            kind = detect(list(reader.fieldnames))
            if kind:
                fh.close()
                return read_cloud_csv(path, kind)
        cols = {_norm(h): h for h in reader.fieldnames}
        mapping: dict[str, str] = {}
        for field, names in ALIASES.items():
            for n in names:
                if n in cols:
                    mapping[field] = cols[n]
                    break
        for field, col in (column_map or {}).items():
            if col not in reader.fieldnames:
                raise ValueError(f"{path.name}: column '{col}' not found. Columns: {', '.join(reader.fieldnames)}")
            mapping[field] = col
        if "model" not in mapping or not ({"input_tokens", "output_tokens", "cost_usd"} & mapping.keys()):
            missing = "a model column" if "model" not in mapping else "token or cost columns"
            raise ValueError(
                f"{path.name}: couldn't find {missing}. Columns in the file: {', '.join(reader.fieldnames)}. "
                "Tell trainlab which column is which, e.g. --map model=\"Model\",cost_usd=\"Amount\" "
                f"(fields: {', '.join(ALIASES)}).")
        via_openrouter = bool(_OPENROUTER_COLS & cols.keys())

        agg: dict[tuple, dict] = {}
        for row in reader:
            model = _clean_model((row.get(mapping["model"]) or "").strip())
            if not model:
                continue
            get = lambda f: row.get(mapping[f]) if f in mapping else None  # noqa: E731
            provider = (get("provider") if not via_openrouter else None) or default_provider or \
                ("openrouter" if via_openrouter else _infer_provider(model))
            provider = provider.strip().lower()
            batch_raw = (get("batch") or "").strip().lower()
            batch = batch_raw in ("true", "1", "yes") if batch_raw else None
            key = (_date(get("date")), provider, model, (get("project") or "").strip() or None,
                   (get("api_key") or "").strip() or None, batch)
            a = agg.setdefault(key, {"in": 0, "out": 0, "cached": 0, "req": 0, "cost": None, "rows": 0})
            a["in"] += int(_num(get("input_tokens")))
            a["out"] += int(_num(get("output_tokens")))
            a["cached"] += int(_num(get("cached_input_tokens")))
            req = get("requests")
            a["req"] += int(_num(req)) if req not in (None, "") else 1 if via_openrouter else 0
            cost = get("cost_usd")
            if cost not in (None, ""):
                a["cost"] = (a["cost"] or 0.0) + _num(cost)
            a["rows"] += 1
    return [UsageRecord(date=d, provider=p, model=m, input_tokens=a["in"], output_tokens=a["out"],
                        cached_input_tokens=a["cached"], requests=a["req"], cost_usd=a["cost"],
                        batch=b, project=proj, api_key=k, source=f"csv:{path.name}")
            for (d, p, m, proj, k, b), a in agg.items()]
