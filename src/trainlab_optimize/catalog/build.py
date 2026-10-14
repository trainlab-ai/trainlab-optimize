"""Build the price catalog.

Sources (price *data* only, no code dependency; attribution kept in the output):
- BerriAI/litellm model_prices_and_context_window.json (MIT): ~3,000 text models across
  every major provider and host (OpenAI, Anthropic, Google, AWS, Azure, Mistral, Cohere,
  DashScope/Qwen, DeepSeek, Moonshot, Zhipu, MiniMax, Volcengine, xAI, Groq, Together,
  Fireworks, DeepInfra, Perplexity, ...).
- OpenRouter's public model list (https://openrouter.ai/api/v1/models, no key needed):
  merged as ``openrouter/<vendor>/<model>`` when reachable.

Usage:
    python scripts/build_catalog.py                       # download both sources
    python scripts/build_catalog.py litellm.json [openrouter_models.json]
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SOURCE_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
OUT = Path(__file__).resolve().parent / "prices.json"
USER_CACHE = Path.home() / ".cache" / "trainlab" / "prices.json"

# Text-generation and embedding spend. Images, audio, video, rerank and search are priced per
# unit rather than per token and are out of scope.
KEEP_MODES = {"chat", "completion", "responses", "embedding"}

FIELDS = {
    "input_cost_per_token": "in",
    "output_cost_per_token": "out",
    "cache_read_input_token_cost": "cache_read",
    "cache_creation_input_token_cost": "cache_write",
    "input_cost_per_token_batches": "batch_in",
    "output_cost_per_token_batches": "batch_out",
    "max_input_tokens": "max_in",
    "deprecation_date": "deprecated",
}


def _float(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f >= 0 else None          # OpenRouter uses -1 for "variable" (auto routers)


def build_litellm(raw: dict) -> dict[str, dict]:
    models = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict) or name == "sample_spec":
            continue
        if spec.get("mode") not in KEEP_MODES or spec.get("input_cost_per_token") is None:
            continue
        entry = {"provider": spec.get("litellm_provider") or "unknown", "mode": spec["mode"]}
        for src, dst in FIELDS.items():
            val = spec.get(src)
            if val is not None:
                entry[dst] = val
        models[name] = entry
    return models


def build_openrouter(raw: dict) -> dict[str, dict]:
    """OpenRouter /api/v1/models -> catalog entries keyed openrouter/<vendor>/<model>."""
    models = {}
    for m in raw.get("data", []) if isinstance(raw, dict) else []:
        mid = m.get("id")
        pricing = m.get("pricing") or {}
        pin, pout = _float(pricing.get("prompt")), _float(pricing.get("completion"))
        if not mid or pin is None or pout is None:
            continue
        entry = {"provider": "openrouter", "mode": "chat", "in": pin, "out": pout}
        cr, cw = _float(pricing.get("input_cache_read")), _float(pricing.get("input_cache_write"))
        if cr is not None:
            entry["cache_read"] = cr
        if cw is not None:
            entry["cache_write"] = cw
        if m.get("context_length"):
            entry["max_in"] = m["context_length"]
        models[f"openrouter/{mid}"] = entry
    return models


def build(raw: dict, openrouter: dict | None = None) -> dict:
    models = build_litellm(raw)
    sources = ["BerriAI/litellm model_prices_and_context_window.json (MIT)"]
    if openrouter:
        extra = build_openrouter(openrouter)
        if extra:
            models.update(extra)
            sources.append(f"OpenRouter public model list ({len(extra)} models)")
    return {
        "_meta": {
            "source": "; ".join(sources),
            "source_url": SOURCE_URL,
            "source_license": "MIT",
            "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "unit": "USD per token",
            "model_count": len(models),
        },
        "models": dict(sorted(models.items())),
    }


def _get_json(url: str, timeout: int) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "trainlab-optimize"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def fetch_sources(timeout: int = 60) -> tuple[dict, dict | None]:
    raw = _get_json(SOURCE_URL, timeout)
    try:
        openrouter = _get_json(OPENROUTER_URL, timeout)
    except (OSError, ValueError):
        openrouter = None                 # optional: LiteLLM alone still covers every vendor
    return raw, openrouter


def update(dest: Path = USER_CACHE, timeout: int = 60) -> dict:
    """Download the latest prices and write the catalog to the user cache."""
    catalog = build(*fetch_sources(timeout))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(catalog, indent=0, separators=(",", ":")))
    return catalog


def main() -> None:
    if len(sys.argv) > 1:
        raw = json.loads(Path(sys.argv[1]).read_text())
        openrouter = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else None
    else:
        raw, openrouter = fetch_sources()
    catalog = build(raw, openrouter)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(catalog, indent=0, separators=(",", ":")))
    print(f"wrote {OUT} ({catalog['_meta']['model_count']} models)")


if __name__ == "__main__":
    main()
