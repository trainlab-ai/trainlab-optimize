"""Build the price catalog from LiteLLM's model price map.

Source: https://github.com/BerriAI/litellm (model_prices_and_context_window.json, MIT).
We copy price *data* only (no code dependency) and keep attribution in the output.

Usage:
    python scripts/build_catalog.py                 # download latest from GitHub
    python scripts/build_catalog.py path/to/file.json
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
OUT = Path(__file__).resolve().parent / "prices.json"
USER_CACHE = Path.home() / ".cache" / "trainlab" / "prices.json"

# Providers that matter for organizational spend. Others are dropped to keep the file small.
KEEP_PROVIDERS = {
    "openai", "text-completion-openai", "anthropic", "azure", "azure_ai", "bedrock",
    "bedrock_converse", "vertex_ai-language-models", "vertex_ai-anthropic_models",
    "vertex_ai-embedding-models", "gemini", "mistral", "groq", "together_ai",
    "fireworks_ai", "deepseek", "cohere", "cohere_chat", "xai", "openrouter",
}
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


def build(raw: dict) -> dict:
    models = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict) or name == "sample_spec":
            continue
        if spec.get("litellm_provider") not in KEEP_PROVIDERS:
            continue
        if spec.get("mode") not in KEEP_MODES:
            continue
        if spec.get("input_cost_per_token") is None:
            continue
        entry = {"provider": spec["litellm_provider"], "mode": spec["mode"]}
        for src, dst in FIELDS.items():
            val = spec.get(src)
            if val is not None:
                entry[dst] = val
        models[name] = entry
    return {
        "_meta": {
            "source": "BerriAI/litellm model_prices_and_context_window.json",
            "source_url": SOURCE_URL,
            "source_license": "MIT",
            "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "unit": "USD per token",
            "model_count": len(models),
        },
        "models": dict(sorted(models.items())),
    }


def update(dest: Path = USER_CACHE, timeout: int = 60) -> dict:
    """Download the latest price map and write the catalog to the user cache."""
    with urllib.request.urlopen(SOURCE_URL, timeout=timeout) as resp:
        raw = json.loads(resp.read())
    catalog = build(raw)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(catalog, indent=0, separators=(",", ":")))
    return catalog


def main() -> None:
    if len(sys.argv) > 1:
        raw = json.loads(Path(sys.argv[1]).read_text())
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as resp:
            raw = json.loads(resp.read())
    catalog = build(raw)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(catalog, indent=0, separators=(",", ":")))
    print(f"wrote {OUT} ({catalog['_meta']['model_count']} models)")


if __name__ == "__main__":
    main()
