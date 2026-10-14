"""Mock usage for one scanned repo, built from the models its own code calls.

Three CSV formats rotate across repos so each import path is exercised about 33 times:

- generic     one row per day x model x key, with token and cost columns
- openrouter  OpenRouter activity export: per-request rows, vendor/model slugs, api_key_name
- unfamiliar  a cloud-bill layout with column names trainlab doesn't know ("Billing Item",
              "Pretax Amount (USD)", ...); imported with an explicit column map

Every file also bills a decoy model the code never calls, across two API keys. Expected
totals are computed here from the generated rows, independently of trainlab's importers.
A separate tokens-only file checks pricing against the raw catalog numbers.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from trainlab_optimize.scanner.providers import SELF_HOSTED, maker_of

FORMATS = ("generic", "openrouter", "unfamiliar")
KEYS = (("key-prod", 0.7), ("key-batch", 0.3))
DECOYS = ("gpt-4.1-nano", "claude-3-5-haiku", "gemini-2.0-flash-lite", "mistral-small-2402", "command-r7b-12-2024",
          "qwen-turbo", "deepseek-coder")
FALLBACK = {"openai": "gpt-4o-mini", "azure": "gpt-4o-mini", "anthropic": "claude-haiku-4-5",
            "gemini": "gemini-2.5-flash", "vertex": "gemini-2.5-flash", "dashscope": "qwen-plus",
            "deepseek": "deepseek-chat", "mistral": "mistral-small-latest", "cohere": "command-r-08-2024",
            "xai": "grok-4", "groq": "llama-3.3-70b-versatile", "moonshot": "kimi-k2.5", "zhipu": "glm-4.6",
            "bedrock": "anthropic.claude-haiku-4-5-20251001-v1:0", "openrouter": "openai/gpt-4o-mini"}
OR_VENDOR = {"openai": "openai", "anthropic": "anthropic", "gemini": "google", "dashscope": "qwen",
             "deepseek": "deepseek", "mistral": "mistralai", "xai": "x-ai", "moonshot": "moonshotai",
             "zhipu": "z-ai", "cohere": "cohere", "minimax": "minimax"}
DEFAULT_RATE = (1e-6, 3e-6)          # USD/token for models the catalog doesn't list
_SANE = re.compile(r"^[\w.\-:/@]{2,79}$")
CATALOG = json.loads((Path(__file__).resolve().parents[1] / "src/trainlab_optimize/catalog/prices.json")
                     .read_text())["models"]


def _rates(model: str) -> tuple[float, float]:
    e = CATALOG.get(model) or CATALOG.get(model.split("/", 1)[-1])
    return (e["in"], e.get("out", 0.0)) if e else DEFAULT_RATE


def code_models(sites) -> list[str]:
    c = Counter(s.model for s in sites
                if s.model and _SANE.match(s.model) and s.provider not in SELF_HOSTED
                and s.kind in ("chat", "embedding", "batch", "model_config"))
    return [m for m, _ in c.most_common(5)]


def generate(repo: str, sites, out_dir: Path, fmt: str) -> dict:
    rng = random.Random(int(hashlib.sha1(repo.encode()).hexdigest()[:8], 16))
    in_code = code_models(sites)
    models = list(in_code)
    if not models:                          # nothing resolvable: bill the provider's usual model
        prov = Counter(s.provider for s in sites).most_common(1)
        models = [FALLBACK.get(prov[0][0] if prov else "openai", "gpt-4o-mini")]
    from trainlab_optimize.analyzer.spend import base_model
    mentioned = {base_model(s.model) for s in sites if s.model}
    decoy = next(d for d in DECOYS if base_model(d) not in mentioned and d not in models)
    billed = models + [decoy]

    rows = []        # (date, model, key, in_tok, out_tok, requests, cost)
    for m in billed:
        req_day = rng.randint(50, 5000)
        tin, tout = rng.randint(200, 3000), (0 if "embed" in m.lower() else rng.randint(50, 800))
        rin, rout = _rates(m)
        for day in range(1, 31):
            for key, share in KEYS:
                req = max(1, int(req_day * share * rng.uniform(0.8, 1.2)))
                i, o = req * tin, req * tout
                rows.append((f"2026-09-{day:02d}", m, key, i, o, req, round(i * rin + o * rout, 6)))

    path = out_dir / f"usage_{fmt}.csv"
    column_map = None
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if fmt == "generic":
            w.writerow(["date", "provider", "model", "input_tokens", "output_tokens", "requests", "cost_usd", "api_key"])
            for d, m, k, i, o, r, c in rows:
                w.writerow([d, maker_of(m) or "openai", m, i, o, r, c, k])
        elif fmt == "openrouter":
            w.writerow(["generation_id", "created_at", "cost_total", "byok_usage_inference", "tokens_prompt",
                        "tokens_completion", "tokens_cached", "model_permaslug", "provider_name", "api_key_name"])
            n = 0
            for d, m, k, i, o, r, c in rows:
                slug = m if "/" in m else f"{OR_VENDOR.get(maker_of(m) or '', 'openai')}/{m}"
                half_i, half_o, half_c = i // 2, o // 2, round(c / 2, 6)
                for part in range(2):           # two generations per aggregate row
                    n += 1
                    w.writerow([f"gen-{n}", f"{d} 1{part}:00:00",
                                half_c if part == 0 else round(c - half_c, 6), 0,
                                half_i if part == 0 else i - half_i, half_o if part == 0 else o - half_o, 0,
                                slug, "Provider", k])
        else:
            w.writerow(["Billing Date", "Product", "Billing Item", "Key ID", "Usage", "Pretax Amount (USD)"])
            for d, m, k, i, o, r, c in rows:
                rin, rout = _rates(m)
                cin = round(i * rin, 6)
                w.writerow([d, "Model Studio", f"{m} input_tokens", k, i, cin])
                w.writerow([d, "Model Studio", f"{m} output_tokens", k, o, round(c - cin, 6)])
            column_map = {"model": "Billing Item", "cost_usd": "Pretax Amount (USD)", "date": "Billing Date",
                          "api_key": "Key ID"}

    by_key: dict[str, float] = {}
    by_model: dict[str, float] = {}
    for d, m, k, i, o, r, c in rows:
        by_key[k] = by_key.get(k, 0.0) + c
        by_model[m] = by_model.get(m, 0.0) + c

    # Tokens-only file: trainlab must price it from the catalog; expected from the raw catalog numbers
    tok_model = next((m for m in in_code if m in CATALOG and "/" not in m), "gpt-4o-mini")
    tok_path = out_dir / "usage_tokens_only.csv"
    rin, rout = CATALOG[tok_model]["in"], CATALOG[tok_model].get("out", 0.0)
    tok_expected = 0.0
    with tok_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "provider", "model", "input_tokens", "output_tokens"])
        for day in range(1, 31):
            i, o = 1_000_000 + day * 1000, (0 if CATALOG[tok_model]["mode"] == "embedding" else 200_000)
            w.writerow([f"2026-09-{day:02d}", CATALOG[tok_model]["provider"], tok_model, i, o])
            tok_expected += i * rin + o * rout

    return {"format": fmt, "path": str(path), "column_map": column_map, "rows_written": len(rows),
            "total_cost": round(sum(r[6] for r in rows), 4), "by_key": {k: round(v, 4) for k, v in by_key.items()},
            "by_model": {m: round(v, 4) for m, v in by_model.items()}, "code_models": in_code,
            "fallback_models": [m for m in models if m not in in_code], "decoy": decoy,
            "tokens_only": {"path": str(tok_path), "model": tok_model, "expected": round(tok_expected, 4)}}
