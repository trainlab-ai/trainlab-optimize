"""Generate 30 days of synthetic usage for examples/demo_app in the real API response formats.

Writes:
  examples/demo_usage/openai_usage_completions.json   (OpenAI /organization/usage/completions pages)
  examples/demo_usage/openai_usage_embeddings.json    (OpenAI /organization/usage/embeddings pages)
  examples/demo_usage/anthropic_usage.json            (Anthropic usage_report/messages pages)
  examples/demo_usage/billing_export.csv              (same data as one generic CSV)
"""
import csv
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "examples/demo_usage"
START = datetime(2026, 9, 1, tzinfo=timezone.utc)
DAYS = 30
random.seed(7)

# model: (requests/day, input tokens/req, output tokens/req, cached share, project)
OPENAI = {
    "gpt-4o-2024-08-06": (60_000, 1_500, 300, 0.0, "proj_invoices"),
    "gpt-4.1": (40_000, 120, 2, 0.0, "proj_support"),
    "gpt-4-0613": (2_000, 800, 250, 0.0, "proj_support"),
    "gpt-5-mini": (10_000, 200, 150, 0.0, "proj_support"),
}
EMBED = {"text-embedding-3-large": (50_000, 4_000, "proj_search")}
ANTHROPIC = {"claude-sonnet-4-5-20250929": (20_000, 2_500, 400, 0.0, "wrkspc_support")}


def jitter(v):
    return int(v * random.uniform(0.85, 1.15))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    comp_buckets, emb_buckets, ant_buckets, rows = [], [], [], []
    for d in range(DAYS):
        t0 = START + timedelta(days=d)
        t1 = t0 + timedelta(days=1)
        day = t0.strftime("%Y-%m-%d")
        results = []
        for model, (req, tin, tout, cached, proj) in OPENAI.items():
            n = jitter(req)
            r = {"object": "organization.usage.completions.result", "input_tokens": n * tin,
                 "output_tokens": n * tout, "num_model_requests": n,
                 "input_cached_tokens": int(n * tin * cached), "model": model,
                 "project_id": proj, "batch": False}
            results.append(r)
            rows.append([day, "openai", model, r["input_tokens"], r["output_tokens"],
                         r["input_cached_tokens"], n, "", proj])
        comp_buckets.append({"object": "bucket", "start_time": int(t0.timestamp()),
                             "end_time": int(t1.timestamp()), "results": results})
        eres = []
        for model, (req, tin, proj) in EMBED.items():
            n = jitter(req)
            eres.append({"object": "organization.usage.embeddings.result", "input_tokens": n * tin,
                         "num_model_requests": n, "model": model, "project_id": proj})
            rows.append([day, "openai", model, n * tin, 0, 0, n, "", proj])
        emb_buckets.append({"object": "bucket", "start_time": int(t0.timestamp()),
                            "end_time": int(t1.timestamp()), "results": eres})
        ares = []
        for model, (req, tin, tout, cached, ws) in ANTHROPIC.items():
            n = jitter(req)
            ares.append({"uncached_input_tokens": n * tin, "cache_read_input_tokens": 0,
                         "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
                         "output_tokens": n * tout, "model": model, "workspace_id": ws,
                         "service_tier": "standard"})
            rows.append([day, "anthropic", model, n * tin, n * tout, 0, n, "", ws])
        ant_buckets.append({"starting_at": t0.strftime("%Y-%m-%dT00:00:00Z"),
                            "ending_at": t1.strftime("%Y-%m-%dT00:00:00Z"), "results": ares})

    # Split into 31-bucket pages exactly as the APIs do (single page here).
    page = lambda data: {"object": "page", "data": data, "has_more": False, "next_page": None}  # noqa
    (OUT / "openai_usage_completions.json").write_text(json.dumps([page(comp_buckets)], indent=1))
    (OUT / "openai_usage_embeddings.json").write_text(json.dumps([page(emb_buckets)], indent=1))
    (OUT / "anthropic_usage.json").write_text(
        json.dumps([{"data": ant_buckets, "has_more": False, "next_page": None}], indent=1))
    with (OUT / "billing_export.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "provider", "model", "input_tokens", "output_tokens",
                    "cached_input_tokens", "requests", "cost_usd", "project"])
        w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
