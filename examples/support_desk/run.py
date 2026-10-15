"""Northwind support desk: a realistic multi-feature OpenAI app to try trainlab on.

    python run.py --mock                    # offline, free: a local mock of the OpenAI API
    python run.py                           # real OpenAI API (needs OPENAI_API_KEY), about $0.30 for 100 tickets

    trainlab trace -- python run.py         # same run, recorded per line, then the report

What it does with each ticket: routes it to a queue (gpt-4o), extracts CRM fields as JSON
(gpt-4o), scores mood with the async client (gpt-4o-mini), answers shipping questions with a
tool-calling agent (gpt-4o-mini) and everything else with a streamed reply grounded in
help-centre articles found by embedding search (text-embedding-3-small, gpt-4o-mini). Finally it
writes a daily digest with the Responses API (gpt-4.1). Threads, asyncio, streaming, JSON mode,
tools and two API styles: the shapes real apps have.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Rough real-API cost per ticket across all steps, from OpenAI list prices (Oct 2026).
COST_PER_TICKET_USD = 0.003


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Northwind support desk on a batch of tickets.")
    ap.add_argument("--tickets", type=int, default=100, help="how many tickets to process (default 100)")
    ap.add_argument("--mock", action="store_true", help="use a local mock of the OpenAI API (offline, free)")
    ap.add_argument("--workers", type=int, default=8, help="threads for triage and extraction (default 8)")
    ap.add_argument("--seed", type=int, default=7, help="which tickets to generate (default 7)")
    ap.add_argument("--yes", action="store_true", help="skip the cost question for large real runs")
    ap.add_argument("--prompt", choices=["v1", "v2"], default="v2",
                    help="extraction prompt: v1 the vague first draft, v2 (default) with every field defined")
    a = ap.parse_args()

    if a.mock:
        import mock_openai
        _, url = mock_openai.start()
        os.environ["OPENAI_BASE_URL"] = url
        os.environ["OPENAI_API_KEY"] = "mock"
        print(f"Using the mock OpenAI API at {url}")
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print("Set OPENAI_API_KEY, or run with --mock to try it offline.", file=sys.stderr)
            return 2
        est = a.tickets * COST_PER_TICKET_USD
        print(f"Using the real OpenAI API: {a.tickets} tickets, about ${est:.2f}.")
        if est > 1.0 and not a.yes:
            if input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
                return 1

    from support_desk import extract
    from support_desk.data import tickets
    from support_desk.pipeline import process
    extract.PROMPT_VERSION = a.prompt

    batch = tickets(a.tickets, seed=a.seed)
    t0 = time.perf_counter()
    print(f"Processing {len(batch)} tickets")
    try:
        result = process(batch, workers=a.workers, show_reply=True)
    except Exception as exc:                     # openai.APIError and friends: say what failed, plainly
        print(f"\nStopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps({"tickets": batch, **result}, indent=1), encoding="utf-8")
    print(f"\nDigest:\n  {result['digest']}\n")
    print(f"Done in {time.perf_counter() - t0:.1f}s. Triage accuracy {result['triage_accuracy']:.0%}. "
          f"Results in {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
