"""Export bench/summary.json to the JSON files the results dashboard reads.

    python bench/export_dashboard.py OUT_DIR
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = [
    ("T01", "Scan completes", "The scanner returns without an exception"),
    ("T02", "Scan speed", "60 s or less, or at least 150 files/s"),
    ("T03", "Deterministic", "A second scan gives identical call sites"),
    ("T04", "Recall vs text search", "95% of baseline lines found, or every miss classified as not a call"),
    ("T05", "No false positives", "Zero LLM calls reported in a no-LLM control repo"),
    ("T06", "Provider consistency", "No call left 'unknown' when its model's maker is known"),
    ("T07", "Mock usage import", "Imported spend equals the generated bill (±0.5%)"),
    ("T08", "Re-import is idempotent", "Importing the same file again changes nothing"),
    ("T09", "Attribution", "Billed models the code calls are fully attributed; the decoy model is not"),
    ("T10", "Spend per API key", "Keys and per-key totals equal the bill (±0.5%)"),
    ("T11", "Pricing from tokens", "Tokens-only usage priced exactly from the catalog (±0.5%)"),
    ("T12", "Reports", "HTML, JSON and Markdown written; JSON complete; HTML loads nothing external"),
    ("T13", "Savings sanity", "0 ≤ low ≤ high ≤ monthly spend"),
    ("T14", "Finding locations", "95% of cited lines contain the call they claim"),
]


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    s = json.loads((HERE / "summary.json").read_text())
    rows = s["rows"]

    tests = []
    for tid, name, cond in TESTS:
        c = s["tests"].get(tid, {})
        tests.append({"id": tid, "test": name, "condition": cond, "pass": c.get("pass", 0),
                      "fail": c.get("fail", 0) + c.get("review", 0), "skip": c.get("skip", 0)})
    runs = sum(t["pass"] + t["fail"] for t in tests)
    controls = [r for r in rows if r["category"].startswith("negative")]
    head = {
        "repos": len(rows), "repos_passing": sum(1 for r in rows if r["status"] == "pass"),
        "test_runs": runs, "test_passes": sum(t["pass"] for t in tests), "test_fails": sum(t["fail"] for t in tests),
        "raw_recall_pct": s["recall"]["raw_pct"], "adjusted_recall_pct": s["recall"]["adjusted_pct"],
        "baseline_lines": s["recall"]["baseline_lines"], "noise_lines": s["recall"]["noise_lines"],
        "unexplained_misses": s["recall"]["unexplained_misses"],
        "control_repos": len(controls), "control_false_positives": sum(r["call_sites"] or 0 for r in controls),
        "files": s["totals"]["files"], "call_sites": s["totals"]["call_sites"], "scan_secs": s["totals"]["scan_secs"],
    }
    cats = defaultdict(lambda: {"repos": 0, "passing": 0, "files": 0, "call_sites": 0, "scan_secs": 0.0,
                                "found": 0, "denom": 0})
    for r in rows:
        c = cats[r["category"]]
        c["repos"] += 1
        c["passing"] += r["status"] == "pass"
        c["files"] += r["files"] or 0
        c["call_sites"] += r["call_sites"] or 0
        c["scan_secs"] += r["scan_secs"] or 0
        if r["baseline"]:
            c["found"] += r["baseline_found"]
            c["denom"] += r["recall_denominator"]
    categories = [{"category": k, "repos": v["repos"], "passing": v["passing"], "files": v["files"],
                   "call_sites": v["call_sites"], "scan_secs": round(v["scan_secs"], 1),
                   "adjusted_recall_pct": round(100 * v["found"] / v["denom"]) if v["denom"] else None}
                  for k, v in cats.items()]
    noise = [{"kind": k, "lines": v} for k, v in sorted(s["noise"].items(), key=lambda kv: -kv[1])]
    repos = [{k: r[k] for k in ("repo", "category", "status", "files", "scan_secs", "call_sites", "providers",
                                "languages", "adjusted_recall_pct", "recall_pct", "mock_format", "mock_spend",
                                "attributed_share", *[t[0] for t in TESTS])} for r in rows]

    for name, data in (("headline", [head]), ("tests", tests), ("categories", categories), ("noise", noise),
                       ("repos", repos), ("fixes", json.loads((HERE / "fixes.json").read_text()))):
        (out / f"{name}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print(json.dumps(head, indent=1))


if __name__ == "__main__":
    main()
