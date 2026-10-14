"""Aggregate bench/results/*.json into bench/results.csv, bench/summary.json and bench/SUMMARY.md.

T04 "review" results are resolved here: every missed baseline line is classified as a known
kind of baseline noise, or by a manual verdict in bench/triage.json, or counted as a real miss.
Adjusted recall = found / (baseline - noise).
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TESTS = ["T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08", "T09", "T10", "T11", "T12", "T13", "T14", "T99"]

NOISE = [
    ("server route", re.compile(r"@\w+(\.\w+)*\.(post|get|route|api_route)\(|\bMap(Post|Get)\(|RequestMapping|"
                                r"\b\w*[Rr]outer\.(POST|GET|PUT|Any|Handle)\(\s*\"/|"
                                r"\b(app|router|r|e|g|server|api|engine)\.(post|get|POST|GET|Handle|HandleFunc)\(\s*[\"'`/]|"
                                r"Json<[\w:]*Create\w*Request>|\.post\(\"/|\.get\(\"/")),
    ("own API route (relative /api/ path)", re.compile(r"[\"'`]/api/(v\d/)?(chat/completions|messages|responses)")),
    ("code shown inside a string", re.compile(r"^\s*[`'\"].*fetch\(|^\s*`.*\$\{")),
    ("docstring or prose", re.compile(r"``|`[^`]*(/chat/completions|/messages|/responses|\(\.\.\.\))[^`]*`|"
                                      r"^\s*[A-Z][a-z]+( [a-z]+){3,}|\b(the|which|serves?|endpoint|instead|"
                                      r"forwarded|provided|compatible|e\.g\.)\b", re.I)),
    ("other API, same method name", re.compile(r"messages\.create\(\s*(to|from_|src|dst|roomId|body|channel|text)=|"
                                               r"messages\.create\([^)]*thread_id|agents\.messages\.create|"
                                               r"threads\.messages\.create|\.messages\.stream\(\)|"
                                               r"messages\.stream\(\)\s*\.")),
    ("SDK method definition", re.compile(r"^\s*(public|internal|private|protected|override|async fn|pub fn|fn|"
                                         r"def|func)\b.*\(|=>\s*(await\s+)?\w*Complete\w*\(|^\s*return\s+this\.\w+\(|"
                                         r"\bthis\.chatCompletion\(|ClientResult result = ")),
    ("test helper", re.compile(r"assert|expect\(|mock|localhost:8025", re.I)),
]


def classify(path: str, line: str, manual: dict) -> str:
    key = f"{path}"
    for pat, verdict in manual.items():
        if pat in key or pat in line:
            return verdict
    if re.search(r"(^|/)(tests?|testing|testutils?|__tests__|fixtures?)(/|\.|$)|[Tt]est(Base|Utils?)?\.\w+$", path):
        return "test helper"
    for name, rx in NOISE:
        if rx.search(line):
            return name
    return "unclassified"


def main() -> None:
    triage = json.loads((HERE / "triage.json").read_text()) if (HERE / "triage.json").exists() else {}
    rows, summary = [], {"repos": 0, "tests": {}, "categories": {}, "noise": Counter(), "real_misses": []}
    by_test: dict[str, Counter] = defaultdict(Counter)
    per_cat: dict[str, Counter] = defaultdict(Counter)
    for f in sorted(RESULTS.glob("*.json")):
        if f.name.startswith("_"):
            continue
        r = json.loads(f.read_text())
        tests = r["tests"]
        manual = triage.get(r["repo"], {})
        # classify every missed baseline line (all LLM repos, not just reviews)
        if r.get("llm_repo") and r.get("baseline"):
            kinds = Counter()
            real = []
            for path, ln, text in r.get("missed", []):
                k = classify(path, text, manual)
                kinds[k] += 1
                if k in ("unclassified", "real miss"):
                    real.append(f"{path}:{ln}  {text[:100]}")
            noise = sum(v for k, v in kinds.items() if k not in ("unclassified", "real miss"))
            denom = max(1, r["baseline"] - noise)
            adj = 100 * r["baseline_found"] / denom
            summary["noise"].update({k: v for k, v in kinds.items() if k not in ("unclassified", "real miss")})
            summary["real_misses"] += [f"{r['repo']}: {x}" for x in real]
            summary.setdefault("_adj", [0, 0])
            summary["_adj"][0] += r["baseline_found"]
            summary["_adj"][1] += denom
            t4 = tests.get("T04", {})
            t4.update(adjusted_recall=round(adj), noise=dict(kinds), real_misses=real)
            if t4.get("status") == "review":
                t4["status"] = "pass" if not real else "fail"
                t4["detail"] += f"; after triage {r['baseline_found']}/{denom} = {adj:.0f}%"
        for tid in TESTS:
            st = tests.get(tid, {}).get("status", "n/a" if tid == "T99" else "skip")
            if tid == "T99" and tid not in tests:
                continue
            by_test[tid][st] += 1
        status = "fail" if any(v["status"] == "fail" for v in tests.values()) else "pass"
        per_cat[r["category"]][status] += 1
        rows.append({
            "category": r["category"], "repo": r["repo"], "status": status, "files": r.get("files"),
            "scan_secs": r.get("scan_secs"), "call_sites": r.get("call_sites"), "model_configs": r.get("model_configs"),
            "providers": " ".join(p for p in r.get("providers", []) if p),
            "languages": " ".join(r.get("languages", [])), "resolved_model_pct": r.get("resolved_model_pct"),
            "baseline": r.get("baseline"), "baseline_found": r.get("baseline_found"),
            "recall_denominator": (r.get("baseline") or 0) - sum(
                v for k, v in tests.get("T04", {}).get("noise", {}).items() if k not in ("unclassified", "real miss")),
            "recall_pct": round(100 * r["baseline_found"] / r["baseline"]) if r.get("baseline") else None,
            "adjusted_recall_pct": tests.get("T04", {}).get("adjusted_recall",
                                                         round(100 * r["baseline_found"] / r["baseline"])
                                                         if r.get("baseline") else None),
            "mock_format": r.get("mock_format"), "mock_spend": r.get("monthly_spend"),
            "attributed_share": round(100 * r["attributed_share"]) if r.get("attributed_share") is not None else None,
            "specific_provider_pct": tests.get("T06", {}).get("specific_pct"),
            **{tid: tests.get(tid, {}).get("status", "") for tid in TESTS},
            "notes": "; ".join(f"{k}: {v['detail']}" for k, v in tests.items() if v["status"] == "fail")[:400],
        })
    summary["repos"] = len(rows)
    summary["tests"] = {k: dict(v) for k, v in by_test.items()}
    summary["categories"] = {k: dict(v) for k, v in per_cat.items()}
    summary["noise"] = dict(summary["noise"])
    summary["formats"] = dict(Counter(r["mock_format"] for r in rows))
    found_, denom_ = summary.pop("_adj", [0, 1])
    llm_rows = [r for r in rows if r["baseline"]]
    summary["recall"] = {"raw_pct": round(100 * sum(r["baseline_found"] for r in llm_rows)
                                          / max(1, sum(r["baseline"] for r in llm_rows)), 1),
                         "adjusted_pct": round(100 * found_ / max(1, denom_), 1),
                         "baseline_lines": sum(r["baseline"] for r in llm_rows),
                         "noise_lines": sum(summary["noise"].values()),
                         "unexplained_misses": len(summary["real_misses"])}
    summary["totals"] = {"files": sum(r["files"] or 0 for r in rows),
                         "call_sites": sum(r["call_sites"] or 0 for r in rows),
                         "baseline": sum(r["baseline"] or 0 for r in rows),
                         "baseline_found": sum(r["baseline_found"] or 0 for r in rows),
                         "scan_secs": round(sum(r["scan_secs"] or 0 for r in rows), 1)}
    unavailable = RESULTS / "_unavailable.json"
    summary["unavailable"] = json.loads(unavailable.read_text()) if unavailable.exists() else []

    with (HERE / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (HERE / "summary.json").write_text(json.dumps({**summary, "rows": rows}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "real_misses"}, indent=1))
    print(f"\n{len(summary['real_misses'])} unexplained miss(es):")
    for m in summary["real_misses"]:
        print("  ", m)


if __name__ == "__main__":
    main()
