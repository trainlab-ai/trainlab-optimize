"""Frozen baseline for the 100-repo suite.

    python bench/freeze.py save     # results of a --lock run become bench/golden.json
    python bench/freeze.py check    # compare the current results with the golden baseline

A frozen run uses exactly the repos and commits in bench/repos.lock.json
(`python bench/run.py --lock --fresh`), so any difference comes from trainlab, not from
the repos changing. `check` exits 1 on a regression (a test that passed now fails, a call
site that was found is gone, recall went down) and 0 otherwise; new call sites are listed
for review but don't fail the check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
GOLDEN = HERE / "golden.json"
TESTS = [f"T{i:02d}" for i in range(1, 15)] + ["T99"]
RANK = {"pass": 0, "skip": 0, "n/a": 0, "review": 1, "fail": 2}


def current() -> dict:
    out = {}
    summary = json.loads((HERE / "summary.json").read_text()) if (HERE / "summary.json").exists() else {}
    triaged = {r["repo"]: r for r in summary.get("rows", [])}
    for f in sorted(RESULTS.glob("*.json")):
        if f.name.startswith("_"):
            continue
        r = json.loads(f.read_text())
        row = triaged.get(r["repo"], {})
        out[r["repo"]] = {
            "category": r["category"], "sha": r.get("sha"),
            "tests": {t: (row.get(t) or r["tests"].get(t, {}).get("status", "n/a")) for t in TESTS
                      if t in r["tests"] or row.get(t)},
            "sites": r.get("site_keys", []),
            "baseline": r.get("baseline"), "baseline_found": r.get("baseline_found"),
            "adjusted_recall_pct": row.get("adjusted_recall_pct"),
            "mock_spend": r.get("monthly_spend"),
        }
    return out


def save() -> None:
    cur = current()
    missing_sha = [k for k, v in cur.items() if not v["sha"]]
    if missing_sha:
        sys.exit(f"{len(missing_sha)} result(s) are not from a frozen run (no commit): run bench/run.py --lock --fresh")
    GOLDEN.write_text(json.dumps({"repos": cur}, indent=1))
    print(f"golden baseline saved: {len(cur)} repos, {sum(len(v['sites']) for v in cur.values())} call sites")


def check() -> int:
    gold = json.loads(GOLDEN.read_text())["repos"]
    cur = current()
    regressions, changes = [], []
    for repo, g in gold.items():
        c = cur.get(repo)
        if c is None:
            regressions.append(f"{repo}: no result in this run")
            continue
        if c["sha"] and g["sha"] and c["sha"] != g["sha"]:
            changes.append(f"{repo}: scanned commit {c['sha'][:8]} differs from frozen {g['sha'][:8]}")
        for t, gs in g["tests"].items():
            cs = c["tests"].get(t, "n/a")
            if RANK.get(cs, 2) > RANK.get(gs, 0):
                regressions.append(f"{repo}: {t} {gs} -> {cs}")
        gone = sorted(set(g["sites"]) - set(c["sites"]))
        new = sorted(set(c["sites"]) - set(g["sites"]))
        if gone:
            regressions.append(f"{repo}: {len(gone)} call site(s) no longer found, e.g. {gone[0]}")
        if new:
            changes.append(f"{repo}: {len(new)} new call site(s), e.g. {new[0]}")
        ga, ca = g.get("adjusted_recall_pct"), c.get("adjusted_recall_pct")
        if isinstance(ga, (int, float)) and isinstance(ca, (int, float)) and ca < ga:
            regressions.append(f"{repo}: recall {ga}% -> {ca}%")
    print(f"Compared {len(cur)} repos with the frozen baseline ({len(gold)} repos).")
    for c in changes:
        print("  changed:   ", c)
    for r in regressions:
        print("  REGRESSION:", r)
    print("OK: no regressions" if not regressions else f"{len(regressions)} regression(s)")
    return 1 if regressions else 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "save":
        save()
    else:
        sys.exit(check())
