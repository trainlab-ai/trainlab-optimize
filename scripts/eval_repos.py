"""Benchmark: scan real repos and compare detected call sites against a grep baseline.

Recall = share of grep-matched SDK call lines (non-test) that the scanner also found.
Reference set (shallow clones, Oct 2026): Aider-AI/aider mem0ai/mem0 microsoft/graphrag
khoj-ai/khoj browser-use/browser-use anthropics/claude-quickstarts
langchain-ai/open_deep_research pipecat-ai/pipecat mckaywrigley/chatbot-ui

usage:
  for r in <owner/repo ...>; do git clone --depth 1 https://github.com/$r OSS_DIR/$(basename $r); done
  python scripts/eval_repos.py OSS_DIR aider mem0 graphrag ...
"""
import collections
import json
import re
import sys
from datetime import date
from pathlib import Path

from trainlab_optimize.analyzer.engine import analyze
from trainlab_optimize.scanner import EXCLUDE_DIRS, iter_files, scan
from trainlab_optimize.scanner.python_ast import TEST_PATH_RE

BASELINE = re.compile(
    r"\.chat\.completions\.(?:create|parse)\s*\(|\.responses\.(?:create|parse)\s*\(|"
    r"\.messages\.(?:create|stream)\s*\(|\.embeddings\.create\s*\(|\blitellm\.a?completion\s*\(|"
    r"\.converse\s*\(|\.invoke_model\s*\(|\.models\.generate_content\s*\(|"
    r"\b(?:generateText|streamText|generateObject|streamObject)\s*\(")
EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs"}
JS_TEST = re.compile(r"\.(test|spec)\.[jt]sx?$")

root = Path(sys.argv[1])
out = {}
for name in sys.argv[2:]:
    repo = root / name
    res = scan(repo)
    a = analyze(res, [], today=date(2026, 10, 14))
    # baseline
    base = []
    for p in iter_files(repo):
        if p.suffix not in EXTS:
            continue
        rel = p.relative_to(repo).as_posix()
        if TEST_PATH_RE.search(rel) or JS_TEST.search(rel):
            continue
        try:
            for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                s = line.strip()
                if s.startswith(("#", "//", "*")) or len(line) > 300:
                    continue
                if BASELINE.search(line):
                    base.append((rel, i, s[:110]))
        except OSError:
            pass
    sites = collections.defaultdict(list)
    for s in res.call_sites:
        sites[s.path].append(s.line)
    missed = [b for b in base if not any(abs(b[1] - ln) <= 4 for ln in sites.get(b[0], []))]
    found = len(base) - len(missed)
    rules = collections.Counter(f.rule for f in a.findings)
    out[name] = {
        "files": res.files_scanned, "secs": round(res.duration_s, 1), "sites": len(res.call_sites),
        "kinds": dict(collections.Counter(s.kind for s in res.call_sites)),
        "providers": dict(collections.Counter(s.provider for s in res.call_sites)),
        "resolved_model_pct": round(100 * sum(1 for s in res.call_sites if s.model) / max(1, len(res.call_sites))),
        "baseline": len(base), "baseline_found": found,
        "recall_pct": round(100 * found / len(base)) if base else None,
        "missed": missed[:15], "missed_n": len(missed),
        "refs": len(res.model_refs), "findings": dict(rules),
        "parse_errors": len(res.parse_errors),
    }
    a_json = a.to_dict()
    (root / f"report-{name}.json").write_text(json.dumps(a_json, indent=1))

for name, r in out.items():
    print(f"\n=== {name}: files={r['files']} {r['secs']}s sites={r['sites']} {r['kinds']} "
          f"model_resolved={r['resolved_model_pct']}% refs={r['refs']} parse_err={r['parse_errors']}")
    print(f"    providers={r['providers']}")
    print(f"    grep baseline={r['baseline']} found={r['baseline_found']} recall={r['recall_pct']}%")
    print(f"    findings={r['findings']}")
    for m in r["missed"]:
        print(f"    MISS {m[0]}:{m[1]}  {m[2]}")
    if r["missed_n"] > 15:
        print(f"    ... {r['missed_n'] - 15} more misses")
