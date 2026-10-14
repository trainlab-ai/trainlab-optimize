"""100-repo benchmark: clone -> scan -> mock usage -> analyze -> test cases -> results.

    python bench/run.py [--work DIR] [--only CATEGORY] [--fresh]

Results go to bench/results/<category>__<owner>__<repo>.json (one file per repo; the run
resumes where it left off). `python bench/report.py` aggregates them. Test cases are
documented in bench/TESTS.md.
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# Frozen runs price with the catalog bundled in this commit, never a newer `prices update` cache
import os  # noqa: E402
os.environ["TRAINLAB_PRICES"] = str(HERE / "_bundled_prices_only.json")

from mock_usage import FORMATS, generate  # noqa: E402

from trainlab_optimize.analyzer.engine import analyze  # noqa: E402
from trainlab_optimize.analyzer.spend import base_model  # noqa: E402
from trainlab_optimize.importers.auto import read_any  # noqa: E402
from trainlab_optimize.report import write_reports  # noqa: E402
from trainlab_optimize.scanner import iter_files, scan  # noqa: E402
from trainlab_optimize.scanner.generic import OTHER_TEST  # noqa: E402
from trainlab_optimize.scanner.providers import maker_of  # noqa: E402
from trainlab_optimize.scanner.python_ast import TEST_PATH_RE  # noqa: E402
from trainlab_optimize.workspace import Workspace  # noqa: E402

TODAY = date(2026, 10, 14)
RESULTS = HERE / "results"
CALL_KINDS = ("chat", "embedding", "batch")

# Text baseline: lines a simple search would call "an LLM call". Recall is measured against it.
BASELINE = re.compile(
    r"\.chat\.completions\.(?:create|parse)\s*\(|\.responses\.(?:create|parse)\s*\(|"
    r"\.messages\.(?:create|stream)\s*\(|\.embeddings\.create\s*\(|\blitellm\.a?completion\s*\(|"
    r"\.converse\s*\(|\.invoke_model\s*\(|\.models\.generate_content\s*\(|"
    r"\b(?:generateText|streamText|generateObject|streamObject)\s*\(|"
    r"\.Chat\.Completions\.New|\.Messages\.New\(|\bCreateChatCompletion(?:Stream)?\s*\(|\.Responses\.New\(|"
    r"\.chat\(\)\s*\.completions\(\)\s*\.create|\.messages\(\)\s*\.create|CompleteChat(?:Async)?\(|"
    r"GetChatCompletions(?:Streaming)?(?:Async)?\s*\(|"
    r"\.(?:chat|messages|embeddings)\(\s*parameters:|->(?:chat|messages|embeddings|responses)\(\)\s*->\s*create|"
    r"\.chat\(\)\s*\.create\(|\.chatCompletions?\(|ChatModel\.builder\(\)|chatClient\.prompt\(|"
    r"\bGeneration\.call\(|\.chat\.complete(?:_async)?\(|\bollama\.(?:chat|generate)\(|"
    r"[\"'`][^\"'`\s]*(?:/chat/completions|api\.anthropic\.com/v1/messages|/v1/messages|"
    r"services/aigc/text-generation)[\"'`?]")
EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".go", ".java", ".kt", ".cs", ".rb", ".php", ".rs", ".ipynb"}


# ---------------------------------------------------------------- helpers
BLOB_LIMIT = "--filter=blob:limit=25m"      # notebooks can be large; the scanner skips files over its own limits


def clone(full: str, dest: Path, timeout: int = 900, sha: str | None = None) -> tuple[bool, str]:
    """Shallow clone of HEAD, or of one pinned commit (frozen runs)."""
    url = f"https://github.com/{full}.git"
    if sha:
        cmds = [["git", "init", "-q", str(dest)],
                ["git", "-C", str(dest), "fetch", "-q", "--depth", "1", BLOB_LIMIT, url, sha],
                ["git", "-C", str(dest), "checkout", "-q", "FETCH_HEAD"]]
    else:
        cmds = [["git", "clone", "--quiet", "--depth", "1", "--single-branch", BLOB_LIMIT, url, str(dest)]]
    try:
        for cmd in cmds:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if p.returncode != 0:
                return False, (p.stderr or "")[-300:]
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "clone timed out"


def baseline(repo: Path) -> list[tuple[str, int, str]]:
    out = []
    for p in iter_files(repo):
        if p.suffix not in EXTS:
            continue
        rel = p.relative_to(repo).as_posix()
        if TEST_PATH_RE.search(rel) or OTHER_TEST.search(rel):
            continue
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        if not BASELINE.search(text):
            continue
        if p.suffix == ".ipynb":
            # Notebooks: only code cells count (markdown cells hold documentation snippets)
            try:
                nb = json.loads(text)
                text = "\n".join("".join(c.get("source", "")) if isinstance(c.get("source"), list)
                                 else str(c.get("source", ""))
                                 for c in nb.get("cells", []) if c.get("cell_type") == "code")
            except ValueError:
                continue
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if s.startswith(("#", "//", "*")) or len(line) > 300:
                continue
            if BASELINE.search(line):
                out.append((rel, i, s[:140]))
    return out


def recall(res, base) -> tuple[int, list]:
    lines: dict[str, list[int]] = {}
    nb = set()
    for s in res.call_sites:
        lines.setdefault(s.path, []).append(s.line)
        if s.cell is not None:
            nb.add(s.path)
    missed = [b for b in base if (b[0] not in nb if b[0].endswith(".ipynb")
                                  else not any(abs(b[1] - ln) <= 4 for ln in lines.get(b[0], [])))]
    return len(base) - len(missed), missed


def evidence_ok(repo: Path, site) -> bool | None:
    """Does the cited line (+-2) contain the call it claims? None for notebooks."""
    if site.cell is not None:
        return None
    try:
        lines = (repo / site.path).read_text(errors="replace").splitlines()
    except OSError:
        return False
    window = "\n".join(lines[max(0, site.line - 3): site.line + 2])
    needles = [x for x in re.split(r"[.\s()]+", site.sdk_call.replace("HTTP ", "")) if len(x) > 2][-1:]
    if site.model:
        needles.append(site.model)
    if site.sdk_call.startswith("HTTP"):
        needles.append(site.sdk_call.split()[-1].strip("/").split("/")[-1].split(":")[-1])
    return any(n.lower() in window.lower() for n in needles)


def t(status: str, detail: str = "", **data) -> dict:
    return {"status": status, "detail": detail, **data}


def close(a: float, b: float, tol: float = 0.005) -> bool:
    return abs(a - b) <= tol * max(abs(b), 1e-9) + 0.01


# ---------------------------------------------------------------- one repo
def run_repo(full: str, cat: str, llm: bool, repo: Path, fmt: str, work: Path) -> dict:
    tests: dict[str, dict] = {}
    r: dict = {"repo": full, "category": cat, "llm_repo": llm, "mock_format": fmt, "tests": tests}

    t0 = time.time()
    try:
        res = scan(repo)
    except Exception as exc:  # noqa: BLE001 - a crash is the finding
        tests["T01"] = t("fail", f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-1500:])
        return r
    secs = time.time() - t0
    sites = res.call_sites
    calls = [s for s in sites if s.kind in CALL_KINDS]
    r.update(files=res.files_scanned, scan_secs=round(secs, 2), sites=len(sites), call_sites=len(calls),
             model_configs=sum(1 for s in sites if s.kind == "model_config"),
             providers=sorted({s.provider for s in sites}), parse_errors=len(res.parse_errors),
             languages=sorted({s.language for s in sites}),
             resolved_model_pct=round(100 * sum(1 for s in sites if s.model) / len(sites)) if sites else None)
    tests["T01"] = t("pass", f"{res.files_scanned} files, {len(sites)} sites")
    r["site_keys"] = sorted(f"{s.path}|{s.cell or ''}|{s.line}|{s.kind}|{s.provider}|{s.model or ''}" for s in sites)

    rate = res.files_scanned / max(secs, 1e-6)
    tests["T02"] = t("pass" if secs <= 60 or rate >= 150 else "fail", f"{secs:.1f}s, {rate:.0f} files/s")

    if secs <= 40:
        again = scan(repo)
        key = lambda x: sorted((s.id, s.path, s.line, s.kind, s.provider, s.model or "") for s in x.call_sites)  # noqa: E731
        tests["T03"] = t("pass" if key(again) == key(res) else "fail")
    else:
        tests["T03"] = t("skip", "scan over 40s")

    base = baseline(repo)
    found, missed = recall(res, base)
    r.update(baseline=len(base), baseline_found=found, missed=missed[:1000])
    if llm:
        if not base:
            tests["T04"] = t("skip", "no baseline matches")
        else:
            pct = 100 * found / len(base)
            tests["T04"] = t("pass" if pct >= 95 else "review", f"{found}/{len(base)} = {pct:.0f}%", recall=pct)
        tests["T05"] = t("skip", "LLM repo")
    else:
        tests["T04"] = t("skip", "negative control")
        bad = [f"{s.location} {s.sdk_call}" for s in calls]
        tests["T05"] = t("pass" if not bad else "fail", f"{len(bad)} LLM call site(s) reported", examples=bad[:10])

    specific = [s for s in calls if s.provider not in ("unknown", "openai_compatible")]
    inconsistent = [s.location for s in sites if s.provider == "unknown" and s.model and maker_of(s.model)]
    tests["T06"] = t("fail" if inconsistent else "pass" if calls else "skip",
                     f"{len(specific)}/{len(calls)} calls name a specific provider"
                     + (f"; {len(inconsistent)} unknown despite a known model" if inconsistent else ""),
                     specific_pct=round(100 * len(specific) / len(calls)) if calls else None,
                     examples=inconsistent[:5])

    # ---- mock usage
    mock_dir = Path(tempfile.mkdtemp(dir=work))
    exp = generate(full, sites, mock_dir, fmt)
    r["mock"] = {k: v for k, v in exp.items() if k not in ("path",)}
    ws = Workspace(mock_dir / "ws")
    try:
        records, kind = read_any(exp["path"], column_map=exp["column_map"])
        ws.add_usage(records)
        usage = ws.load_usage()
        a = analyze(res, usage, today=TODAY)
        got = a.spend.monthly_total
        tests["T07"] = t("pass" if records and close(got, exp["total_cost"]) else "fail",
                         f"{kind}: {len(records)} records, ${got:,.2f} vs expected ${exp['total_cost']:,.2f}")
        records2, _ = read_any(exp["path"], column_map=exp["column_map"])
        ws.add_usage(records2)
        a2 = analyze(res, ws.load_usage(), today=TODAY)
        tests["T08"] = t("pass" if close(a2.spend.monthly_total, got) else "fail",
                         f"after re-import ${a2.spend.monthly_total:,.2f}")

        decoy_b = base_model(exp["decoy"])
        unattr = {base_model(m) for m in a.coverage.get("unattributed_by_model", {})}
        problems = []
        if decoy_b not in unattr:
            problems.append("decoy model was attributed to code")
        for m in exp["code_models"]:
            b = base_model(m)
            ms = a.spend.by_model.get(b)
            if not ms:
                problems.append(f"{m}: billed spend not found")
                continue
            att = sum(x.monthly_spend_usd for x in a.attributions if base_model(x.model) == b)
            if att < 0.99 * ms.monthly_cost:
                problems.append(f"{m}: {att:,.0f} of {ms.monthly_cost:,.0f} attributed")
        tests["T09"] = t("fail" if problems else "pass",
                         "; ".join(problems) or f"{len(exp['code_models'])} code model(s) attributed, decoy unattributed",
                         attributed_share=a.coverage.get("attributed_share"))

        keys = a.spend.by_api_key
        ok = set(keys) == set(exp["by_key"]) and all(close(keys[k], exp["by_key"][k]) for k in keys)
        tests["T10"] = t("pass" if ok else "fail",
                         ", ".join(f"{k} ${v:,.0f}" for k, v in sorted(keys.items())) or "no keys")

        tok = exp["tokens_only"]
        trec, _ = read_any(tok["path"])
        ta = analyze(res, trec, today=TODAY)
        tests["T11"] = t("pass" if close(ta.spend.monthly_total, tok["expected"]) else "fail",
                         f"{tok['model']}: ${ta.spend.monthly_total:,.2f} vs ${tok['expected']:,.2f}")

        out = mock_dir / "report"
        write_reports(a, out, ("html", "json", "md"))
        problems = []
        for n in ("trainlab-report.html", "trainlab-report.json", "trainlab-report.md"):
            if not (out / n).exists() or (out / n).stat().st_size == 0:
                problems.append(f"{n} missing")
        if not problems:
            j = json.loads((out / "trainlab-report.json").read_text())
            for k in ("schema_version", "scan", "spend", "attribution", "coverage", "findings", "savings"):
                if k not in j:
                    problems.append(f"json missing {k}")
            html = (out / "trainlab-report.html").read_text()
            if re.search(r"<(script|link|img|iframe)[^>]+(src|href)=[\"']https?://", html, re.I):
                problems.append("html loads an external resource")
        tests["T12"] = t("fail" if problems else "pass", "; ".join(problems))

        lo, hi, tot = a.savings_low, a.savings_high, a.spend.monthly_total
        sane = (lo is None and hi is None) or (lo is not None and hi is not None and 0 <= lo <= hi <= tot + 1e-6)
        tests["T13"] = t("pass" if sane else "fail", f"${lo or 0:,.0f}-${hi or 0:,.0f} of ${tot:,.0f}")

        by_id = {s.id: s for s in sites}
        checked, bad = 0, []
        for f in a.findings:
            for sid in f.call_site_ids[:5]:
                s = by_id.get(sid)
                ok = evidence_ok(repo, s) if s else False
                if ok is None:
                    continue
                checked += 1
                if not ok:
                    bad.append(f"{f.rule} {s.location if s else sid}")
        pct = 100 * (checked - len(bad)) / checked if checked else None
        tests["T14"] = t("skip" if not checked else "pass" if pct >= 95 else "fail",
                         f"{checked - len(bad)}/{checked} cited lines contain the call" if checked else "no cited lines",
                         examples=bad[:8])
        r.update(findings={f.rule: sum(1 for x in a.findings if x.rule == f.rule) for f in a.findings},
                 monthly_spend=round(tot, 2), savings_high=round(hi, 2) if hi else None,
                 attributed_share=a.coverage.get("attributed_share"))
    except Exception as exc:  # noqa: BLE001
        tests["T99"] = t("fail", f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-2000:])
    finally:
        shutil.rmtree(mock_dir, ignore_errors=True)
    return r


# ---------------------------------------------------------------- driver
def slug(full: str) -> str:
    return full.replace("/", "__")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None, help="scratch dir for clones")
    ap.add_argument("--only", help="run one category")
    ap.add_argument("--fresh", action="store_true", help="ignore existing results")
    ap.add_argument("--rerun-failed", action="store_true", help="re-run repos with a failed test")
    ap.add_argument("--lock", action="store_true",
                    help="frozen run: exactly the repos and commits in bench/repos.lock.json")
    args = ap.parse_args()
    work = Path(args.work or tempfile.mkdtemp(prefix="tl-bench-"))
    work.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    plan = json.loads((HERE / "repos.json").read_text())

    # Work list: per category, take draws in order until the quota is met (spares replace failures).
    jobs: "queue.Queue" = queue.Queue(maxsize=2)
    unavailable: list[dict] = []
    fmt_counter = [0]

    def producer_locked():
        lock = json.loads((HERE / "repos.lock.json").read_text())
        for item in lock["repos"]:
            cat, full, sha = item["category"], item["repo"], item["sha"]
            if args.only and cat != args.only:
                continue
            outp = RESULTS / f"{cat}__{slug(full)}.json"
            if outp.exists() and not args.fresh:
                continue
            dest = work / slug(full)
            shutil.rmtree(dest, ignore_errors=True)
            ok, err = clone(full, dest, sha=sha)
            if not ok:
                unavailable.append({"repo": full, "category": cat, "sha": sha, "error": err.strip()})
                shutil.rmtree(dest, ignore_errors=True)
                continue
            jobs.put((cat, plan["categories"][cat]["llm"], full, dest, outp, sha))
        jobs.put(None)

    def producer():
        for cat, spec in plan["categories"].items():
            if args.only and cat != args.only:
                continue
            done = 0
            for full in spec["draw"]:
                if done >= spec["quota"]:
                    break
                outp = RESULTS / f"{cat}__{slug(full)}.json"
                if outp.exists() and not args.fresh:
                    prev = json.loads(outp.read_text())
                    failed = any(v["status"] == "fail" for v in prev.get("tests", {}).values())
                    if not (args.rerun_failed and failed):
                        done += 1
                        continue
                dest = work / slug(full)
                shutil.rmtree(dest, ignore_errors=True)
                ok, err = clone(full, dest)
                if not ok:
                    unavailable.append({"repo": full, "category": cat, "error": err.strip()})
                    shutil.rmtree(dest, ignore_errors=True)
                    continue
                done += 1
                jobs.put((cat, spec["llm"], full, dest, outp, None))
        jobs.put(None)

    th = threading.Thread(target=producer_locked if args.lock else producer, daemon=True)
    th.start()
    n = 0
    while True:
        job = jobs.get()
        if job is None:
            break
        cat, llm, full, dest, outp, sha = job
        fmt = FORMATS[(sum(map(ord, full))) % len(FORMATS)]
        n += 1
        t0 = time.time()
        try:
            result = run_repo(full, cat, llm, dest, fmt, work)
        except Exception as exc:  # noqa: BLE001 - harness bug: record it, keep going
            result = {"repo": full, "category": cat, "llm_repo": llm, "mock_format": fmt,
                      "tests": {"T99": t("fail", f"harness: {type(exc).__name__}: {exc}",
                                         trace=traceback.format_exc()[-2000:])}}
        finally:
            shutil.rmtree(dest, ignore_errors=True)
        result["wall_secs"] = round(time.time() - t0, 1)
        result["sha"] = sha
        outp.write_text(json.dumps(result, indent=1, default=str))
        st = {k: v["status"] for k, v in result["tests"].items()}
        fails = [k for k, v in st.items() if v in ("fail", "review")]
        print(f"[{n:3}] {cat:20} {full:45} sites={result.get('sites', '-'):>5} "
              f"{'OK' if not fails else 'CHECK ' + ','.join(fails)}", flush=True)
    th.join()
    if unavailable:
        (RESULTS / "_unavailable.json").write_text(json.dumps(unavailable, indent=1))
        print(f"{len(unavailable)} repo(s) could not be cloned and were replaced from the same category")


if __name__ == "__main__":
    main()
