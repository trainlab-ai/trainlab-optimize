"""Stage 2 of the 100-repo experiment test: run `trainlab experiment` on every repo and check it.

    python bench/experiments/prepare.py WORK      # stage 1: workspaces (scan + mock usage)
    python bench/experiments/run.py WORK          # stage 2: experiments against the model zoo

Per repo, one or two experiments:
  auto      a call site `trainlab experiment plan` proposes (model + cheaper candidates)
  explicit  the repo's top call site with its billed model named by hand (--baseline/--provider)
  dataset   no usable call site (negative controls): a bare dataset with a cross-provider candidate
Each varies how the reference is obtained (production outputs, fresh baseline calls, expected
answers), whether a judge is used, and whether the run is cut by its budget and resumed.

Checks (X01-X13): run completes; cost equals the zoo's token log x catalog prices; no false
PASS / no false FAIL against each candidate's designed accuracy; per-sample pass/fail equals
the truth; nothing sent anywhere but the zoo and no key on disk; retries recovered every 429/500;
report shows the finding as validated with the measured saving; saving % matches the log;
generation without a judge never validates; budget never exceeded; resume never pays twice;
the CLI refuses exactly the runs that can't give a verdict, before spending.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import datasets  # noqa: E402
import zoo  # noqa: E402

from trainlab_optimize.analyzer.engine import analyze  # noqa: E402
from trainlab_optimize.analyzer.rules import family, load_candidates  # noqa: E402
from trainlab_optimize.experiment import candidates_for, eligible  # noqa: E402
from trainlab_optimize.experiment import gates  # noqa: E402
from trainlab_optimize.experiment.clients import COMPAT, resolve  # noqa: E402
from trainlab_optimize.experiment.runner import Config, load_results, plan, run  # noqa: E402
from trainlab_optimize.experiment.samples import load_samples  # noqa: E402
from trainlab_optimize.pricing import PriceCatalog  # noqa: E402
from trainlab_optimize.workspace import Workspace  # noqa: E402

TODAY = date(2026, 10, 14)
DESIGNS = [0.999, 0.98, 0.955, 0.90, 0.80, 0.30]       # candidate accuracies, cycled
JUDGE = "gpt-4.1"
CAT = PriceCatalog.load()
SECRET = "SECRET-7f3a9"
_counter = [0]
_clock = threading.Lock()


def sh(s: str) -> int:
    """Stable hash (Python's hash() changes between processes)."""
    import hashlib
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


def next_design() -> float:
    with _clock:
        _counter[0] += 1
        return DESIGNS[_counter[0] % len(DESIGNS)]


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def setup_env(base: str) -> None:
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY"):
        os.environ.pop(k, None)
    os.environ.update({"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
                       "ANTHROPIC_BASE_URL": base, "TRAINLAB_BASE_GEMINI": base, "TRAINLAB_BASE_BEDROCK": base,
                       "TRAINLAB_BASE_AZURE": base, "TRAINLAB_BASE_OPENAI_COMPATIBLE": f"{base}/v1",
                       "AWS_ACCESS_KEY_ID": "AKIAZOOTEST000000000", "AWS_SECRET_ACCESS_KEY": f"aws-{SECRET}",
                       "AWS_DEFAULT_REGION": "us-east-1", "AWS_EC2_METADATA_DISABLED": "true",
                       "ANTHROPIC_API_KEY": f"ant-{SECRET}", "GEMINI_API_KEY": f"gem-{SECRET}",
                       "AZURE_OPENAI_API_KEY": f"az-{SECRET}", "TRAINLAB_API_KEY": f"tl-{SECRET}"})
    for prov, (_, keys) in COMPAT.items():
        os.environ[f"TRAINLAB_BASE_{prov.upper()}"] = f"{base}/v1"
        for k in keys:
            os.environ[k] = f"{prov}-{SECRET}"
    os.environ["OPENAI_BASE_URL"] = f"{base}/v1"


def supported(m: str, provider: str | None = None) -> bool:
    return resolve(m, provider).api != "litellm" and CAT.lookup(m, None) is not None


# ------------------------------------------------------------------ experiment specs
def specs_for(ws: Workspace, meta: dict) -> list[dict]:
    scan = ws.load_scan()
    usage = ws.load_usage()
    if scan is None:
        return []
    a = analyze(scan, usage, today=TODAY)
    out = []
    if meta.get("category") in ("negative-controls", "negative-traps"):
        # No LLM code to scan: the dataset-only path (CSV, no workspace scan, CLI), cross-provider.
        out.append({"kind": "dataset", "site": None, "site_id": None, "baseline": "claude-sonnet-4-5",
                    "provider": "anthropic", "csv": True, "hints": [], "spend": None, "system": None,
                    "candidates": ["claude-haiku-4-5", "azure/gpt-4o-mini", "gemini/gemini-2.5-flash-lite",
                                   "mistral/mistral-small-latest"]})
    for row in eligible(a, CAT)[:2]:
        prov = row.get("provider")
        if not supported(row["model"], prov) and resolve(row["model"], prov).api == "litellm":
            continue
        cands = [c for c in row["candidates"] if supported(c)][:3]
        if cands and CAT.lookup(row["model"], prov):
            out.append({"kind": "auto", "site": row["site"], "site_id": row["site_id"], "baseline": row["model"],
                        "provider": prov, "candidates": cands, "hints": row["task_hints"],
                        "spend": row["monthly_spend_usd"], "system": _system(scan, row["site_id"])})
    if out:
        return out
    # explicit: the repo's most expensive billed model that has cheaper candidates, at its top chat site
    by_cost = defaultdict(float)
    prov_of = {}
    for r in usage:
        by_cost[r.model] += r.cost_usd or 0.0
        prov_of[r.model] = r.provider
    decoy = (meta.get("mock") or {}).get("decoy")
    chats = sorted((s for s in scan.call_sites if s.kind == "chat" and not s.is_test), key=lambda s: s.path)
    for model, _ in sorted(by_cost.items(), key=lambda kv: -kv[1]):
        if model == decoy:
            continue
        prov = prov_of.get(model)
        if not supported(model, prov):
            continue
        current = CAT.lookup(model, prov)
        fam = prov if prov in load_candidates() else (resolve(model, prov).provider)
        cands = [c for c in load_candidates().get(fam, []) if supported(c) and current
                 and CAT.lookup(c, None).blended_per_1k() < current.blended_per_1k()][:3]
        if not cands:
            continue
        site = chats[0] if chats else None
        return [{"kind": "explicit", "site": f"{site.path}:{site.line}" if site else None,
                 "site_id": site.id if site else None, "baseline": model, "provider": prov, "candidates": cands,
                 "hints": list(site.task_hints) if site else [], "spend": None,
                 "system": _system(scan, site.id) if site else None}]
    return [{"kind": "dataset", "site": None, "site_id": None, "baseline": "claude-sonnet-4-5", "provider": "anthropic",
             "candidates": ["claude-haiku-4-5", "gpt-4o-mini"], "hints": [], "spend": None, "system": None}]


def _system(scan, sid):
    s = next((x for x in scan.call_sites if x.id == sid), None)
    return s.prompt_preview if s and s.prompt_preview and len(s.prompt_preview) > 20 else None


# ------------------------------------------------------------------ one experiment
def run_spec(repo: str, ws: Workspace, spec: dict, idx: int, zoo_base: str, zoo_log: Path, use_cli: bool) -> dict:
    name = f"{repo.replace('/', '__')}-{idx}"
    task = datasets.task_for(spec["hints"], name)
    mode = ["reference", "fresh", "gold", "fresh"][sh(name) % 4] if spec["kind"] != "dataset" else "gold"
    judge = JUDGE if task == "generation" and (sh(name) % 5) else None
    stress = "budget" if idx == 0 and sh(repo) % 10 == 0 else "resume" if idx == 0 and sh(repo) % 10 == 1 else None
    n = 100
    cfg_routes = Config(baseline=spec["baseline"], candidates=spec["candidates"], baseline_provider=spec["provider"])
    zid = {m: cfg_routes.route(m).model for m in [spec["baseline"], *spec["candidates"]]}
    design = {c: next_design() for c in spec["candidates"]}
    pb = 0.995 if task != "generation" else 0.98
    profile = {zid[spec["baseline"]]: pb, **{zid[c]: design[c] for c in spec["candidates"]}}
    rows = datasets.build(name, task, n, None, name)
    system = spec.get("system") or datasets.SYSTEM[task]
    entries, sample_lines = [], []
    for i, r in enumerate(rows):
        key = zoo.key_of(r["user"])
        e = {"key": key, "exp": name, "task": task, "gold": r["gold"], "labels": r.get("labels"), "profile": profile,
             "json_break": {zid[c]: 0.3 for c in spec["candidates"]}}
        entries.append(e)
        params = {}
        if task == "classification" and i % 2:
            params["max_tokens"] = 10
        if task == "extraction" and i % 2:
            params["response_format"] = {"type": "json_object"}
        if task == "generation" and i % 3 == 0:
            params["max_tokens"] = 200
        if stress:
            params["max_tokens"] = 256          # stress runs: production-style output limit keeps the worst case small
        line = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": r["user"]}],
                "params": params, "site": spec["site"], "model": spec["baseline"]}
        if mode == "reference":
            out = zoo.answer(e, zid[spec["baseline"]])
            line["output"] = out
            line["input_tokens"] = zoo.tokens_in(system + r["user"])
            line["output_tokens"] = zoo.tokens_out(out)
        if mode == "gold":
            line["expected"] = json.dumps(r["gold"]) if task == "extraction" else r["gold"]
        sample_lines.append(line)
    req = urllib.request.Request(f"{zoo_base}/__register", data=json.dumps({"entries": entries}).encode(),
                                 headers={"content-type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=30).read()
    sdir = ws.path / "bench-samples"
    sdir.mkdir(parents=True, exist_ok=True)
    if spec.get("csv"):
        import csv as _csv
        sfile = sdir / f"{name}.csv"
        with open(sfile, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["ticket text", "correct answer"])
            for x in sample_lines:
                w.writerow([x["messages"][-1]["content"], x.get("expected", "")])
        samples = load_samples(sfile, input_column="ticket text", expected_column="correct answer", system=system)
        ws = Workspace(ws.path / "dataset-only")              # a workspace with no scan at all
        ws.ensure()
        use_cli = True
    else:
        sfile = sdir / f"{name}.jsonl"
        sfile.write_text("\n".join(json.dumps(x) for x in sample_lines) + "\n")
        samples = load_samples(sfile)
    for s in samples:                                          # captured token counts travel in meta
        src = next(x for x in sample_lines if x["messages"][-1]["content"] == s.messages[-1]["content"])
        if "input_tokens" in src:
            s.meta.update(input_tokens=src["input_tokens"], output_tokens=src["output_tokens"])

    cfg = Config(baseline=spec["baseline"], candidates=spec["candidates"], site=spec["site"], site_id=spec["site_id"],
                 n=n, judge=judge, concurrency=6, hints=spec["hints"], name=name, monthly_spend=spec["spend"],
                 baseline_provider=spec["provider"], max_cost=1.0)
    cfg.csv_system = system if spec.get("csv") else None
    p = plan(cfg, samples, CAT)
    full = round(max(p["max_cost_usd"] * 1.3, 0.0001), 6)
    r = {"repo": repo, "exp": name, "kind": spec["kind"], "task": task, "mode": mode, "judge": judge,
         "stress": stress, "baseline": spec["baseline"], "candidates": spec["candidates"], "design": design,
         "plan_max": p["max_cost_usd"], "cli": use_cli, "checks": {}}
    runs = []
    est = max(p["est_cost_usd"], 1e-6)
    # Exact cost of the candidate (and baseline) calls of a full run: the zoo is deterministic.
    exact = 0.0
    for e, line in zip(entries, sample_lines):
        text_in = "".join(m["content"] for m in line["messages"])
        for m in spec["candidates"] + ([] if mode == "reference" else [spec["baseline"]]):
            route = cfg_routes.route(m)
            price = CAT.lookup(m, route.provider) or CAT.lookup(route.model)
            exact += price.cost(zoo.tokens_in(text_in), zoo.tokens_out(zoo.answer(e, route.model)))
    # Stress runs go one call at a time with room for one worst-case call plus 10% of the expected
    # cost: they can make calls, and must stop part-way.
    first = round(p["worst_single_call_usd"] * 1.05 + exact * 0.1, 8)
    r.update(exact_cost=exact, worst_call=p["worst_single_call_usd"])
    budgets = [first] if stress == "budget" else [first, full] if stress == "resume" else [full]
    stress_concurrency = 1
    for i_b, b in enumerate(budgets):
        cfg.max_cost = b
        cfg.concurrency = stress_concurrency if stress and i_b == 0 else 6
        if use_cli:
            try:
                res = _run_cli(ws, cfg, sfile)
            except Refused as exc:
                # The CLI refuses, before spending anything, runs that can't produce a verdict.
                # Record that (X13) and run through the API so the other checks still apply.
                r["cli_refused"] = str(exc)
                res = run(cfg, samples, ws.path, CAT)
        else:
            res = run(cfg, samples, ws.path, CAT)
        runs.append(res)
    res = runs[-1]
    r["result"] = {k: res.get(k) for k in ("id", "task", "threshold", "reference", "recommended", "spent_usd",
                                            "spent_this_run_usd", "stopped", "calls", "data_sent_to")}
    r["candidates_result"] = [{k: c.get(k) for k in ("model", "verdict", "passed", "scored", "lower_95", "upper_95",
                                                      "savings_pct", "stopped_early", "errors", "samples_needed")}
                              for c in res["candidates"]]
    check(r, res, runs, budgets, cfg, entries, zid, zoo_log, ws, samples, sample_lines, mode, task, pb)
    # X13 the CLI refuses exactly the runs that can't give a verdict (free text without a judge,
    # fewer than 20 samples), before any call
    if use_cli:
        cant = (res.get("task") == "generation" and not cfg.judge) or len(samples) < 20
        # refused up front, or (no production outputs to detect free text from) stopped after the
        # baseline calls with no candidate called
        cand_calls = sum((res.get("models", {}).get(m) or {}).get("calls", 0) for m in cfg.candidates)
        refused = "cli_refused" in r or (cant and cand_calls == 0)
        r["checks"]["X13"] = t("pass" if refused == cant else "fail",
                               r.get("cli_refused", "ran") + f" (task {res.get('task')}, judge {cfg.judge}, "
                                                             f"{len(samples)} samples)")
    return r


class Refused(Exception):
    pass


def _run_cli(ws, cfg, sfile) -> dict:
    cmd = [sys.executable, "-m", "trainlab_optimize.cli", "--workspace", str(ws.path), "experiment", "run",
           "--samples", str(sfile), "--baseline", cfg.baseline, "--candidates", ",".join(cfg.candidates),
           "--max-cost", str(cfg.max_cost), "--n", str(cfg.n), "--name", cfg.name, "--yes",
           "--concurrency", str(cfg.concurrency)]
    if getattr(cfg, "csv_system", None):
        cmd += ["--system", cfg.csv_system, "--input-column", "ticket text", "--expected-column", "correct answer"]
    if cfg.site:
        cmd += ["--site", cfg.site]
    if cfg.baseline_provider:
        cmd += ["--provider", cfg.baseline_provider]
    if cfg.judge:
        cmd += ["--judge", cfg.judge]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=os.environ.copy())
    rf = ws.path / "experiments" / cfg.name / "result.json"
    if p.returncode == 2 and not rf.exists() and ("--judge MODEL" in p.stdout or "are needed for a verdict" in p.stdout):
        raise Refused(next(ln.strip(" !") for ln in p.stdout.splitlines()
                           if "--judge MODEL" in ln or "needed for a verdict" in ln))
    if p.returncode != 0 or not rf.exists():
        raise RuntimeError(f"cli exit {p.returncode}: {p.stderr[-800:]} {p.stdout[-400:]}")
    return json.loads(rf.read_text())


# ------------------------------------------------------------------ checks
def t(status, detail="", **kw):
    return {"status": status, "detail": detail, **kw}


def population_rate(task, mode, pc, pb, labels=5):
    if mode == "gold":
        return pc
    if task == "classification":
        return pc * pb + (1 - pc) * (1 - pb) / (labels - 1)
    if task == "extraction":
        return pc * pb
    return pc + (1 - pc) * (1 - pb)


def check(r, res, runs, budgets, cfg, entries, zid, zoo_log, ws, samples, sample_lines, mode, task, pb):
    C = r["checks"]
    C["X01"] = t("pass", f"{len(runs)} run(s)")
    log = [json.loads(x) for x in zoo_log.read_text().splitlines() if f'"{r["exp"]}"' in x]
    log = [x for x in log if x.get("exp") == r["exp"]]
    ok = [x for x in log if x["status"] == 200]
    # X02 cost: engine's recorded spend == zoo token log x catalog price
    by_zid = {}
    for m in [cfg.baseline, *cfg.candidates] + ([cfg.judge] if cfg.judge else []):
        by_zid.setdefault(cfg.route(m).model, m)
    zoo_cost2 = 0.0
    for x in ok:
        m = by_zid[x["model"]]
        route = cfg.route(m)
        price = CAT.lookup(m, route.provider) or CAT.lookup(route.model)
        zoo_cost2 += price.cost(x["input_tokens"], x["output_tokens"])
    eng = res["spent_usd"]
    C["X02"] = t("pass" if math.isclose(eng, zoo_cost2, rel_tol=1e-6, abs_tol=1e-6) else "fail",
                 f"engine ${eng:.6f} vs zoo log ${zoo_cost2:.6f}")
    # X12 budget: never exceeded; budget-stress runs stop on budget
    over = [(b, x["spent_this_run_usd"]) for b, x in zip(budgets, runs) if x["spent_this_run_usd"] > b + 1e-12]
    stress_ok = r["stress"] != "budget" or runs[-1]["stopped"] == "budget"
    C["X12"] = t("pass" if not over and stress_ok else "fail",
                 f"spent {[x['spent_this_run_usd'] for x in runs]} of {budgets}; stopped={runs[-1]['stopped']}")
    # X11 resume: no request answered twice for the same model, sample and purpose
    seen = defaultdict(int)
    for x in ok:
        seen[(x["model"], x["key"], x.get("sig"))] += 1
    dup = sum(1 for v in seen.values() if v > 1)
    calib_dups = 0
    # Not testable when one worst-case call costs more than the whole run: any budget that allows a
    # call then covers everything, so nothing can stop part-way (correct, given worst-case reservation).
    untestable = r["stress"] == "resume" and r["worst_call"] * 1.05 >= r["exact_cost"] * 0.9
    resumed_ok = r["stress"] != "resume" or untestable or (runs[0]["stopped"] == "budget"
                                                           and runs[-1]["calls"]["cached"] > 0
                                                           and runs[-1]["stopped"] is None)
    C["X11"] = t("pass" if dup == calib_dups and resumed_ok else "fail",
                 f"{dup} duplicate paid requests" + (f"; first run stopped={runs[0]['stopped']}, resumed run reused "
                                                     f"{runs[-1]['calls']['cached']} cached" if len(runs) > 1 else "")
                 + ("; resume not testable: one worst-case call costs more than the whole run" if untestable else ""))
    # X07 privacy
    hosts = set(res["data_sent_to"])
    leaked = []
    for f in (ws.path / "experiments").rglob("*"):
        if f.is_file() and SECRET in f.read_text(errors="ignore"):
            leaked.append(f.name)
    text_in_result = any("text" in ps for ps in res.get("per_sample", []))
    C["X07"] = t("pass" if all(h.startswith("127.0.0.1") for h in hosts) and not leaked and not text_in_result else "fail",
                 f"hosts {sorted(hosts)}; key leaks {leaked}; text in result {text_in_result}")
    # X08 retries: every injected 429/500 recovered
    injected = sum(1 for x in log if x["status"] in (429, 500))
    errors = sum(m["errors"] for m in res["models"].values())
    C["X08"] = t("pass" if errors == 0 else "fail", f"{injected} injected 429/500, {errors} unrecovered")
    # X06 per-sample truth
    ent = {e["key"]: e for e in entries}
    sample_key = {s.id: zoo.key_of(s.messages[-1]["content"]) for s in samples}
    wrong = []
    for ps in res.get("per_sample", []):
        e = ent[sample_key[ps["sample"]]]
        cand = zoo.answer(e, cfg.route(ps["model"]).model)
        if mode == "gold":
            ref_ok = True
            ref = None
        else:
            ref = zoo.answer(e, cfg.route(cfg.baseline).model)
            ref_ok = zoo.correct(e, cfg.route(cfg.baseline).model)
        c_ok = zoo.correct(e, cfg.route(ps["model"]).model)
        if task == "classification":
            truth = (c_ok and ref_ok) or (not c_ok and not ref_ok and _label(cand) == _label(ref))
        elif task == "extraction":
            truth = c_ok and ref_ok
        else:
            if ps.get("pass") is None:
                continue
            if cfg.judge is None:
                truth = None
            else:
                ref_text = e["gold"] if mode == "gold" else ref
                ab = zoo.judge(e, ref_text, cand, cfg.route(cfg.judge).model)
                ba = zoo.judge(e, cand, ref_text, cfg.route(cfg.judge).model)
                truth = gates.judge_pair(ab, ba)[0]
        if truth is not None and bool(ps["pass"]) != truth:
            wrong.append(f"{ps['model']}:{ps['sample']}")
    C["X06"] = t("pass" if not wrong else "fail", f"{len(wrong)} sample verdicts differ from truth", examples=wrong[:5])
    # X04/X05 verdict safety and power against the designed accuracy
    thr = res["threshold"]
    false_pass, false_fail, rows = [], [], []
    for c in res["candidates"]:
        rate = population_rate(task, mode, r["design"][c["model"]], pb)
        rows.append(f"{c['model']} true {rate:.3f} -> {c['verdict']}")
        if c["verdict"] == "PASS" and rate <= thr - 0.03:
            false_pass.append(c["model"])
        if c["verdict"] == "FAIL" and rate >= thr + 0.03:
            false_fail.append(c["model"])
    C["X04"] = t("pass" if not false_pass else "fail", "; ".join(rows), false_pass=false_pass)
    C["X05"] = t("pass" if not false_fail else "fail", "; ".join(rows), false_fail=false_fail)
    # X09 generation without a judge never validates
    if task == "generation" and not cfg.judge:
        bad = [c["model"] for c in res["candidates"] if c["verdict"] != "NEEDS-JUDGE"]
        C["X09"] = t("pass" if not bad else "fail", f"verdicts {[c['verdict'] for c in res['candidates']]}")
    else:
        C["X09"] = t("n/a")
    # X10 saving % = measured cost per call, from the zoo log
    best = next((c for c in res["candidates"] if c["model"] == res.get("recommended")), None)
    if best and best.get("savings_pct") is not None:
        cand_zid = cfg.route(best["model"]).model
        scored = {sample_key[ps["sample"]] for ps in res["per_sample"] if ps["model"] == best["model"] and ps.get("pass") is not None}
        cc = [x for x in ok if x["model"] == cand_zid and x["key"] in scored and not x.get("judge")]
        price_c = CAT.lookup(best["model"], cfg.route(best["model"]).provider) or CAT.lookup(cand_zid)
        cand_cpc = sum(price_c.cost(x["input_tokens"], x["output_tokens"]) for x in cc) / max(1, len(cc))
        base_zid = cfg.route(cfg.baseline).model
        price_b = CAT.lookup(cfg.baseline, cfg.baseline_provider) or CAT.lookup(base_zid)
        if mode == "reference":
            bl = [(x["input_tokens"], x["output_tokens"]) for x in sample_lines
                  if zoo.key_of(x["messages"][-1]["content"]) in {x["key"] for x in cc}]
        else:
            bl = [(x["input_tokens"], x["output_tokens"]) for x in ok
                  if x["model"] == base_zid and x["key"] in {y["key"] for y in cc} and not x.get("judge")]
        base_cpc = sum(price_b.cost(i, o) for i, o in bl) / max(1, len(bl))
        expect = 1 - cand_cpc / base_cpc if base_cpc else None
        C["X10"] = t("pass" if expect is not None and abs(expect - best["savings_pct"]) < 0.002 else "fail",
                     f"engine {best['savings_pct']:.4f} vs log {expect if expect is None else round(expect, 4)}")
    else:
        C["X10"] = t("n/a", "no passing candidate")
    # X03 report integration
    if cfg.site_id:
        a = analyze(ws.load_scan(), ws.load_usage(), today=TODAY, experiments=[res])
        fs = [f for f in a.findings if cfg.site_id in f.call_site_ids and f.rule in ("R101", "R110")]
        if res.get("recommended"):
            f = fs[0] if fs else None
            spend = next((x.monthly_spend_usd for x in a.attributions if x.call_site_id == cfg.site_id), None)
            exp_sav = spend * best["savings_pct"] if spend and best else None
            good = f is not None and f.tier == "validated" and (exp_sav is None or math.isclose(f.savings_high_usd or 0, exp_sav, rel_tol=1e-6))
            C["X03"] = t("pass" if good else "fail", f"finding {f.rule if f else None} tier {f.tier if f else None} "
                                                      f"saving {f.savings_high_usd if f else None} vs {exp_sav}")
        elif res["candidates"] and all(c["verdict"] == "FAIL" for c in res["candidates"]):
            f = fs[0] if fs else None
            good = f is None or (f.tier == "validated" and f.severity == "info" and not f.savings_high_usd)
            C["X03"] = t("pass" if good else "fail", f"all failed; finding {f.title if f else 'none (no R101)'}")
        else:
            C["X03"] = t("pass", "inconclusive: estimate kept")
    else:
        C["X03"] = t("n/a", "no call site")


def _label(s):
    from trainlab_optimize.experiment.gates import norm_label
    return norm_label(s)


# ------------------------------------------------------------------ driver
def main():
    work = Path(sys.argv[1])
    only = sys.argv[2:]
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    port = free_port()
    zoo_log = work / "zoo.log"
    zoo_log.unlink(missing_ok=True)
    zp = subprocess.Popen([sys.executable, str(HERE / "zoo.py"), str(port), str(zoo_log)], stdout=subprocess.PIPE)
    (work / "zoo.pid").write_text(str(zp.pid))
    zp.stdout.readline()
    base = f"http://127.0.0.1:{port}"
    setup_env(base)
    repos = sorted((work / "ws").iterdir())
    if only:
        repos = [r for r in repos if any(o.replace("/", "__") == r.name for o in only)]

    def one(i_d):
        i, d = i_d
        meta = json.loads((d / "meta.json").read_text())
        ws = Workspace(d)
        import shutil
        shutil.rmtree(d / "experiments", ignore_errors=True)
        shutil.rmtree(d / "dataset-only", ignore_errors=True)
        rows = []
        try:
            specs = specs_for(ws, meta)
        except Exception as exc:  # noqa: BLE001
            return [{"repo": meta["repo"], "error": f"spec: {exc}", "trace": traceback.format_exc()[-1500:], "checks": {}}]
        for j, spec in enumerate(specs):
            try:
                rows.append(run_spec(meta["repo"], ws, spec, j, base, zoo_log, use_cli=(i % 8 == 3)))
            except Exception as exc:  # noqa: BLE001
                rows.append({"repo": meta["repo"], "exp": f"{d.name}-{j}", "kind": spec["kind"], "checks":
                             {"X01": t("fail", f"{type(exc).__name__}: {exc}")}, "trace": traceback.format_exc()[-2000:]})
        return rows

    t0 = time.time()
    all_rows = []
    try:
        with cf.ThreadPoolExecutor(4) as ex:
            for rows in ex.map(one, enumerate(repos)):
                for r in rows:
                    all_rows.append(r)
                    st = "".join("." if c["status"] in ("pass", "n/a") else "F" for c in r.get("checks", {}).values())
                    vs = ",".join(f"{c['verdict'][:4]}" for c in r.get("candidates_result", []))
                    print(f"{r.get('repo', '?')[:34]:34} {r.get('kind', ''):8} {r.get('task', ''):14} "
                          f"{r.get('mode', ''):9} {st:12} {vs}", flush=True)
    finally:
        zp.terminate()
    (out_dir / "results.json").write_text(json.dumps({"seconds": round(time.time() - t0), "rows": all_rows}, indent=1))
    print(f"done in {time.time() - t0:.0f}s: {len(all_rows)} experiments")


if __name__ == "__main__":
    main()
