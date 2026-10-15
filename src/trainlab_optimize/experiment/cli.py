"""`trainlab experiment ...` commands."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ..pricing import PriceCatalog


def add_parser(sub, common) -> None:
    x = sub.add_parser("experiment", parents=[common],
                       help="test cheaper models on real inputs from a call site (uses your API keys)")
    xs = x.add_subparsers(dest="xcmd", required=True)

    def run_args(p):
        p.add_argument("--site", help="call site as path:line (from `trainlab experiment plan`)")
        p.add_argument("--samples", help="samples file or folder (default: captured samples in .trainlab/samples)")
        p.add_argument("--baseline", help="current model (default: the site's model)")
        p.add_argument("--provider", help="where the current model is served (default: the site's provider)")
        p.add_argument("--candidates", help="comma-separated models to test (default: cheaper models of the same provider)")
        p.add_argument("--n", type=int, default=100, help="number of samples (default 100)")
        p.add_argument("--judge", help="LLM judge for free-text answers, e.g. gpt-4.1 or claude-sonnet-4-5")
        p.add_argument("--task", choices=["classification", "extraction", "generation"],
                       help="how to score answers (default: detected from the outputs)")
        p.add_argument("--threshold", type=float, help="required pass rate (default 0.95; 0.90 for generation)")
        p.add_argument("--input-column", help="CSV: column with the input")
        p.add_argument("--expected-column", help="CSV: column with the correct answer")
        p.add_argument("--system", help="system prompt to add to samples that have none")
        p.add_argument("--redact", action="store_true", help="redact emails/phones/keys in samples before sending")

    pl = xs.add_parser("plan", help="list call sites worth testing, or show what a run would cost")
    run_args(pl)
    pl.set_defaults(fn=cmd_plan)

    r = xs.add_parser("run", help="run experiments and update the report (no --site: every call site with "
                                  "captured samples)")
    run_args(r)
    r.add_argument("--max-cost", type=float, default=2.0,
                   help="hard budget in USD for this run, shared across call sites (default 2)")
    r.add_argument("--open", action="store_true", help="open the HTML report when done")
    r.add_argument("--out", help="report folder (default .trainlab/report)")
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--calibrate", action="store_true",
                   help="also run the current model twice to measure how often it agrees with itself")
    r.add_argument("--rerun-baseline", action="store_true",
                   help="call the current model even when samples carry production outputs")
    r.add_argument("--name", help="name for this experiment (default: derived from its settings)")
    r.add_argument("--allow-unpriced", action="store_true", help="run models with no catalog price (no budget guard)")
    r.add_argument("--yes", "-y", action="store_true", help="don't ask before sending data to a new provider")
    r.set_defaults(fn=cmd_run)

    ls = xs.add_parser("list", help="experiments run so far")
    ls.set_defaults(fn=cmd_list)
    sh = xs.add_parser("show", help="results of one experiment")
    sh.add_argument("id")
    sh.set_defaults(fn=cmd_show)


def _ws(args):
    from ..workspace import Workspace
    return Workspace(args.workspace)


def _analysis(ws):
    from ..analyzer.engine import analyze
    from .runner import load_results
    scan = ws.load_scan()
    if scan is None:
        return None
    return analyze(scan, ws.load_usage(), experiments=load_results(ws.path))


def _site_info(args, ws, catalog):
    """(eligible row for --site, analysis) or (None, analysis)."""
    from . import eligible
    a = _analysis(ws)
    if a is None or not args.site:
        return None, a
    rows = eligible(a, catalog)
    path, _, line = args.site.rpartition(":")
    for row in rows:
        rp, _, rl = row["site"].rpartition(":")
        if rp == path and line.isdigit() and abs(int(rl) - int(line)) <= 3:
            return row, a
    for s in a.scan.call_sites:                             # any site, even without a cheaper model listed
        if s.path == path and line.isdigit() and abs(s.line - int(line)) <= 3:
            return {"site_id": s.id, "site": f"{s.path}:{s.line}", "model": s.model, "candidates": [],
                    "task_hints": list(s.task_hints),
                    "monthly_spend_usd": next((x.monthly_spend_usd for x in a.attributions if x.call_site_id == s.id),
                                              None)}, a
    return None, a


def _samples(args, ws, row):
    from .samples import load_samples, select
    src = Path(args.samples) if args.samples else ws.path / "samples"
    if not src.exists():
        return [], src
    samples = load_samples(src, do_redact=args.redact, input_column=args.input_column,
                           expected_column=args.expected_column, system=args.system)
    return select(samples, args.n, site=(row or {}).get("site") or args.site), src


def _config(args, row):
    from .runner import Config
    baseline = args.baseline or (row or {}).get("model")
    cands = [c.strip() for c in (args.candidates or "").split(",") if c.strip()] or (row or {}).get("candidates", [])
    return Config(baseline=baseline or "", candidates=cands, site=(row or {}).get("site") or args.site,
                  site_id=(row or {}).get("site_id"), task=args.task, threshold=args.threshold, n=args.n,
                  max_cost=getattr(args, "max_cost", 2.0), judge=args.judge,
                  concurrency=getattr(args, "concurrency", 4), calibrate=getattr(args, "calibrate", False),
                  rerun_baseline=getattr(args, "rerun_baseline", False), hints=(row or {}).get("task_hints", []),
                  name=getattr(args, "name", None), allow_unpriced=getattr(args, "allow_unpriced", False),
                  monthly_spend=(row or {}).get("monthly_spend_usd"),
                  baseline_provider=args.provider or (row or {}).get("provider"))


def _money(v):
    return "—" if v is None else f"${v:,.4f}" if v < 1 else f"${v:,.2f}"


def cmd_plan(args) -> int:
    from . import eligible
    from .runner import plan
    ws, catalog = _ws(args), PriceCatalog.load()
    row, a = _site_info(args, ws, catalog)
    if not args.site:
        if a is None:
            print("No scan found. Run `trainlab scan <path>` first, or pass --samples and --baseline/--candidates.")
            return 2
        rows = eligible(a, catalog)
        if not rows:
            print("No call site with a cheaper model to test.")
            return 0
        cap = ws.path / "samples"
        have = {p.stem for p in cap.glob("*.jsonl")} if cap.exists() else set()
        print(f"{'Call site':52} {'Model':26} {'Spend/mo':>10}  Candidates")
        for r in rows[:30]:
            slug = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in r["site"]).strip("_")
            mark = "  [samples captured]" if slug in have else ""
            site = r["site"] if len(r["site"]) <= 52 else "…" + r["site"][-51:]       # keep file and line visible
            print(f"{site:52} {str(r['model'])[:26]:26} {_money(r['monthly_spend_usd']):>10}  "
                  f"{', '.join(r['candidates'])}{mark}")
        if len(rows) > 30:
            print(f"... and {len(rows) - 30} more")
        print("\nNext: capture samples with instrument(capture={'sites': [...]}) or bring a JSONL/CSV, then\n"
              "  trainlab experiment plan --site <path:line> [--samples FILE]")
        return 0
    samples, src = _samples(args, ws, row)
    cfg = _config(args, row)
    if not cfg.baseline or not cfg.candidates:
        print("Pass --baseline and --candidates (the site has no known model or no cheaper candidate).", file=sys.stderr)
        return 2
    if not samples:
        print(f"No samples for {cfg.site} in {src}. Capture some with instrument(capture=...) or pass --samples.",
              file=sys.stderr)
        return 2
    p = plan(cfg, samples, catalog)
    _print_plan(cfg, p)
    return 0


def _print_plan(cfg, p) -> None:
    print(f"Experiment on {cfg.site or 'samples'}: {p['samples']} samples "
          f"({p['with_reference']} with production output, {p['with_expected']} with expected answer)")
    for m in p["models"]:
        print(f"  {m['role']:9} {m['model']:42} {m['calls']:>5} calls  est {_money(m.get('est_cost_usd'))}  "
              f"max {_money(m.get('max_cost_usd'))}  -> {m.get('endpoint', '')}")
    print(f"  estimated {_money(p['est_cost_usd'])}, at most {_money(p['max_cost_usd'])}; budget {_money(p['budget_usd'])}")
    print("  data is sent to: " + "; ".join(f"{h} ({', '.join(ms)})" for h, ms in p["data_goes_to"].items()))
    for pr in p["problems"]:
        print(f"  ! {pr}")


def _slug(site: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in site).strip("_")


def _run_one(args, ws, catalog, row, *, compact: bool = False):
    """Plan and run one experiment. Returns the result dict, or an exit code when it can't run."""
    from .runner import plan, run
    samples, src = _samples(args, ws, row)
    cfg = _config(args, row)
    if not cfg.baseline or not cfg.candidates:
        print("Pass --baseline and --candidates (the site has no known model or no cheaper candidate).", file=sys.stderr)
        return 2
    if not samples:
        print(f"No samples in {src}.", file=sys.stderr)
        return 2
    p = plan(cfg, samples, catalog)
    if compact and p.get("blocking"):
        print("  skipped: " + "; ".join(p["blocking"]))
        return 2
    _print_plan(cfg, p)
    blocking = [x for x in p["problems"] if "no API key" in x or ("no price" in x and not cfg.allow_unpriced)
                or x in p.get("blocking", [])]
    if blocking:
        print("Fix the problems above first.", file=sys.stderr)
        return 2
    base_host = cfg.route(cfg.baseline).host
    new_hosts = [h for h in p["data_goes_to"] if h != base_host]
    if new_hosts and not args.yes:
        if not sys.stdin.isatty():
            print(f"Samples would be sent to {', '.join(new_hosts)}, which doesn't serve this call today. "
                  "Re-run with --yes to confirm.", file=sys.stderr)
            return 2
        if input(f"Send samples to {', '.join(new_hosts)}? [y/N] ").strip().lower() not in ("y", "yes"):
            return 1
    done = [0]

    def progress(purpose, sid, e):
        done[0] += 1
        if done[0] % 20 == 0:
            print(f"  {done[0]} calls ...", flush=True)
    try:
        res = run(cfg, samples, ws.path, catalog, progress=progress)
    except ValueError as exc:
        print(f"Experiment failed: {exc}", file=sys.stderr)
        return 1
    _print_result(res)
    return res


def _write_report(args) -> int:
    from argparse import Namespace
    from ..cli import cmd_report
    rargs = Namespace(workspace=args.workspace, out=getattr(args, "out", None),
                      format=getattr(args, "format", None) or "html,json", fail_on="none")
    if _ws(args).load_scan() is None:
        return 0
    print()
    rc = cmd_report(rargs)
    if rc == 0 and getattr(args, "open", False):
        from ..trace_cmd import open_report
        open_report(rargs)
    return rc


def _run_all(args, ws, catalog) -> int:
    """Every call site that has captured samples and a cheaper model, under one total budget."""
    import copy
    from . import eligible
    a = _analysis(ws)
    if a is None:
        print("No scan found. Run `trainlab trace --capture -- <your app>` (or `trainlab scan .`) first.",
              file=sys.stderr)
        return 2
    cap = ws.path / "samples"
    have = {p.stem for p in cap.glob("*.jsonl")} if cap.exists() else set()
    rows = [r for r in eligible(a, catalog) if _slug(r["site"]) in have and r["candidates"]]
    if not rows:
        if not have:
            print("No captured samples yet. Record some with:\n  trainlab trace --capture -- python app.py\n"
                  "or bring your own: trainlab experiment run --site path:line --samples FILE", file=sys.stderr)
        else:
            print("The call sites with samples already use the cheapest listed model; nothing to test.")
        return 2
    total = args.max_cost
    print(f"Testing {len(rows)} call site(s) with captured samples; budget ${total:.2f} in total.")
    remaining, ran, failed = total, 0, 0
    for i, row in enumerate(rows):
        share = remaining / (len(rows) - i)
        print(f"\n[{i + 1}/{len(rows)}] {row['site']}  {row['model']} -> {', '.join(row['candidates'])}"
              f"  (budget ${share:.2f})")
        sub = copy.copy(args)
        sub.site, sub.max_cost = row["site"], share
        res = _run_one(sub, ws, catalog, row, compact=True)
        if isinstance(res, dict):
            ran += 1
            remaining = max(0.0, remaining - (res.get("spent_this_run_usd") or 0.0))
        else:
            failed += 1
    print(f"\nRan {ran} experiment(s)" + (f", skipped {failed}" if failed else "")
          + f"; spent ${total - remaining:.4f} of ${total:.2f}.")
    rc = _write_report(args)
    return rc if ran else 1


def cmd_run(args) -> int:
    ws, catalog = _ws(args), PriceCatalog.load()
    if not args.site and not args.samples:
        return _run_all(args, ws, catalog)
    row, _ = _site_info(args, ws, catalog)
    res = _run_one(args, ws, catalog, row)
    if not isinstance(res, dict):
        return res
    return _write_report(args)


def _print_result(res: dict) -> None:
    print(f"\nExperiment {res['id']}: {res['samples']} samples, task {res['task']}, threshold {res['threshold']:.0%}, "
          f"reference: {res['reference']}")
    if res.get("calibration"):
        print(f"  current model agrees with itself on {res['calibration']['baseline_self_agreement']:.0%}")
    if res.get("baseline_accuracy") is not None:
        print(f"  current model accuracy on expected answers: {res['baseline_accuracy']:.0%}")
    print(f"  {'Model':40} {'Verdict':13} {'Pass':>9} {'95% low':>8} {'Cost/call':>11} {'Saving':>7}")
    for c in res["candidates"]:
        pr = f"{c['passed']}/{c['scored']}"
        sav = "—" if c.get("savings_pct") is None else (f"{c['savings_pct']:.0%}" if c["savings_pct"] > 0
                                                         else f"+{-c['savings_pct']:.0%} cost")
        extra = f"  (~{c['samples_needed']} samples to settle)" if c.get("samples_needed") else ""
        extra += "  stopped early" if c.get("stopped_early") else ""
        extra += "  costs more per call: not recommended" if c.get("costs_more") else ""
        print(f"  {c['model'][:40]:40} {c['verdict']:13} {pr:>9} {c['lower_95']:>8.1%} "
              f"{_money(c.get('cost_per_call_usd')):>11} {sav:>7}{extra}")
    if res.get("recommended"):
        best = next(c for c in res["candidates"] if c["model"] == res["recommended"])
        ms = f", about ${best['monthly_savings_usd']:,.0f}/month" if best.get("monthly_savings_usd") else ""
        print(f"\n  Validated: `{res['recommended']}` keeps quality at {best['savings_pct']:.0%} lower cost{ms}.")
    print(f"  Spent {_money(res['spent_this_run_usd'])} this run ({res['calls']['new_calls']} calls, "
          f"{res['calls']['cached']} reused); budget {_money(res['budget_usd'])}"
          + (f"; stopped: {res['stopped']}" if res.get("stopped") else ""))


def cmd_list(args) -> int:
    p = _ws(args).path / "experiments" / "index.json"
    idx = json.loads(p.read_text()) if p.exists() else []
    if not idx:
        print("No experiments yet.")
        return 0
    for x in idx:
        v = ", ".join(f"{m}: {r}" for m, r in x["verdicts"].items())
        print(f"{x['id']:16} {x['created']:20} {str(x.get('site')):32} {x['baseline']:24} {v}")
    return 0


def cmd_show(args) -> int:
    p = _ws(args).path / "experiments" / args.id / "result.json"
    if not p.exists():
        print(f"No experiment {args.id}.", file=sys.stderr)
        return 2
    _print_result(json.loads(p.read_text()))
    return 0
