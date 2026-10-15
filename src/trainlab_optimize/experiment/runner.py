"""Run an experiment: samples x (baseline + candidates) [+ judge], under a hard budget.

- Budget: before a call is sent, its worst-case cost (pessimistic input estimate + the output
  cap) is reserved; the call is skipped if that would exceed --max-cost. Actual cost from the
  provider's token counts replaces the reservation when it returns. Spend can't exceed the cap.
- Cache: every response is stored locally (.trainlab/experiments/<id>/calls.jsonl) keyed by
  model + request; re-running the same experiment reuses them, so an interrupted run resumes
  without paying twice.
- Order: candidates are called sample by sample, round robin, so a run cut short by the
  budget still compares all candidates on the same samples. A candidate that has clearly
  failed (upper confidence bound below the threshold) after 30 samples is not called again.
- Nothing is sent anywhere except the model endpoints listed in the plan.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import math
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..pricing import PriceCatalog
from . import gates
from .clients import Result, Route, call, resolve
from .samples import Sample


@dataclass
class Config:
    baseline: str
    candidates: list[str]
    site: str | None = None
    task: str | None = None
    threshold: float | None = None
    n: int = 100
    max_cost: float = 2.0
    judge: str | None = None
    concurrency: int = 4
    calibrate: bool = False
    rerun_baseline: bool = False
    early_stop: bool = True
    hints: list[str] = field(default_factory=list)
    name: str | None = None
    allow_unpriced: bool = False
    monthly_spend: float | None = None
    site_id: str | None = None
    baseline_provider: str | None = None      # where the call is served today (the site's provider)

    def route(self, model: str) -> Route:
        return resolve(model, self.baseline_provider if model == self.baseline else None)


# ------------------------------------------------------------------ cost helpers
def _price(catalog: PriceCatalog, model: str, route: Route):
    p = catalog.lookup(model, route.provider if route.provider not in ("litellm",) else None)
    return p or catalog.lookup(route.model)


def _upper_cost(price, prompt_chars: int, out_cap: int) -> float:
    return price.cost(math.ceil(prompt_chars / 3) + 20, out_cap) * 1.1


_CAP_KEYS = ("max_tokens", "max_completion_tokens", "max_output_tokens")


def _out_cap(s: Sample, ref_text: str | None = None, model: str = "") -> int:
    """The output limit a call runs with. Production's own limit when it set one; otherwise a
    generous one (4x the reference answer + 256, at least 512; 2048 when there is no reference;
    8192 for reasoning models) so the budget reservation is a true worst case."""
    for k in _CAP_KEYS:
        if isinstance(s.params.get(k), int):
            return s.params[k]
    ref = ref_text or s.reference or s.expected
    cap = max(512, 4 * math.ceil(len(ref) / 3) + 256) if ref else 2048
    if re.match(r"^(.*/)?(o\d|gpt-5)", model or ""):
        cap = max(cap, 8192)
    return min(cap, 32768)


def _with_cap(s: Sample, cap: int) -> dict:
    if any(isinstance(s.params.get(k), int) for k in _CAP_KEYS):
        return s.params
    return {**s.params, "max_tokens": cap}


class Budget:
    def __init__(self, cap: float):
        self.cap, self.spent, self.reserved = cap, 0.0, 0.0
        self.lock = threading.Lock()

    def reserve(self, amount: float) -> bool:
        with self.lock:
            if self.spent + self.reserved + amount > self.cap + 1e-12:
                return False
            self.reserved += amount
            return True

    def settle(self, reserved: float, actual: float) -> None:
        with self.lock:
            self.reserved -= reserved
            self.spent += actual


class CallCache:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data: dict[str, dict] = {}
        if path.exists():
            for ln in path.read_text().splitlines():
                try:
                    d = json.loads(ln)
                    self.data[d["key"]] = d
                except (ValueError, KeyError):
                    continue

    def get(self, key):
        return self.data.get(key)

    def put(self, entry: dict) -> None:
        with self.lock:
            self.data[entry["key"]] = entry
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _key(route: Route, messages, params, purpose: str) -> str:
    blob = json.dumps({"api": route.api, "base": route.base, "model": route.model, "m": messages, "p": params,
                       "purpose": purpose}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def experiment_id(cfg: Config, samples: list[Sample]) -> str:
    if cfg.name:
        return "".join(c if c.isalnum() or c in "-_" else "-" for c in cfg.name)[:60]
    blob = json.dumps({"s": cfg.site, "b": cfg.baseline, "c": sorted(cfg.candidates), "t": cfg.task,
                       "j": cfg.judge, "ids": sorted(s.id for s in samples)}, sort_keys=True)
    return "x-" + hashlib.sha256(blob.encode()).hexdigest()[:10]


# ------------------------------------------------------------------ plan
def plan(cfg: Config, samples: list[Sample], catalog: PriceCatalog | None = None) -> dict:
    """What a run would do and cost, before anything is sent."""
    catalog = catalog or PriceCatalog.load()
    models = [cfg.baseline] + list(cfg.candidates)
    have_ref = sum(1 for s in samples if s.reference) if not cfg.rerun_baseline else 0
    rows, problems, dest = [], [], {}
    total = upper = 0.0
    for i, m in enumerate(models):
        route = cfg.route(m)
        price = _price(catalog, m, route)
        calls = len(samples) - (have_ref if i == 0 else 0)
        if i == 0 and cfg.calibrate:
            calls += len(samples)
        if calls <= 0:
            rows.append({"model": m, "role": "baseline", "calls": 0, "note": "uses the production outputs"})
            continue
        if route.api not in ("bedrock", "litellm") and not route.key() and route.provider != "ollama" \
                and not (route.base and route.base.startswith("http://127.0.0.1")):
            problems.append(f"{m}: no API key ({' or '.join(route.key_env) or 'none needed'}) in the environment")
        if route.api == "litellm":
            problems.append(f"{m}: no built-in route; will use LiteLLM if installed")
        if price is None:
            problems.append(f"{m}: no price in the catalog, so the budget can't be enforced")
        est = up = None
        if price:
            est = sum(price.cost(math.ceil(s.prompt_chars() / 4), math.ceil(len(s.reference or s.expected or "x" * 400) / 4))
                      for s in samples[:calls])
            up = sum(_upper_cost(price, s.prompt_chars(), _out_cap(s, model=m)) for s in samples[:calls])
            est, up = est * calls / max(1, min(calls, len(samples))), up * calls / max(1, min(calls, len(samples)))
            total += est
            upper += up
        dest.setdefault(route.host, set()).add(m)
        rows.append({"model": m, "role": "baseline" if i == 0 else "candidate", "calls": calls,
                     "api": route.api, "endpoint": route.host,
                     "est_cost_usd": round(est, 4) if est is not None else None,
                     "max_cost_usd": round(up, 4) if up is not None else None})
    if cfg.judge:
        route = resolve(cfg.judge)
        price = _price(catalog, cfg.judge, route)
        calls = 2 * len(samples) * len(cfg.candidates)
        if price:
            est = sum(price.cost(math.ceil((s.prompt_chars() + 2 * len(s.reference or "x" * 400) + 600) / 4), 2)
                      for s in samples) * 2 * len(cfg.candidates)
            total += est
            upper += est * 1.6
        else:
            problems.append(f"{cfg.judge}: no price in the catalog")
        dest.setdefault(route.host, set()).add(cfg.judge)
        rows.append({"model": cfg.judge, "role": "judge", "calls": calls, "api": route.api, "endpoint": route.host})
    worst = 0.0
    called = [m for i, m in enumerate(models) if i > 0 or have_ref < len(samples) or cfg.calibrate]
    for m in called + ([cfg.judge] if cfg.judge else []):
        route = cfg.route(m)
        price = _price(catalog, m, route)
        if price:
            worst = max(worst, max(_upper_cost(price, s.prompt_chars(), _out_cap(s, model=m)) for s in samples))
    if worst > cfg.max_cost:
        problems.append(f"the budget (${cfg.max_cost:g}) is below the worst case of a single call (${worst:.4f}); "
                        "raise --max-cost or set max_tokens in the samples")
    blocking = []
    refs = [s.expected or s.reference or "" for s in samples]
    task = cfg.task or (gates.detect_task(refs, cfg.hints, json_mode=any(s.json_mode for s in samples))
                        if sum(1 for r in refs if r) >= 0.9 * len(samples) else None)
    if task == "generation" and not cfg.judge:
        blocking.append("the answers are free text: add --judge MODEL (e.g. gpt-4.1) to score them; "
                        "without a judge nothing can be validated")
    if len(samples) < 20:
        blocking.append(f"only {len(samples)} sample(s); at least 20 are needed for a verdict")
    problems.extend(blocking)
    return {"samples": len(samples), "task": task, "blocking": blocking,
            "with_reference": sum(1 for s in samples if s.reference),
            "with_expected": sum(1 for s in samples if s.expected), "models": rows,
            "est_cost_usd": round(total, 4), "max_cost_usd": round(upper, 4), "budget_usd": cfg.max_cost,
            "worst_single_call_usd": round(worst, 6),
            "data_goes_to": {h: sorted(ms) for h, ms in dest.items()}, "problems": problems}


# ------------------------------------------------------------------ run
def run(cfg: Config, samples: list[Sample], workdir: str | Path, catalog: PriceCatalog | None = None,
        progress=None, call_fn=call) -> dict:
    catalog = catalog or PriceCatalog.load()
    if not samples:
        raise ValueError("no samples to run")
    xid = experiment_id(cfg, samples)
    out = Path(workdir) / "experiments" / xid
    out.mkdir(parents=True, exist_ok=True)
    cache = CallCache(out / "calls.jsonl")
    budget = Budget(cfg.max_cost)
    routes = {m: cfg.route(m) for m in [cfg.baseline, *cfg.candidates] + ([cfg.judge] if cfg.judge else [])}
    prices = {m: _price(catalog, m, r) for m, r in routes.items()}
    unpriced = [m for m, p in prices.items() if p is None]
    if unpriced and not cfg.allow_unpriced:
        raise ValueError(f"no price for {', '.join(unpriced)}: the budget can't be enforced "
                         "(add the model to the catalog or pass --allow-unpriced)")

    results: dict[tuple[str, str], dict] = {}        # (model or purpose, sample id) -> entry
    stopped: set[str] = set()
    stop_reason: list[str] = []
    counters = {"new_calls": 0, "cached": 0, "skipped_budget": 0}
    lock = threading.Lock()

    def entry_cost(model, r: Result, s_chars: int, out_text: str | None) -> tuple[float, bool]:
        p = prices.get(model)
        if p is None:
            return 0.0, True
        estimated = r.input_tokens is None or r.output_tokens is None
        tin = r.input_tokens if r.input_tokens is not None else math.ceil(s_chars / 4)
        tout = r.output_tokens if r.output_tokens is not None else math.ceil(len(out_text or "") / 4)
        return p.cost(tin, tout, r.cached_input_tokens or 0), estimated

    inflight: dict[str, threading.Event] = {}

    def do_call(model: str, purpose: str, s: Sample, messages: list[dict], params: dict, out_cap: int):
        route = routes[model]
        # Judge calls are content-addressed: two candidates with the same answer need one verdict.
        key = _key(route, messages, params, "judge" if purpose.startswith("judge:") else purpose)
        with lock:
            hit = cache.get(key)
            waiter = inflight.get(key) if hit is None else None
            if hit is None and waiter is None:
                inflight[key] = threading.Event()
        if waiter is not None:                          # the same request is already on its way
            waiter.wait()
            hit = cache.get(key)
        if hit is not None:
            with lock:
                counters["cached"] += 1
                results[(purpose, s.id)] = hit
            return
        try:
            _send(model, purpose, s, messages, params, out_cap, route, key)
        finally:
            with lock:
                ev = inflight.pop(key, None)
            if ev is not None:
                ev.set()

    def _send(model, purpose, s, messages, params, out_cap, route, key):
        p = prices.get(model)
        chars = sum(len(m.get("content") or "") for m in messages)
        reserve = _upper_cost(p, chars, out_cap) if p else 0.0
        if not budget.reserve(reserve):
            with lock:
                counters["skipped_budget"] += 1
                if "budget" not in stop_reason:
                    stop_reason.append("budget")
            return
        r = call_fn(route, messages, params)
        cost, est = entry_cost(model, r, chars, r.text)
        budget.settle(reserve, cost)
        e = {"key": key, "model": model, "purpose": purpose, "sample": s.id, "text": r.text, "error": r.error,
             "input_tokens": r.input_tokens, "output_tokens": r.output_tokens, "cached_input_tokens": r.cached_input_tokens,
             "cost_usd": cost, "cost_estimated": est, "latency_ms": r.latency_ms, "attempts": r.attempts,
             "served_model": r.served_model, "ts": time.time()}
        cache.put(e)
        with lock:
            counters["new_calls"] += 1
            results[(purpose, s.id)] = e
        if progress:
            progress(purpose, s.id, e)

    def run_phase(jobs):
        """jobs: list of (model, purpose, sample, messages, params, out_cap, candidate_for_early_stop)."""
        with cf.ThreadPoolExecutor(max(1, cfg.concurrency)) as ex:
            inflight: set = set()
            for job in jobs:
                if "budget" in stop_reason:
                    break
                cand = job[6]
                if cand and cand in stopped:
                    continue
                while len(inflight) >= max(1, cfg.concurrency) * 2:
                    done, inflight = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
                    for d in done:
                        d.result()
                    _early_stop_check()
                inflight.add(ex.submit(do_call, *job[:6]))
            for d in cf.as_completed(inflight):
                d.result()

    # ---- references: production output (if any) unless the baseline is re-run
    ref_text: dict[str, str | None] = {}
    base_jobs = []
    for s in samples:
        if s.reference and not cfg.rerun_baseline:
            ref_text[s.id] = s.reference
        else:
            cap = _out_cap(s, model=cfg.baseline)
            base_jobs.append((cfg.baseline, "baseline", s, s.messages, _with_cap(s, cap), cap, None))
        if cfg.calibrate:
            cap = _out_cap(s, model=cfg.baseline)
            base_jobs.append((cfg.baseline, "baseline#2", s, s.messages, _with_cap(s, cap), cap, None))
    task_holder: dict = {}

    def _early_stop_check():
        if not cfg.early_stop or task_holder.get("task") in (None, "generation"):
            return
        for c in cfg.candidates:
            if c in stopped:
                continue
            ks = [gates.score(task_holder["task"], task_holder["refs"].get(s.id), results[(c, s.id)].get("text"))
                  for s in samples if (c, s.id) in results and task_holder["refs"].get(s.id) is not None]
            n = len(ks)
            if n >= 30:
                k = sum(1 for x in ks if x["pass"])
                if gates.wilson(k, n)[1] < task_holder["thr"]:
                    stopped.add(c)

    run_phase(base_jobs)
    for s in samples:
        if s.id not in ref_text:
            e = results.get(("baseline", s.id))
            ref_text[s.id] = e.get("text") if e and not e.get("error") else None

    # ---- task and threshold
    gold = {s.id: s.expected for s in samples if s.expected is not None}
    use_gold = len(gold) >= 0.9 * len(samples)
    task = cfg.task or gates.detect_task([ref_text.get(s.id) or s.expected or "" for s in samples], cfg.hints,
                                        json_mode=any(s.json_mode for s in samples))
    threshold = cfg.threshold if cfg.threshold is not None else gates.DEFAULT_THRESHOLD[task]
    refs = gold if use_gold else {k: v for k, v in ref_text.items() if v is not None}
    task_holder.update(task=task, refs=refs, thr=threshold)
    # Free text without a judge can't be validated: once the references show that, stop before
    # paying for any candidate call (the plan catches it earlier when production outputs exist).
    no_verdict = task == "generation" and not cfg.judge
    if no_verdict and not stop_reason:
        stop_reason.append("free-text answers need --judge to be scored; no candidate was called")

    # ---- candidates, round robin over samples
    cand_jobs = []
    for s in samples:
        for c in cfg.candidates:
            if refs.get(s.id) is None or no_verdict:
                continue
            cap = _out_cap(s, ref_text.get(s.id), c)
            cand_jobs.append((c, c, s, s.messages, _with_cap(s, cap), cap, c))
    run_phase(cand_jobs)
    _early_stop_check()

    # ---- judge (generation)
    judged = task == "generation" and cfg.judge
    if judged:
        jobs = []
        for s in samples:
            ref = refs.get(s.id)
            for c in cfg.candidates:
                e = results.get((c, s.id))
                if ref is None or not e or e.get("error") or not e.get("text"):
                    continue
                jobs.append((cfg.judge, f"judge:{c}:ab", s, gates.judge_messages(s.messages, ref, e["text"]),
                             {"max_tokens": 5, "temperature": 0}, 5, None))
                jobs.append((cfg.judge, f"judge:{c}:ba", s, gates.judge_messages(s.messages, e["text"], ref),
                             {"max_tokens": 5, "temperature": 0}, 5, None))
        run_phase(jobs)

    # ---- calibration: how often does the baseline agree with itself?
    calibration = None
    if cfg.calibrate and task != "generation":
        pairs = [(ref_text.get(s.id), (results.get(("baseline#2", s.id)) or {}).get("text")) for s in samples]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        if pairs:
            agree = sum(1 for a, b in pairs if gates.score(task, a, b)["pass"]) / len(pairs)
            calibration = {"baseline_self_agreement": round(agree, 4), "n": len(pairs)}
            if not use_gold:
                threshold = min(threshold, max(0.0, agree - 0.02))

    baseline_acc = None
    if use_gold:
        bs = [gates.score(task, gold[s.id], ref_text.get(s.id)) for s in samples if s.id in gold and ref_text.get(s.id)]
        if bs and task != "generation":
            baseline_acc = sum(1 for x in bs if x["pass"]) / len(bs)
            threshold = min(threshold, max(0.0, baseline_acc - 0.02))

    # ---- score
    def cost_of(model, purpose, s):
        e = results.get((purpose, s.id))
        return e["cost_usd"] if e and not e.get("error") else None

    base_costs = {}
    for s in samples:
        c = cost_of(cfg.baseline, "baseline", s)
        if c is None and prices.get(cfg.baseline) and ref_text.get(s.id):
            tin = s.meta.get("input_tokens") or math.ceil(s.prompt_chars() / 4)
            tout = s.meta.get("output_tokens") or math.ceil(len(ref_text[s.id]) / 4)
            c = prices[cfg.baseline].cost(tin, tout)
        if c is not None:
            base_costs[s.id] = c
    base_cost_measured = any(("baseline", s.id) in results for s in samples)

    per_sample = []
    cands = []
    for c in cfg.candidates:
        k = n = errors = 0
        field_scores, lat, costs, base_for_same = [], [], [], []
        judge_tally = {"not_worse": 0, "worse": 0, "unclear": 0}
        for s in samples:
            ref = refs.get(s.id)
            e = results.get((c, s.id))
            if ref is None or e is None:
                continue
            if e.get("error"):
                errors += 1
            sc = gates.score(task, ref, e.get("text"))
            if task == "generation":
                if not judged or sc["pass"] is False:
                    passed = sc["pass"]
                else:
                    ab = gates.judge_choice((results.get((f"judge:{c}:ab", s.id)) or {}).get("text"))
                    ba = gates.judge_choice((results.get((f"judge:{c}:ba", s.id)) or {}).get("text"))
                    if ab is None or ba is None:
                        judge_tally["unclear"] += 1
                        passed = None
                    else:
                        passed = ab in ("B", "TIE") and ba in ("A", "TIE")
                        judge_tally["not_worse" if passed else "worse"] += 1
                sc["pass"] = passed
            if sc["pass"] is None:
                per_sample.append({"sample": s.id, "model": c, **sc})
                continue
            n += 1
            k += 1 if sc["pass"] else 0
            if "field_score" in sc:
                field_scores.append(sc["field_score"])
            if not e.get("error"):
                lat.append(e.get("latency_ms") or 0)
                costs.append(e["cost_usd"])
                if s.id in base_costs:
                    base_for_same.append(base_costs[s.id])
            per_sample.append({"sample": s.id, "model": c, **sc})
        lo, hi = gates.wilson(k, n)
        if task == "generation" and not judged:
            v, need = "NEEDS-JUDGE", None
        elif n and errors / max(1, n) > 0.2:
            v, need = "ERROR", None
        else:
            v, need = gates.verdict(k, n, threshold)
        cpc = sum(costs) / len(costs) if costs else None
        bpc = sum(base_for_same) / len(base_for_same) if base_for_same else None
        sav = (1 - cpc / bpc) if cpc is not None and bpc else None
        lat_s = sorted(lat)
        cands.append({
            "model": c, "verdict": v, "pass_rate": round(k / n, 4) if n else None, "passed": k, "scored": n,
            "lower_95": round(lo, 4), "upper_95": round(hi, 4), "samples_needed": need, "errors": errors,
            "stopped_early": c in stopped,
            "field_agreement": round(sum(field_scores) / len(field_scores), 4) if field_scores else None,
            "judge": judge_tally if task == "generation" and judged else None,
            "cost_per_call_usd": cpc, "baseline_cost_per_call_usd": bpc,
            "savings_pct": round(sav, 4) if sav is not None else None,
            "monthly_savings_usd": round(cfg.monthly_spend * sav, 2) if cfg.monthly_spend and sav is not None else None,
            "latency_p50_ms": lat_s[len(lat_s) // 2] if lat_s else None,
            "latency_p95_ms": lat_s[min(len(lat_s) - 1, int(len(lat_s) * 0.95))] if lat_s else None,
        })
    for c in cands:
        c["costs_more"] = bool(c["savings_pct"] is not None and c["savings_pct"] <= 0)
    # Quality alone is not a saving: a model that passes but costs as much or more per call (hidden
    # reasoning tokens, longer answers) is reported, never recommended.
    passing = sorted([c for c in cands if c["verdict"] == "PASS" and not c["costs_more"]],
                     key=lambda c: c["cost_per_call_usd"] or 0)
    models = {}
    for m in {cfg.baseline, *cfg.candidates, *([cfg.judge] if cfg.judge else [])}:
        es = [e for (pur, _), e in results.items() if e["model"] == m]
        models[m] = {"calls": len(es), "errors": sum(1 for e in es if e.get("error")),
                     "cost_usd": round(sum(e["cost_usd"] for e in es), 6),
                     "input_tokens": sum(e.get("input_tokens") or 0 for e in es),
                     "output_tokens": sum(e.get("output_tokens") or 0 for e in es),
                     "endpoint": routes[m].host}
    result = {
        "id": xid, "name": cfg.name, "created": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "site": cfg.site, "site_id": cfg.site_id, "baseline": cfg.baseline, "candidates": cands,
        "task": task, "threshold": round(threshold, 4), "reference": "expected answers" if use_gold else (
            "production outputs" if any(s.reference for s in samples) and not cfg.rerun_baseline else "baseline model"),
        "samples": len(samples), "baseline_accuracy": round(baseline_acc, 4) if baseline_acc is not None else None,
        "calibration": calibration, "baseline_cost_measured": base_cost_measured,
        "recommended": passing[0]["model"] if passing else None,
        "spent_usd": round(budget.spent, 6), "budget_usd": cfg.max_cost,
        "spent_this_run_usd": round(budget.spent, 6),
        "stopped": stop_reason[0] if stop_reason else None, "calls": counters,
        "models": models, "data_sent_to": sorted({routes[m].host for m in models}),
        "monthly_spend_usd": cfg.monthly_spend, "config": {k: v for k, v in asdict(cfg).items()
                                                          if k not in ("monthly_spend",)},
        "per_sample": per_sample,
    }
    result["spent_usd"] = round(sum(e["cost_usd"] for e in cache.data.values()), 6)   # incl. earlier runs
    (out / "result.json").write_text(json.dumps(result, indent=1))
    _index(Path(workdir), result)
    return result


def _index(workdir: Path, result: dict) -> None:
    p = workdir / "experiments" / "index.json"
    idx = json.loads(p.read_text()) if p.exists() else []
    idx = [x for x in idx if x.get("id") != result["id"]]
    idx.append({k: result.get(k) for k in ("id", "name", "created", "site", "site_id", "baseline", "task",
                                            "recommended", "spent_usd", "samples")}
               | {"verdicts": {c["model"]: c["verdict"] for c in result["candidates"]}})
    p.write_text(json.dumps(idx, indent=1))


def load_results(workdir: str | Path) -> list[dict]:
    root = Path(workdir) / "experiments"
    out = []
    for p in sorted(root.glob("*/result.json")):
        try:
            out.append(json.loads(p.read_text()))
        except ValueError:
            continue
    return out

