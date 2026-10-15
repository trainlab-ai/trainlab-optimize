"""Data behind the UI's four screens: Overview, Spend, Live and Experiments.

Everything is read from the workspace the CLI writes (.trainlab/), so the UI and the CLI can be
used side by side: `trainlab trace` in one terminal, the Live screen fills in as calls arrive.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from .analyzer.engine import analyze
from .pricing import PriceCatalog


# ------------------------------------------------------------------ analysis (Overview, Spend)
def analysis(ws):
    sc = ws.load_scan()
    if sc is None:
        return None
    from .experiment.runner import load_results
    return analyze(sc, ws.load_usage(), experiments=load_results(ws.path))


def _site_label(s) -> str:
    return f"{s.path}:{s.line}"


def overview(ws, jobs: list[dict]) -> dict:
    a = analysis(ws)
    if a is None:
        return {"ready": False}
    sites = {s.id: s for s in a.scan.call_sites}
    att = {x.call_site_id: x for x in a.attributions}
    steps, tested, untested, other = [], [], [], []
    for f in a.findings:
        if f.strategy == "coverage":
            continue
        site = sites.get(f.call_site_ids[0]) if f.call_site_ids else None
        row = {"id": f.id, "title": f.title.replace("`", ""), "summary": f.summary.replace("`", ""),
               "recommendation": f.recommendation.replace("`", ""), "tier": f.tier, "severity": f.severity,
               "rule": f.rule, "site": _site_label(site) if site else None,
               "function": site.function if site else None, "feature": None,
               "spend": f.monthly_spend_usd, "low": f.savings_low_usd, "high": f.savings_high_usd,
               "evidence": {k: v for k, v in (f.evidence or {}).items()
                            if k in ("tested_model", "pass_rate", "lower_95", "samples", "verdicts",
                                     "measured_cost_reduction", "experiment", "field_mismatch")}}
        if f.tier == "validated" and f.savings_high_usd:
            tested.append(row)
        elif f.tier == "validated":
            other.append({**row, "kind": "kept"})
        elif f.rule in ("R101", "R110") and f.savings_high_usd:
            untested.append(row)
        elif f.severity in ("high", "medium"):
            other.append({**row, "kind": "other"})
    tested.sort(key=lambda r: -(r["high"] or 0))
    untested.sort(key=lambda r: -(r["high"] or 0))
    running = [j for j in jobs if j["status"] == "running"]
    validated = sum(r["high"] for r in tested)
    traced = sum(x.monthly_spend_usd for x in a.attributions if x.method == "exact_trace")
    return {
        "ready": True,
        "has_usage": a.spend.has_usage, "monthly": round(a.spend.monthly_total, 2) if a.spend.has_usage else None,
        "days": a.spend.days, "period": [a.spend.start, a.spend.end], "sources": a.spend.sources,
        "traced_share": (traced / a.spend.monthly_total) if a.spend.has_usage and a.spend.monthly_total else 0,
        "validated": round(validated, 2), "estimated_low": a.savings_low, "estimated_high": a.savings_high,
        "tested": tested, "untested": untested, "other": other[:6], "running": running,
        "call_sites": len(a.scan.call_sites), "repos": a.scan.repos or [Path(a.scan.root).name],
        "by_tag": a.spend.by_tag, "generated": a.generated_at,
        "att": {k: {"spend": v.monthly_spend_usd, "requests": v.monthly_requests} for k, v in att.items()},
    }


def spend(ws) -> dict:
    a = analysis(ws)
    if a is None:
        return {"ready": False}
    att = {x.call_site_id: x for x in a.attributions}
    by_site: dict[str, list] = {}
    for f in a.findings:
        if f.strategy == "coverage":
            continue
        for sid in f.call_site_ids:
            by_site.setdefault(sid, []).append({"id": f.id, "title": f.title.replace("`", ""), "tier": f.tier,
                                                "high": f.savings_high_usd, "severity": f.severity})
    files: dict[str, dict] = {}
    for s in a.scan.call_sites:
        key = f"{s.repo}/{s.path}" if s.repo else s.path
        x = att.get(s.id)
        fe = files.setdefault(key, {"file": key, "repo": s.repo, "path": s.path, "spend": 0.0, "sites": []})
        sp = x.monthly_spend_usd if x else None
        fe["spend"] += sp or 0.0
        fe["sites"].append({"id": s.id, "line": s.line, "function": s.function, "model": s.model, "kind": s.kind,
                            "provider": s.provider, "spend": sp, "requests": x.monthly_requests if x else None,
                            "method": x.method if x else None, "findings": by_site.get(s.id, []),
                            "is_test": s.is_test, "scope": s.scope})
    out = sorted(files.values(), key=lambda f: (-f["spend"], f["file"]))
    for f in out:
        f["sites"].sort(key=lambda s: s["line"])
    return {"ready": True, "has_usage": a.spend.has_usage,
            "monthly": round(a.spend.monthly_total, 2) if a.spend.has_usage else None, "days": a.spend.days,
            "files": out, "by_tag": a.spend.by_tag,
            "by_model": [{"model": m.model, "spend": m.monthly_cost, "requests": m.monthly_requests}
                         for m in sorted(a.spend.by_model.values(), key=lambda x: -x.monthly_cost)]}


def source(ws, repo: str | None, path: str) -> dict:
    """Lines of a scanned file. Only files that hold a call site are served."""
    sc = ws.load_scan()
    if sc is None:
        raise ValueError("Scan your code first.")
    if not any(s.path == path and (s.repo or None) == (repo or None) for s in sc.call_sites):
        raise ValueError("Not a scanned file.")
    root = Path(sc.root)
    for cand in ([root / repo / path] if repo else []) + [root / path]:
        if cand.is_file():
            text = cand.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            if len(lines) > 5000:
                lines = lines[:5000]
            return {"path": path, "repo": repo, "lines": lines}
    raise ValueError(f"File not found on disk: {path} (moved since the scan?)")


# ------------------------------------------------------------------ Live: follow .trainlab/traces
class LiveFeed:
    """Reads new lines from the trace files on every poll (by remembered offsets) and keeps the
    last 24 hours of calls in memory."""

    def __init__(self, traces: Path, keep: int = 200_000):
        self.dir = traces
        self.offsets: dict[str, int] = {}
        self.events: deque = deque(maxlen=keep)
        self.lock = threading.Lock()
        self.catalog = PriceCatalog.load()
        self.prices: dict[str, object] = {}

    def _price(self, model: str, provider: str | None):
        k = f"{provider}|{model}"
        if k not in self.prices:
            self.prices[k] = self.catalog.lookup(model, provider if provider not in (None, "openai_compatible")
                                                 else None) or self.catalog.lookup(model)
        return self.prices[k]

    def poll(self) -> int:
        new = 0
        if not self.dir.is_dir():
            return 0
        with self.lock:
            for f in sorted(self.dir.glob("*.jsonl")):
                key = str(f)
                try:
                    size = f.stat().st_size
                except OSError:
                    continue
                pos = self.offsets.get(key, 0)
                if size < pos:                       # rotated or truncated: start over
                    pos = 0
                if size == pos:
                    continue
                with open(f, "rb") as fh:
                    fh.seek(pos)
                    chunk = fh.read(size - pos)
                end = chunk.rfind(b"\n")
                if end < 0:
                    continue                         # a partial line: wait for the rest
                for line in chunk[: end + 1].splitlines():
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    ts = _parse_ts(e.get("ts"))
                    if ts is None:
                        continue
                    p = self._price(e.get("model") or "", e.get("provider"))
                    cost = p.cost(e.get("input_tokens") or 0, e.get("output_tokens") or 0) if p else None
                    self.events.append({
                        "ts": ts, "site": f"{e.get('path')}:{e.get('line')}" if e.get("path") else "(unknown line)",
                        "func": e.get("func"), "model": e.get("model"), "in": e.get("input_tokens") or 0,
                        "out": e.get("output_tokens") or 0, "ms": e.get("ms"), "ok": e.get("ok", True),
                        "error": e.get("error_type") or e.get("error"), "cost": cost,
                        "w": e.get("weight") or 1.0, "tags": e.get("tags") or {}, "stream": e.get("stream")})
                    new += 1
                self.offsets[key] = pos + end + 1
        return new

    def snapshot(self, minutes: int = 60) -> dict:
        self.poll()
        with self.lock:
            evs = list(self.events)
        if not evs:
            return {"events": 0, "sites": [], "recent": [], "live": False}
        now = time.time()
        last = max(e["ts"] for e in evs)
        live = now - last < 120
        anchor = now if live else last                       # nothing new: show the last active hour
        start = anchor - minutes * 60
        window = [e for e in evs if e["ts"] > start]
        sites: dict[str, dict] = {}
        for e in window:
            s = sites.setdefault(e["site"], {"site": e["site"], "func": e["func"], "models": {}, "calls": 0,
                                             "errors": 0, "cost": 0.0, "ms": [], "per_min": [0] * minutes,
                                             "tags": {}})
            s["calls"] += e["w"]
            s["models"][e["model"]] = s["models"].get(e["model"], 0) + 1
            s["errors"] += 0 if e["ok"] else 1
            s["cost"] += (e["cost"] or 0.0) * e["w"]
            if e["ms"] is not None:
                s["ms"].append(e["ms"])
            i = min(minutes - 1, max(0, int((e["ts"] - start) // 60)))
            s["per_min"][i] += e["w"]
            for k, v in e["tags"].items():
                s["tags"].setdefault(k, str(v))
        out = []
        for s in sites.values():
            ms = sorted(s.pop("ms"))
            s["p50"] = ms[len(ms) // 2] if ms else None
            s["p95"] = ms[min(len(ms) - 1, int(len(ms) * 0.95))] if ms else None
            s["model"] = max(s["models"], key=s["models"].get) if s["models"] else None
            s["error_rate"] = s["errors"] / s["calls"] if s["calls"] else 0
            s["cost_per_hour"] = s["cost"] * 60 / minutes
            s.pop("models")
            out.append(s)
        out.sort(key=lambda s: -s["cost"])
        today0 = datetime.fromtimestamp(anchor).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        today = [e for e in evs if e["ts"] >= today0]
        last_min = [e for e in window if e["ts"] > anchor - 60]
        recent = sorted(window, key=lambda e: -e["ts"])[:40]
        return {
            "events": len(evs), "live": live, "anchor": anchor, "last_call": last, "minutes": minutes,
            "calls_per_min": sum(e["w"] for e in last_min),
            "calls_window": sum(e["w"] for e in window),
            "cost_per_hour": sum((e["cost"] or 0) * e["w"] for e in window) * 60 / minutes,
            "cost_today": sum((e["cost"] or 0) * e["w"] for e in today), "calls_today": sum(e["w"] for e in today),
            "errors_window": sum(1 for e in window if not e["ok"]),
            "sites": out,
            "recent": [{"ts": e["ts"], "site": e["site"], "model": e["model"], "in": e["in"], "out": e["out"],
                        "ms": e["ms"], "ok": e["ok"], "error": e["error"], "cost": e["cost"],
                        "feature": e["tags"].get("feature")} for e in recent],
        }


def _parse_ts(v):
    """Trace timestamps are ISO 8601 in UTC ("2026-10-15T07:05:30.444Z")."""
    if isinstance(v, (int, float)):
        return float(v)
    if not v:
        return None
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


# ------------------------------------------------------------------ Experiments
def _slug(site: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in site).strip("_")


def experiments(ws, jobs: list[dict]) -> dict:
    from .experiment import eligible
    from .experiment.runner import load_results
    results = sorted(load_results(ws.path), key=lambda x: x.get("created") or "", reverse=True)
    rows = []
    for x in results:
        rows.append({"id": x["id"], "site": x.get("site"), "baseline": x.get("baseline"), "task": x.get("task"),
                     "created": x.get("created"), "samples": x.get("samples"), "recommended": x.get("recommended"),
                     "spent": x.get("spent_usd"), "stopped": x.get("stopped"), "judge": (x.get("config") or {}).get("judge"),
                     "candidates": [{"model": c["model"], "verdict": c["verdict"], "passed": c.get("passed"),
                                     "scored": c.get("scored"), "savings_pct": c.get("savings_pct"),
                                     "costs_more": c.get("costs_more")} for c in x.get("candidates", [])]})
    sites = []
    a = analysis(ws)
    if a is not None:
        cap = ws.path / "samples"
        counts = {}
        if cap.is_dir():
            for f in cap.glob("*.jsonl"):
                try:
                    counts[f.stem] = sum(1 for _ in open(f, "rb"))
                except OSError:
                    pass
        latest = {}
        for x in results:
            latest.setdefault(x.get("site"), x["id"])
        for r in eligible(a, PriceCatalog.load()):
            n = counts.get(_slug(r["site"]), 0)
            sites.append({"site": r["site"], "model": r["model"], "candidates": r["candidates"],
                          "spend": r["monthly_spend_usd"], "samples": n, "finding": r["finding"],
                          "reason": r["reason"], "last": latest.get(r["site"])})
    return {"experiments": rows, "sites": sites, "jobs": jobs}


def experiment_detail(ws, xid: str) -> dict:
    p = ws.path / "experiments" / xid / "result.json"
    if not p.is_file() or "/" in xid or "\\" in xid or ".." in xid:
        raise ValueError(f"No experiment {xid}.")
    x = json.loads(p.read_text())
    x.pop("per_sample", None)                   # large; the examples carry what the screen shows
    return x


class Jobs:
    """Experiments started from the UI run in background threads, one at a time per site."""

    def __init__(self, ws):
        self.ws = ws
        self.items: dict[str, dict] = {}
        self.lock = threading.Lock()

    def list(self) -> list[dict]:
        with self.lock:
            return [dict(j) for j in sorted(self.items.values(), key=lambda j: -j["started"])][:20]

    def _config(self, body: dict):
        from types import SimpleNamespace
        from .experiment import eligible
        from .experiment.cli import _config
        a = analysis(self.ws)
        if a is None:
            raise ValueError("Scan your code first.")
        row = next((r for r in eligible(a, PriceCatalog.load()) if r["site"] == body.get("site")), None)
        if row is None:
            raise ValueError("Pick a call site from the list.")
        args = SimpleNamespace(
            site=row["site"], samples=None, baseline=None, provider=None,
            candidates=",".join(body.get("candidates") or []) or None, n=int(body.get("n") or 100),
            judge=body.get("judge") or None, task=None, threshold=None, input_column=None, expected_column=None,
            system=None, redact=False, ignore_fields=body.get("ignore_fields") or None,
            strict_fields=body.get("strict_fields") or None, max_cost=float(body.get("max_cost") or 1.0),
            concurrency=4, calibrate=False, rerun_baseline=False, name=None, allow_unpriced=False)
        from .experiment.cli import _samples
        samples, _ = _samples(args, self.ws, row)
        if not samples:
            raise ValueError("No captured samples for this call site. Record some with "
                             "`trainlab trace --capture -- <your app>`.")
        return _config(args, row), samples

    def plan(self, body: dict) -> dict:
        from .experiment.runner import plan
        cfg, samples = self._config(body)
        p = plan(cfg, samples, PriceCatalog.load())
        base_host = cfg.route(cfg.baseline).host
        p["new_hosts"] = [h for h in p["data_goes_to"] if h != base_host]
        p["baseline"], p["candidates"], p["site"] = cfg.baseline, cfg.candidates, cfg.site
        return p

    def start(self, body: dict) -> dict:
        from .experiment.runner import plan, run
        cfg, samples = self._config(body)
        catalog = PriceCatalog.load()
        p = plan(cfg, samples, catalog)
        hard = [x for x in p["problems"] if "no API key" in x or "no price" in x] + list(p.get("blocking") or [])
        if hard:
            raise ValueError("; ".join(hard))
        base_host = cfg.route(cfg.baseline).host
        if [h for h in p["data_goes_to"] if h != base_host] and not body.get("confirm_hosts"):
            raise ValueError("Confirm that samples may be sent to a provider that doesn't serve this call today.")
        with self.lock:
            if any(j["site"] == cfg.site and j["status"] == "running" for j in self.items.values()):
                raise ValueError("An experiment is already running for this call site.")
            jid = uuid.uuid4().hex[:8]
            total = sum(m.get("calls") or 0 for m in p["models"])
            job = {"id": jid, "site": cfg.site, "status": "running", "calls": 0, "total": total, "started": time.time(),
                   "spent": 0.0, "result": None, "error": None, "budget": cfg.max_cost}
            self.items[jid] = job

        def progress(purpose, sid, e):
            with self.lock:
                job["calls"] += 1
                job["spent"] += e.get("cost_usd") or 0.0

        def work():
            try:
                res = run(cfg, samples, self.ws.path, catalog, progress=progress)
                with self.lock:
                    job.update(status="done", result=res["id"], stopped=res.get("stopped"),
                               spent=res.get("spent_this_run_usd"), recommended=res.get("recommended"))
            except Exception as exc:  # noqa: BLE001 - shown on the screen
                with self.lock:
                    job.update(status="error", error=f"{type(exc).__name__}: {exc}")
        threading.Thread(target=work, daemon=True).start()
        return dict(job)
