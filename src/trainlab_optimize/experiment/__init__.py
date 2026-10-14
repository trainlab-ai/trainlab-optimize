"""Profile 3: prove a saving before making it.

    trainlab experiment plan                       # which call sites to test, with which models
    trainlab experiment run --site app/x.py:42 --samples .trainlab/samples --max-cost 5

Replays real inputs from one call site against the current model and cheaper candidates,
using your own API keys, scores every answer against quality gates, and records the result.
`trainlab report` then shows that finding as validated, with the measured saving.
"""
from __future__ import annotations

from ..analyzer.rules import family, load_candidates
from ..pricing import PriceCatalog
from .gates import detect_task, verdict, wilson
from .runner import Config, load_results, plan, run
from .samples import Sample, load_samples, make_sample, select

__all__ = ["Config", "Sample", "candidates_for", "detect_task", "eligible", "load_results", "load_samples",
           "make_sample", "plan", "run", "select", "verdict", "wilson"]


def candidates_for(site, catalog: PriceCatalog | None = None, limit: int = 3) -> list[str]:
    """Cheaper models to test for a call site: the curated economy models of its provider
    family that are priced, not deprecated and cheaper than the current model, cheapest last."""
    from ..analyzer.rules import site_price
    catalog = catalog or PriceCatalog.load()
    current = site_price(site, catalog)
    if not current:
        return []
    out = []
    for name in load_candidates().get(family(site), []):
        p = catalog.lookup(name, None)
        if not p or p.deprecated or p.blended_per_1k() >= current.blended_per_1k():
            continue
        out.append((p.blended_per_1k(), name))
    out.sort(reverse=True)                              # strongest (priciest) cheaper model first
    return [n for _, n in out[:limit]]


def eligible(analysis, catalog: PriceCatalog | None = None) -> list[dict]:
    """Call sites worth an experiment: model-switch findings first, then other chat sites with a
    known model and spend, by monthly spend."""
    catalog = catalog or PriceCatalog.load()
    sites = {s.id: s for s in analysis.scan.call_sites}
    spend = {a.call_site_id: a.monthly_spend_usd for a in analysis.attributions}
    out, seen = [], set()
    for f in analysis.findings:
        if f.rule != "R101":
            continue
        for sid in f.call_site_ids:
            s = sites.get(sid)
            if not s or sid in seen or s.kind == "embedding":
                continue
            seen.add(sid)
            out.append({"site_id": sid, "site": f"{s.path}:{s.line}", "model": s.model, "provider": s.provider,
                        "finding": f.id,
                        "candidates": candidates_for(s, catalog), "task_hints": list(s.task_hints),
                        "monthly_spend_usd": spend.get(sid), "reason": f.title})
    rest = sorted((s for s in analysis.scan.call_sites
                   if s.id not in seen and s.kind == "chat" and s.model and not s.is_test),
                  key=lambda s: (s.scope == "example", -(spend.get(s.id) or 0)))
    for s in rest:
        c = candidates_for(s, catalog)
        if c:
            out.append({"site_id": s.id, "site": f"{s.path}:{s.line}", "model": s.model, "provider": s.provider,
                        "finding": None,
                        "candidates": c, "task_hints": list(s.task_hints), "monthly_spend_usd": spend.get(s.id),
                        "reason": "chat call with a cheaper model available"})
    # App code before example/docs/notebook code, findings first within each.
    import re
    built = re.compile(r"(^|/)(_next/static|dist|build|out|vendor|node_modules)/|\.min\.js$|\.bundle\.js$")
    scope = {s.id: s.scope for s in analysis.scan.call_sites}
    out.sort(key=lambda r: (bool(built.search(r["site"].rpartition(":")[0])),       # generated bundles last
                            scope.get(r["site_id"]) == "example", r["finding"] is None))
    return out
