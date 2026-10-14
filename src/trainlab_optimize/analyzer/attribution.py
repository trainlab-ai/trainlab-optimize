"""Attribute measured spend to call sites.

v0.1 correlates by model name: spend for a model is assigned to the call sites that use
that model (exactly when there is one site, evenly split otherwise). Runtime span tagging
(profile 2, trainlab.instrument) replaces this with exact per-site attribution.
"""
from __future__ import annotations

from ..schema import Attribution, CallSite
from .spend import SpendSummary, base_model

COST_KINDS = {"chat", "embedding", "batch", "model_config"}


def _path_matches(trace_path: str, site_path: str) -> bool:
    """Traces usually hold absolute (container) paths; sites hold repo-relative paths."""
    t = trace_path.replace("\\", "/")
    return t == site_path or t.endswith("/" + site_path)


def attribute(sites: list[CallSite], spend: SpendSummary) -> tuple[list[Attribution], dict]:
    attributions: list[Attribution] = []
    by_base: dict[str, list[CallSite]] = {}
    for s in sites:
        if s.kind in COST_KINDS and s.model:
            by_base.setdefault(base_model(s.model), []).append(s)

    # 1. Exact: trace records that carry a source location.
    located: dict[str, float] = {}            # base model -> spend matched to a call site
    exact: dict[str, list] = {}               # site id -> [cost, requests, model]
    for (path, line, b), (cost, req) in spend.by_location.items():
        best = None
        for s in sites:
            if s.kind not in COST_KINDS or s.cell is not None or not _path_matches(path, s.path):
                continue
            dist = abs((line or s.line) - s.line)
            if dist <= 3 and (best is None or dist < best[0]):
                best = (dist, s)
        if best is None:
            continue
        s = best[1]
        e = exact.setdefault(s.id, [0.0, 0.0, spend.by_model[b].model if b in spend.by_model else b])
        e[0] += cost
        e[1] += req
        located[b] = located.get(b, 0.0) + cost
    for sid, (cost, req, model) in exact.items():
        attributions.append(Attribution(call_site_id=sid, model=model, monthly_spend_usd=cost,
                                        monthly_requests=req, method="exact_trace"))

    # 2. By model name for the remaining spend.
    unattributed: dict[str, float] = {}
    for b, ms in spend.by_model.items():
        remaining = ms.monthly_cost - located.get(b, 0.0)
        if remaining <= 1e-9:
            continue
        share_req = ms.monthly_requests * (remaining / ms.monthly_cost) if ms.monthly_cost else 0.0
        matched = [s for s in by_base.get(b, []) if s.id not in exact]
        # Prefer real call sites over framework config objects when both exist.
        calls = [s for s in matched if s.kind != "model_config"] or matched
        if not calls:
            if not located.get(b):
                unattributed[ms.model] = remaining
            else:
                unattributed[ms.model] = remaining
            continue
        share = 1.0 / len(calls)
        method = "exact_model" if len(calls) == 1 else "split_model"
        for s in calls:
            attributions.append(Attribution(
                call_site_id=s.id, model=ms.model, monthly_spend_usd=remaining * share,
                monthly_requests=share_req * share, method=method))

    attributed_total = sum(a.monthly_spend_usd for a in attributions)
    coverage = {
        "attributed_usd": attributed_total,
        "unattributed_usd": sum(unattributed.values()),
        "unattributed_by_model": dict(sorted(unattributed.items(), key=lambda kv: -kv[1])),
        "attributed_share": (attributed_total / spend.monthly_total) if spend.monthly_total else None,
        "exact_share": (sum(e[0] for e in exact.values()) / spend.monthly_total) if spend.monthly_total else None,
    }
    return attributions, coverage
