"""Attribute measured spend to call sites.

v0.1 correlates by model name: spend for a model is assigned to the call sites that use
that model (exactly when there is one site, evenly split otherwise). Runtime span tagging
(profile 2, trainlab.instrument) replaces this with exact per-site attribution.
"""
from __future__ import annotations

from ..schema import Attribution, CallSite
from .spend import SpendSummary, base_model

COST_KINDS = {"chat", "embedding", "batch", "model_config"}


def attribute(sites: list[CallSite], spend: SpendSummary) -> tuple[list[Attribution], dict]:
    attributions: list[Attribution] = []
    by_base: dict[str, list[CallSite]] = {}
    for s in sites:
        if s.kind in COST_KINDS and s.model:
            by_base.setdefault(base_model(s.model), []).append(s)

    unattributed: dict[str, float] = {}
    for b, ms in spend.by_model.items():
        matched = by_base.get(b, [])
        # Prefer real call sites over framework config objects when both exist.
        calls = [s for s in matched if s.kind != "model_config"] or matched
        if not calls:
            unattributed[ms.model] = ms.monthly_cost
            continue
        share = 1.0 / len(calls)
        method = "exact_model" if len(calls) == 1 else "split_model"
        for s in calls:
            attributions.append(Attribution(
                call_site_id=s.id, model=ms.model, monthly_spend_usd=ms.monthly_cost * share,
                monthly_requests=ms.monthly_requests * share, method=method))

    attributed_total = sum(a.monthly_spend_usd for a in attributions)
    coverage = {
        "attributed_usd": attributed_total,
        "unattributed_usd": sum(unattributed.values()),
        "unattributed_by_model": dict(sorted(unattributed.items(), key=lambda kv: -kv[1])),
        "attributed_share": (attributed_total / spend.monthly_total) if spend.monthly_total else None,
    }
    return attributions, coverage
