"""Run attribution + rules and produce the analysis result consumed by the report."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .. import SCHEMA_VERSION, __version__
from ..pricing import PriceCatalog
from ..scanner import ScanResult
from ..schema import Attribution, Finding, UsageRecord
from .attribution import attribute
from .rules import SITE_RULES, RuleContext, aggregate_findings, coverage_findings, load_candidates
from .spend import SpendSummary, summarize

SEV_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


@dataclass
class Analysis:
    scan: ScanResult
    spend: SpendSummary
    attributions: list[Attribution]
    coverage: dict
    findings: list[Finding]
    savings_low: float | None
    savings_high: float | None
    generated_at: str
    catalog_meta: dict = field(default_factory=dict)

    @property
    def validated_savings(self) -> float | None:
        v = [f.savings_high_usd for f in self.findings if f.tier == "validated" and f.savings_high_usd]
        return round(sum(v), 2) if v else None

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "tool_version": __version__,
            "generated_at": self.generated_at,
            "catalog": self.catalog_meta,
            "scan": self.scan.to_dict(),
            "spend": {
                "days": self.spend.days, "start": self.spend.start, "end": self.spend.end,
                "monthly_total_usd": round(self.spend.monthly_total, 2),
                "sources": self.spend.sources,
                "overlaps": self.spend.overlaps,
                "by_model": [
                    {"model": m.model, "provider": m.provider,
                     "monthly_cost_usd": round(m.monthly_cost, 2),
                     "monthly_requests": round(m.monthly_requests),
                     "monthly_input_tokens": round(m.monthly_input_tokens),
                     "monthly_output_tokens": round(m.monthly_output_tokens),
                     "cached_share": round(m.monthly_cached_tokens / m.monthly_input_tokens, 3)
                     if m.monthly_input_tokens else None,
                     "batch_share": round(m.batch_share, 3)}
                    for m in sorted(self.spend.by_model.values(), key=lambda x: -x.monthly_cost)
                ],
                "by_project": {k: round(v, 2) for k, v in
                               sorted(self.spend.by_project.items(), key=lambda kv: -kv[1])},
                "by_api_key": {k: {"monthly_cost_usd": round(v, 2),
                                   "models": sorted(self.spend.key_models.get(k, ()))}
                               for k, v in sorted(self.spend.by_api_key.items(), key=lambda kv: -kv[1])},
                "by_tag": {k: round(v, 2) for k, v in sorted(self.spend.by_tag.items(), key=lambda kv: -kv[1])},
            },
            "attribution": [a.to_dict() for a in self.attributions],
            "coverage": self.coverage,
            "findings": [f.to_dict() for f in self.findings],
            "savings": {"monthly_low_usd": self.savings_low, "monthly_high_usd": self.savings_high,
                        "validated_monthly_usd": self.validated_savings,
                        "tier": "estimated" if self.savings_high is not None else "static"},
        }


def _group(findings: list[Finding], sites: dict) -> list[Finding]:
    """Merge identical findings (same rule + title) in the same file into one finding."""
    merged: dict[tuple, Finding] = {}
    order: list[tuple] = []
    for f in findings:
        path = sites[f.call_site_ids[0]].path if f.call_site_ids else ""
        key = (f.rule, f.title, path)
        if key not in merged:
            merged[key] = f
            order.append(key)
            continue
        g = merged[key]
        g.call_site_ids = g.call_site_ids + [i for i in f.call_site_ids if i not in g.call_site_ids]
        for attr in ("monthly_spend_usd", "savings_low_usd", "savings_high_usd"):
            a, b = getattr(g, attr), getattr(f, attr)
            if a is not None or b is not None:
                setattr(g, attr, (a or 0.0) + (b or 0.0))
        g.severity = min(g.severity, f.severity, key=lambda s: SEV_ORDER.get(s, 9))
    return [merged[k] for k in order]


def analyze(scan: ScanResult, usage: list[UsageRecord], catalog: PriceCatalog | None = None,
            today: date | None = None, experiments: list[dict] | None = None) -> Analysis:
    catalog = catalog or PriceCatalog.load()
    today = today or date.today()
    spend = summarize(usage, catalog)
    attributions, coverage = attribute(scan.call_sites, spend)
    ctx = RuleContext(catalog=catalog, spend=spend,
                      attribution={a.call_site_id: a for a in attributions},
                      today=today, candidates=load_candidates())

    findings: list[Finding] = []
    for site in scan.call_sites:
        for rule in SITE_RULES:
            for f in rule(site, ctx):
                if site.scope == "example":
                    # Example/docs/notebook code: still reported, never above low severity.
                    f.severity = "low" if f.severity != "info" else f.severity
                    f.evidence = {**f.evidence, "scope": "example code"}
                findings.append(f)

    from .experiments import apply as apply_experiments
    findings = apply_experiments(findings, scan.call_sites, ctx.attribution, experiments or [])

    # Non-overlapping total, computed on per-site findings BEFORE grouping:
    # for each call site take only its single largest saving, never the sum of
    # overlapping recommendations (a cheaper model and caching on the same call overlap).
    low = high = None
    if spend.has_usage:
        best_high: dict[str, float] = {}
        best_low: dict[str, float] = {}
        for f in findings:
            if f.savings_high_usd is None:
                continue
            sid = f.call_site_ids[0]
            if f.savings_high_usd > best_high.get(sid, -1.0):
                best_high[sid] = f.savings_high_usd
                best_low[sid] = f.savings_low_usd or 0.0
        high, low = sum(best_high.values()), sum(best_low.values())

    findings = _group(findings, {s.id: s for s in scan.call_sites})
    findings.extend(aggregate_findings(scan.call_sites, ctx))
    findings.extend(coverage_findings(scan.call_sites, coverage, spend, ctx.attribution))

    findings.sort(key=lambda f: (SEV_ORDER.get(f.severity, 9), -(f.savings_high_usd or 0), f.rule))
    for i, f in enumerate(findings, 1):
        f.id = f"F-{i:03d}"

    from datetime import datetime, timezone
    return Analysis(scan=scan, spend=spend, attributions=attributions, coverage=coverage,
                    findings=findings, savings_low=low, savings_high=high,
                    generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    catalog_meta=catalog.meta)
