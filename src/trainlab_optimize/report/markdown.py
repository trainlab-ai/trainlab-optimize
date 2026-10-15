"""Markdown summary: for pull-request comments, CI step summaries and chat."""
from __future__ import annotations

from .. import __version__
from ..analyzer.engine import Analysis


def _money(v) -> str:
    return "—" if v is None else f"${v:,.0f}"


def render_markdown(a: Analysis, max_findings: int = 15) -> str:
    sites = {s.id: s for s in a.scan.call_sites}
    name = ", ".join(a.scan.repos) if a.scan.repos else a.scan.root.rstrip("/").split("/")[-1]
    findings = [f for f in a.findings if f.strategy != "coverage"]
    out = [f"## LLM cost report: {name}", ""]
    if a.spend.has_usage:
        out.append(f"**Monthly LLM spend:** {_money(a.spend.monthly_total)} ({a.spend.days} day{'s' if a.spend.days != 1 else ''} of data)  ")
        for o in a.spend.overlaps:
            out.append(f"_Overlap: {o}._  ")
        if a.savings_high:
            out.append(f"**Potential saving:** {_money(a.savings_low)} – {_money(a.savings_high)}/month "
                       "(estimated, not yet tested)  ")
        if a.validated_savings:
            out.append(f"**Validated by experiment:** {_money(a.validated_savings)}/month  ")
        share = a.coverage.get("attributed_share")
        if share is not None:
            out.append(f"**Spend traced to code:** {share:.0%}  ")
        if a.spend.by_api_key:
            top = sorted(a.spend.by_api_key.items(), key=lambda kv: -kv[1])[:5]
            out.append("**By API key:** " + ", ".join(f"`{k}` {_money(v)}" for k, v in top)
                       + (f" (+{len(a.spend.by_api_key) - 5} more)" if len(a.spend.by_api_key) > 5 else ""))
        if a.spend.by_tag:
            top = sorted(a.spend.by_tag.items(), key=lambda kv: -kv[1])[:6]
            out.append("**By tag:** " + ", ".join(f"`{k}` {_money(v)}" for k, v in top))
    else:
        out.append("_No usage imported: findings come from code only and have no cost attached._")
    out += ["", f"{len(a.scan.call_sites)} LLM call sites in {a.scan.files_scanned} files, "
                f"{len(findings)} findings.", ""]
    if findings:
        out += ["| | Finding | Where | Saving/month |", "|---|---|---|---|"]
        icon = {"high": "🔴", "medium": "🟠", "low": "⚪", "info": "🔵"}
        for f in findings[:max_findings]:
            where = ", ".join(f"`{sites[i].location}`" for i in f.call_site_ids[:2] if i in sites)
            if len(f.call_site_ids) > 2:
                where += f" +{len(f.call_site_ids) - 2}"
            save = (f"{_money(f.savings_low_usd or 0)}–{_money(f.savings_high_usd)}"
                    if f.savings_high_usd else "")
            title = f.title.replace("|", "\\|") + (" ✅" if f.tier == "validated" and f.savings_high_usd else "")
            out.append(f"| {icon.get(f.severity, '')} {f.severity} | {title} | {where} | {save} |")
        if len(findings) > max_findings:
            out.append(f"\n_…and {len(findings) - max_findings} more in the full report._")
    gaps = [f for f in a.findings if f.strategy == "coverage"]
    if gaps:
        out += ["", "**Gaps:** " + "; ".join(f.title for f in gaps)]
    out += ["", f"<sub>trainlab-optimize {__version__}, generated locally. "
                "No code or usage data left the machine.</sub>", ""]
    return "\n".join(out)
