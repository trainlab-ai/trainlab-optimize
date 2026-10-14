"""Self-contained HTML report. No external fonts, scripts or requests: safe to open offline."""
from __future__ import annotations

import html
import re

from .. import __version__
from ..analyzer.engine import Analysis

SEGMENT_COLORS = ["#24476B", "#3C6E91", "#5C8FB0", "#86AFC8", "#B4CEDD", "#7A8796", "#A3ADB8"]


def e(s) -> str:
    return html.escape("" if s is None else str(s))


def md(s: str) -> str:
    """Escape, then render `code` spans."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", e(s))


def money(v, dash="—") -> str:
    if v is None:
        return dash
    if abs(v) >= 100:
        return f"${v:,.0f}"
    return f"${v:,.2f}"


def num(v) -> str:
    if v is None:
        return "—"
    v = float(v)
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(v) >= div:
            return f"{v / div:.1f}{unit}"
    return f"{v:,.0f}"


CSS = """
:root{--paper:#F7F9FB;--sheet:#FFFFFF;--ink:#13233A;--muted:#5B6B7F;--rule:#D7DEE6;--soft:#EEF2F6;
--save:#0F7B6C;--save-soft:#DDF1EC;--high:#B42318;--med:#B45309;--low:#5B6B7F;--info:#3C6E91;}
@media (prefers-color-scheme:dark){:root{--paper:#0E1621;--sheet:#132030;--ink:#E6EDF5;--muted:#9AA9BA;
--rule:#2A3A4E;--soft:#1A2A3C;--save:#4CC2AE;--save-soft:#163A36;--high:#F97066;--med:#F5A524;--low:#9AA9BA;--info:#7FB2D6;}}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
font-variant-numeric:tabular-nums}
.wrap{max-width:1000px;margin:0 auto;padding:40px 24px 64px}
h1,h2,h3{font-family:Charter,"Bitstream Charter","Sitka Text",Cambria,Georgia,serif;font-weight:600;letter-spacing:-.01em}
h1{font-size:34px;line-height:1.15;margin:0 0 6px}
h2{font-size:22px;margin:48px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--rule)}
h3{font-size:17px;margin:0}
p{margin:0 0 10px;max-width:72ch}
.sub{color:var(--muted);margin-bottom:28px}
code{font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--soft);padding:1px 5px;border-radius:4px}
.bill{background:var(--sheet);border:1px solid var(--rule);border-radius:10px;padding:24px 24px 18px}
.bill-row{display:grid;grid-template-columns:150px 1fr 120px;gap:14px;align-items:center;margin:10px 0}
.bill-row .label{color:var(--muted);font-size:14px}
.bill-row .amt{text-align:right;font-size:20px;font-weight:600}
.bar{display:flex;height:30px;border-radius:5px;overflow:hidden;background:var(--soft)}
.bar span{display:block;height:100%}
.bar .cut{background:repeating-linear-gradient(135deg,var(--save) 0 2px,var(--save-soft) 2px 8px)}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:14px;font-size:13px;color:var(--muted)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:18px}
.fact{border-top:2px solid var(--ink);padding-top:8px}
.fact b{display:block;font-size:22px}
.fact span{color:var(--muted);font-size:13px}
.save b{color:var(--save)}
.note{font-size:13px;color:var(--muted)}
.finding{display:grid;grid-template-columns:4px 1fr;gap:16px;padding:18px 0;border-bottom:1px solid var(--rule)}
.finding .sev{border-radius:2px}
.sev-high{background:var(--high)}.sev-medium{background:var(--med)}.sev-low{background:var(--low)}.sev-info{background:var(--info)}
.fhead{display:flex;justify-content:space-between;gap:16px;align-items:baseline}
.fsave{white-space:nowrap;color:var(--save);font-weight:600}
.meta{font-size:13px;color:var(--muted);margin:4px 0 8px}
.tag{display:inline-block;font-size:12px;border:1px solid var(--rule);border-radius:999px;padding:0 8px;margin-right:6px;color:var(--muted)}
.tag.tl{border-color:var(--save);color:var(--save)}
.rec{border-left:2px solid var(--save);padding-left:10px;margin-top:6px}
details{margin-top:8px;font-size:13px;color:var(--muted)}
summary{cursor:pointer}
a{color:var(--info)}
summary:focus-visible,a:focus-visible{outline:2px solid var(--info);outline-offset:2px}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--rule);vertical-align:top}
th{color:var(--muted);font-weight:500}
td.r,th.r{text-align:right}
.tiers{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;font-size:13px}
.tiers div{background:var(--sheet);border:1px solid var(--rule);border-radius:8px;padding:12px}
.tiers b{display:block;font-size:14px;margin-bottom:2px}
.tiers .on{border-color:var(--ink)}
.empty{background:var(--sheet);border:1px dashed var(--rule);border-radius:8px;padding:16px}
footer{margin-top:56px;font-size:12px;color:var(--muted)}
@media print{body{background:#fff;color:#000;font-size:11pt}.wrap{max-width:none;padding:0}
.bill,.tiers div,.empty{border-color:#bbb}details{display:block}details>summary{list-style:none}
details[open] summary~*,details summary~*{display:block}.finding{break-inside:avoid}h2{break-after:avoid}
a{color:#000;text-decoration:none}}
@media (max-width:640px){.bill-row{grid-template-columns:1fr;gap:4px}.bill-row .amt{text-align:left}
.fhead{flex-direction:column}}
"""


def _bill(a: Analysis, sites: dict) -> str:
    total = a.spend.monthly_total
    if not a.spend.has_usage:
        return (
            '<div class="empty"><h3>No usage imported yet</h3><p>This report covers code only, so savings '
            "can't be priced. Import billing data to see spend per call site:</p>"
            "<p><code>trainlab import openai</code> &nbsp; <code>trainlab import anthropic</code> &nbsp; "
            "<code>trainlab import csv billing.csv</code></p></div>"
        )
    # Segments: attributed spend per file:function, plus unattributed.
    groups: dict[str, float] = {}
    for at in a.attributions:
        s = sites.get(at.call_site_id)
        key = (f"{s.repo}/{s.path}" if s.repo else s.path) if s else at.model
        groups[key] = groups.get(key, 0.0) + at.monthly_spend_usd
    if a.coverage.get("unattributed_usd"):
        groups["Not found in this repo"] = a.coverage["unattributed_usd"]
    ordered = sorted(groups.items(), key=lambda kv: -kv[1])
    if len(ordered) > 6:
        rest = sum(v for _, v in ordered[5:])
        ordered = ordered[:5] + [("Other", rest)]
    segs = "".join(
        f'<span style="width:{100 * v / total:.2f}%;background:{SEGMENT_COLORS[i % len(SEGMENT_COLORS)]}" '
        f'title="{e(k)}: {money(v)}"></span>' for i, (k, v) in enumerate(ordered) if total)
    legend = "".join(
        f'<span><i style="background:{SEGMENT_COLORS[i % len(SEGMENT_COLORS)]}"></i>{e(k)}: {money(v)}</span>'
        for i, (k, v) in enumerate(ordered))
    high = a.savings_high or 0.0
    low = a.savings_low or 0.0
    after = max(total - high, 0.0)
    after_bar = (f'<span style="width:{100 * after / total:.2f}%;background:var(--ink)"></span>'
                 f'<span class="cut" style="width:{100 * high / total:.2f}%" title="Potential saving"></span>')
    return f"""
<div class="bill" role="img" aria-label="Monthly LLM spend {money(total)}; potential saving up to {money(high)}">
  <div class="bill-row"><div class="label">Your bill today</div><div class="bar">{segs}</div>
    <div class="amt">{money(total)}<span class="note">/mo</span></div></div>
  <div class="bill-row"><div class="label">If the findings hold</div><div class="bar">{after_bar}</div>
    <div class="amt">{money(after)}<span class="note">/mo</span></div></div>
  <div class="legend">{legend}<span><i style="background:repeating-linear-gradient(135deg,var(--save) 0 2px,var(--save-soft) 2px 6px)"></i>Potential saving</span></div>
  <div class="facts">
    <div class="fact"><b>{money(total)}</b><span>Measured monthly run-rate ({a.spend.days} days of data)</span></div>
    <div class="fact save"><b>{money(low)} – {money(high)}</b><span>Potential monthly saving, estimated, not yet tested</span></div>
    <div class="fact"><b>{(a.coverage.get('attributed_share') or 0):.0%}</b><span>Of spend traced to a line of code in this repo</span></div>
  </div>
</div>"""


def _finding(f, sites: dict) -> str:
    locs = []
    for sid in f.call_site_ids[:6]:
        s = sites.get(sid)
        if s:
            fn = f" in <code>{e(s.function)}()</code>" if s.function else ""
            locs.append(f"<code>{e(s.location)}</code>{fn}")
    more = f" and {len(f.call_site_ids) - 6} more" if len(f.call_site_ids) > 6 else ""
    loc = (", ".join(locs) + more) if locs else ""
    save = ""
    if f.savings_high_usd:
        if f.savings_low_usd:
            save = f'<span class="fsave">save {money(f.savings_low_usd)}–{money(f.savings_high_usd)}/mo</span>'
        else:
            save = f'<span class="fsave">save up to {money(f.savings_high_usd)}/mo</span>'
    elif f.monthly_spend_usd:
        save = f'<span class="note">{money(f.monthly_spend_usd)}/mo affected</span>'
    tags = f'<span class="tag">{e(f.severity)}</span><span class="tag">{e(f.strategy)}</span>'
    tags += f'<span class="tag">{"estimated" if f.tier == "estimated" else "from code only"}</span>'
    if f.evidence.get("scope") == "example code":
        tags += '<span class="tag">example code</span>'
    if f.trainlab_candidate:
        tags += '<span class="tag tl">fine-tuned model candidate</span>'
    ev = ""
    if f.evidence:
        rows = "".join(f"<tr><th>{e(k.replace('_', ' '))}</th><td>{e(v)}</td></tr>" for k, v in f.evidence.items())
        ev = f"<details><summary>Evidence</summary><table>{rows}</table></details>"
    return f"""
<div class="finding" id="{e(f.id)}">
  <div class="sev sev-{e(f.severity)}" aria-hidden="true"></div>
  <div>
    <div class="fhead"><h3>{md(f.title)}</h3>{save}</div>
    <div class="meta">{e(f.id)} &nbsp; {loc}</div>
    <div>{tags}</div>
    <p style="margin-top:8px">{md(f.summary)}</p>
    <p class="rec">{md(f.recommendation)}</p>
    {ev}
  </div>
</div>"""


def render_html(a: Analysis) -> str:
    sites = {s.id: s for s in a.scan.call_sites}
    attr = {at.call_site_id: at for at in a.attributions}
    root = ", ".join(a.scan.repos) if a.scan.repos else a.scan.root.rstrip("/").split("/")[-1]
    period = f"{a.spend.start} to {a.spend.end}" if a.spend.start else "code only"
    findings = [f for f in a.findings if f.strategy != "coverage"]
    coverage = [f for f in a.findings if f.strategy == "coverage"]
    slm = [f for f in findings if f.trainlab_candidate]

    out = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM cost report: {e(root)}</title><style>{CSS}</style></head><body><main class="wrap">
<h1>LLM cost report for {e(root)}</h1>
<p class="sub">Found {len(a.scan.call_sites)} LLM call sites in {a.scan.files_scanned} files. Usage period: {e(period)}. Generated {e(a.generated_at)}.</p>
{_bill(a, sites)}
"""]

    out.append(f"<h2>What to change ({len(findings)})</h2>")
    if findings:
        out.append("<p class=\"note\">Sorted by severity, then by potential saving. Savings on the same line of "
                   "code overlap, so the headline total counts only the largest one per call site.</p>")
        out.extend(_finding(f, sites) for f in findings)
    else:
        out.append('<div class="empty">No waste patterns found in the scanned code.</div>')

    if slm:
        total = sum(f.monthly_spend_usd or 0 for f in slm)
        out.append(f"""<h2>Candidates for a fine-tuned model</h2>
<p>{len(slm)} workload(s) are narrow and repetitive (extraction or classification){f', currently costing {money(total)}/month' if total else ''}.
A small model fine-tuned on your own examples typically handles these at a fraction of frontier-model cost,
with predictable fixed pricing. First validate a cheaper off-the-shelf model with <code>trainlab experiment</code>;
if quality falls short, a custom model is the next step.</p>
<ul>{''.join(f'<li><a href="#{e(f.id)}">{e(f.id)}</a>: {md(f.title)}</li>' for f in slm)}</ul>""")

    if a.spend.has_usage:
        rows = []
        for m in sorted(a.spend.by_model.values(), key=lambda x: -x.monthly_cost):
            share = (m.monthly_cost / a.spend.monthly_total) if a.spend.monthly_total else 0
            cached = (m.monthly_cached_tokens / m.monthly_input_tokens) if m.monthly_input_tokens else 0
            rows.append(f"<tr><td><code>{e(m.model)}</code></td><td>{e(m.provider)}</td>"
                        f"<td class=r>{money(m.monthly_cost)}</td><td class=r>{share:.0%}</td>"
                        f"<td class=r>{num(m.monthly_requests) if m.monthly_requests else '—'}</td>"
                        f"<td class=r>{num(m.monthly_input_tokens)} / {num(m.monthly_output_tokens)}</td>"
                        f"<td class=r>{cached:.0%}</td></tr>")
        out.append(f"""<h2>Spend by model</h2>
<table><thead><tr><th>Model</th><th>Provider</th><th class=r>Per month</th><th class=r>Share</th>
<th class=r>Requests</th><th class=r>Tokens in / out</th><th class=r>Cached input</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>""")
        if a.spend.by_project:
            prow = "".join(f"<tr><td><code>{e(k)}</code></td><td class=r>{money(v)}</td></tr>"
                           for k, v in list(a.spend.by_project.items())[:15])
            out.append(f"<h2>Spend by project</h2><table><thead><tr><th>Project / workspace</th>"
                       f"<th class=r>Per month</th></tr></thead><tbody>{prow}</tbody></table>")

    if a.scan.repos and a.spend.has_usage:
        by_repo: dict[str, float] = {}
        for at in a.attributions:
            s = sites.get(at.call_site_id)
            if s:
                by_repo[s.repo or "?"] = by_repo.get(s.repo or "?", 0.0) + at.monthly_spend_usd
        rrows = "".join(f"<tr><td><code>{e(k)}</code></td><td class=r>{money(v)}</td></tr>"
                        for k, v in sorted(by_repo.items(), key=lambda kv: -kv[1]))
        if a.coverage.get("unattributed_usd"):
            rrows += f"<tr><td>Not found in these repositories</td><td class=r>{money(a.coverage['unattributed_usd'])}</td></tr>"
        out.append(f"<h2>Spend by repository</h2><table><thead><tr><th>Repository</th>"
                   f"<th class=r>Per month</th></tr></thead><tbody>{rrows}</tbody></table>")

    srows = []
    fids: dict[str, list[str]] = {}
    for f in a.findings:
        for sid in f.call_site_ids:
            fids.setdefault(sid, []).append(f.id)
    for s in sorted(a.scan.call_sites, key=lambda s: -(attr[s.id].monthly_spend_usd if s.id in attr else 0)):
        at = attr.get(s.id)
        method = {"exact_model": "", "split_model": " (shared)", "exact_trace": " (traced)"}.get(at.method, "") if at else ""
        model = f"<code>{e(s.model)}</code>" if s.model else f'<span class="note">{e(s.model_expr or "not set")}</span>'
        links = " ".join(f'<a href="#{i}">{i}</a>' for i in fids.get(s.id, []))
        srows.append(f"<tr><td><code>{e(s.location)}</code>{'<br><span class=note>' + e(s.function) + '()</span>' if s.function else ''}</td>"
                     f"<td>{e(s.kind)}</td><td>{model}</td>"
                     f"<td class=r>{money(at.monthly_spend_usd) + method if at else '—'}</td><td>{links}</td></tr>")
    out.append(f"""<h2>Every LLM call site</h2>
<table><thead><tr><th>Location</th><th>Type</th><th>Model</th><th class=r>Spend / month</th><th>Findings</th></tr></thead>
<tbody>{''.join(srows) or '<tr><td colspan=5>No LLM calls found.</td></tr>'}</tbody></table>""")

    if a.scan.model_refs:
        mrows = "".join(f"<tr><td><code>{e(r['path'])}:{r['line']}</code></td><td><code>{e(r['model'])}</code></td></tr>"
                        for r in a.scan.model_refs[:40])
        out.append(f"<h2>Model names in configuration</h2><p class=note>Model names found outside call sites, "
                   f"e.g. in config files. They may be selected at runtime.</p><table><tbody>{mrows}</tbody></table>")

    out.append("<h2>How to read this report</h2>")
    tier = "estimated" if a.spend.has_usage else "static"
    out.append(f"""<div class="tiers">
<div class="{'on' if tier == 'static' else ''}"><b>From code</b>Pattern found in source. No cost attached.</div>
<div class="{'on' if tier == 'estimated' else ''}"><b>Estimated</b>Measured spend × price difference. Quality not yet tested.</div>
<div><b>Validated</b>Tested on your real inputs with <code>trainlab experiment</code>.</div>
<div><b>Realized</b>Confirmed in production billing after the change.</div></div>
<p class="note" style="margin-top:12px">Spend is linked to code by model name: when several call sites use the same model,
its spend is split evenly between them (marked "shared"). Imported traces that carry a source location
(<code>code.file.path</code>, <code>code.line.number</code>) are attributed to that exact line (marked "traced").</p>""")
    if coverage:
        out.append("<h2>Gaps in this analysis</h2>")
        out.extend(_finding(f, sites) for f in coverage)

    cat = a.catalog_meta
    out.append(f"""<footer>Generated locally by trainlab-optimize {e(__version__)}. No code or usage data left this machine.
Prices: {e(cat.get('source', ''))} ({e(cat.get('source_license', ''))}), built {e(cat.get('built_at', ''))}.
Usage sources: {e(', '.join(a.spend.sources) or 'none')}.{''.join('<br>Overlap: ' + e(o) + '.' for o in a.spend.overlaps)}</footer></main></body></html>""")
    return "".join(out)
