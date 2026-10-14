"""Fold experiment results (Profile 3) into findings: estimated savings become validated ones.

For a call site with an experiment whose baseline is the site's current model:
- a candidate PASSED: the model-switch finding becomes "validated", its saving is the site's
  monthly spend x the cost reduction measured on the same samples (cheapest passing model);
  a site with no model-switch finding gets one (R110);
- every candidate FAILED: the finding is kept as information with no saving, so nobody
  switches on the strength of an estimate that a test has disproved;
- otherwise (inconclusive, needs a judge, errors): the estimate stays, with the experiment
  attached and what it would take to settle it.
"""
from __future__ import annotations

from ..schema import Finding
from .spend import base_model


def _match(site, x: dict) -> bool:
    if x.get("site_id") and x["site_id"] == site.id:
        return True
    s = x.get("site") or ""
    path, _, line = s.rpartition(":")
    return bool(path) and path == site.path and line.isdigit() and abs(int(line) - site.line) <= 3


def _latest(sites, experiments: list[dict]) -> dict:
    """site id -> its most recent experiment. Each experiment attaches to one site: the site it was
    run for (by id); if that id is gone (the code changed), the nearest site within 3 lines in the
    same file whose model agrees (or that names no model)."""
    by_site: dict[str, dict] = {}
    ids = {s.id for s in sites}
    for x in sorted(experiments, key=lambda x: x.get("created") or ""):
        target = None
        if x.get("site_id") in ids:
            target = next(s for s in sites if s.id == x["site_id"])
        else:
            near = [s for s in sites if _match(s, x)
                    and (not s.model or base_model(s.model) == base_model(x.get("baseline") or ""))]
            if near:
                line = int((x.get("site") or "0:0").rpartition(":")[2] or 0)
                target = min(near, key=lambda s: abs(s.line - line))
        if target is not None:
            by_site[target.id] = x
    return by_site


def _summary(x: dict, c: dict | None) -> dict:
    ev = {"experiment": x["id"], "samples": x.get("samples"), "task": x.get("task"),
          "threshold": x.get("threshold"), "reference": x.get("reference"), "spent_usd": x.get("spent_usd")}
    if c:
        ev.update({"tested_model": c["model"], "pass_rate": c.get("pass_rate"), "lower_95": c.get("lower_95"),
                   "measured_cost_reduction": c.get("savings_pct")})
    ev["verdicts"] = ", ".join(f"{k['model']}: {k['verdict']}" for k in x.get("candidates", []))
    return ev


def _model_note(f: Finding, x: dict, site) -> None:
    if site.model and base_model(site.model) != base_model(x.get("baseline") or ""):
        f.evidence["note"] = (f"the experiment's current model `{x['baseline']}` differs from `{site.model}` in the code "
                              "(set by configuration at runtime?)")


def apply(findings: list[Finding], sites: list, attribution: dict, experiments: list[dict]) -> list[Finding]:
    if not experiments:
        return findings
    by_site = _latest(sites, experiments)
    site_by_id = {s.id: s for s in sites}
    covered: set[str] = set()
    for f in findings:
        if f.rule != "R101" or len(f.call_site_ids) != 1:
            continue
        sid = f.call_site_ids[0]
        x = by_site.get(sid)
        if not x:
            continue
        covered.add(sid)
        _update(f, x, attribution.get(sid), site_by_id[sid])
    for sid, x in by_site.items():
        if sid in covered or not x.get("recommended"):
            continue
        s = site_by_id[sid]
        f = Finding(id="", rule="R110", title="", severity="medium", strategy="model", call_site_ids=[sid],
                    summary="", recommendation="", evidence={})
        _update(f, x, attribution.get(sid), s)
        findings.append(f)
    return findings


def _update(f: Finding, x: dict, att, site) -> None:
    _model_note(f, x, site)
    spend = att.monthly_spend_usd if att else x.get("monthly_spend_usd")
    cands = x.get("candidates", [])
    best = next((c for c in cands if c["model"] == x.get("recommended")), None)
    if best:
        sav = best.get("savings_pct") or 0.0
        f.tier = "validated"
        f.monthly_spend_usd = spend
        f.savings_low_usd = f.savings_high_usd = (spend * sav) if spend else None
        f.title = f"Validated: switch `{x['baseline']}` to `{best['model']}`"
        f.summary = (f"Tested on {best['scored']} real inputs: `{best['model']}` matched the reference on "
                     f"{best['passed']}/{best['scored']} ({(best['pass_rate'] or 0):.0%}; 95% lower bound "
                     f"{best['lower_95']:.0%} vs threshold {x['threshold']:.0%}) at "
                     f"{sav:.0%} lower cost per call.")
        f.recommendation = (f"Switch this call to `{best['model']}`. Re-run the experiment when the prompt or "
                            "the input mix changes.")
        f.severity = "high" if spend and spend * sav >= 1000 else "medium"
        f.evidence = {**f.evidence, **_summary(x, best)}
    elif cands and all(c["verdict"] == "FAIL" for c in cands):
        f.tier = "validated"
        f.severity = "info"
        f.savings_low_usd = f.savings_high_usd = 0.0
        f.title = f"Tested: no cheaper model kept quality for `{x['baseline']}`"
        f.summary = "; ".join(f"`{c['model']}` {c['passed']}/{c['scored']}" for c in cands) + \
            f" against a {x['threshold']:.0%} threshold."
        f.recommendation = ("Keep the current model here. A fine-tuned small model trained on this call's data is "
                            "the remaining way to cut its cost.")
        f.trainlab_candidate = True
        f.evidence = {**f.evidence, **_summary(x, None)}
    else:
        need = [c.get("samples_needed") for c in cands if c.get("samples_needed")]
        why = ("needs an LLM judge (--judge) for free-text answers" if any(c["verdict"] == "NEEDS-JUDGE" for c in cands)
               else f"run about {min(need)} samples to settle it" if need else "results were inconclusive")
        f.recommendation += f" An experiment ran ({x['id']}): {why}."
        f.evidence = {**f.evidence, **_summary(x, None)}
