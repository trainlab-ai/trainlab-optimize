"""Waste-detection rules.

Each rule looks at one call site (plus its attributed spend, if usage was imported) and
emits Findings. Savings are only ever estimates here ("estimated" tier when spend is
known, "static" otherwise); validation happens in experiments (profile 3).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from importlib import resources

from ..pricing import Price, PriceCatalog
from ..schema import Attribution, CallSite, Finding
from .spend import SpendSummary, base_model

CACHE_MIN_TOKENS = 1024          # providers' minimum cacheable prefix (Anthropic Sonnet/Opus, OpenAI)
PREMIUM_RATIO = 3.0              # current model >= 3x the economy candidate
DEPRECATION_WINDOW_DAYS = 120
SIMPLE_TASK_HINTS = {"extraction", "classification", "structured_output", "forced_tool"}
SLM_HINTS = {"extraction", "classification"}
OFFLINE_WORDS = ("batch", "nightly", "backfill", "process", "summar", "extract", "classif",
                 "ingest", "index", "bulk", "daily", "cron", "job", "etl", "label")


@dataclass
class RuleContext:
    catalog: PriceCatalog
    spend: SpendSummary
    attribution: dict[str, Attribution]
    today: date
    candidates: dict


def load_candidates() -> dict:
    text = resources.files("trainlab_optimize.catalog").joinpath("candidates.json").read_text()
    return json.loads(text)


def family(site: CallSite) -> str:
    """Provider family for picking replacement candidates."""
    m = (site.model or "").lower()
    if site.provider in ("azure",) or m.startswith("azure/"):
        return "azure"
    if site.provider == "bedrock" or "anthropic.claude" in m:
        return "bedrock"
    if m.startswith(("claude", "anthropic/")):
        return "anthropic"
    if m.startswith(("gemini", "gemini/")):
        return "gemini"
    if m.startswith(("gpt", "o1", "o3", "o4", "openai/", "chatgpt")):
        return "openai"
    return site.provider


def site_price(site: CallSite, catalog: PriceCatalog) -> Price | None:
    hint = site.provider if site.provider in ("azure", "bedrock", "gemini") else None
    return catalog.lookup(site.model, hint)


def best_candidate(site: CallSite, current: Price, ctx: RuleContext) -> Price | None:
    key = "embedding" if site.kind == "embedding" else family(site)
    best = None
    for name in ctx.candidates.get(key, []):
        p = ctx.catalog.lookup(name)
        if not p or _deprecated_soon(p, ctx.today):
            continue
        if p.blended_per_1k() * PREMIUM_RATIO > current.blended_per_1k():
            continue
        if best is None or p.blended_per_1k() < best.blended_per_1k():
            best = p
    return best


def _deprecated_soon(p: Price, today: date, window: int = DEPRECATION_WINDOW_DAYS) -> bool:
    if not p.deprecated:
        return False
    try:
        d = date.fromisoformat(p.deprecated[:10])
    except ValueError:
        return False
    return d <= today + timedelta(days=window)


def _money(v: float | None) -> str:
    return "unknown" if v is None else f"${v:,.0f}"


# --------------------------------------------------------------------------- rules

def rule_premium_model(site: CallSite, ctx: RuleContext) -> list[Finding]:
    if site.kind not in ("chat", "embedding", "model_config") or not site.model:
        return []
    current = site_price(site, ctx.catalog)
    if not current:
        return []
    hints = set(site.task_hints)
    max_out = site.params.get("max_tokens") or site.params.get("max_completion_tokens") \
        or site.params.get("max_output_tokens")
    short_output = isinstance(max_out, int) and max_out <= 64
    if site.kind != "embedding" and not (hints & SIMPLE_TASK_HINTS or short_output):
        return []
    cand = best_candidate(site, current, ctx)
    if not cand:
        return []
    ratio = current.blended_per_1k() / cand.blended_per_1k()
    a = ctx.attribution.get(site.id)
    spend = a.monthly_spend_usd if a else None
    high = spend * (1 - 1 / ratio) if spend else None
    low = high * 0.5 if high else None
    slm = bool(hints & SLM_HINTS) and site.kind != "embedding"
    task = ", ".join(sorted(hints)) or ("short outputs" if short_output else "embeddings")
    if site.kind == "embedding":
        title = f"Premium embedding model `{site.model}`"
        rec = (f"Benchmark `{cand.model}` on retrieval quality (recall@k on your queries). "
               "Switching embeddings requires re-indexing the vector store.")
    else:
        title = f"Premium model `{site.model}` on a simple task ({task})"
        rec = (f"Test `{cand.model}` (~{ratio:.0f}x cheaper per token) on a sample of real inputs "
               "with `trainlab experiment`.")
        if slm:
            rec += (" This is a narrow, repetitive task: a fine-tuned small model can usually match "
                    "quality at a fraction of the cost.")
    sev = "high" if (spend and spend >= 1000) or (not spend and ratio >= 10) else "medium"
    return [Finding(
        id="", rule="R101", title=title, severity=sev, strategy="slm" if slm else "model",
        call_site_ids=[site.id],
        summary=(f"`{site.model}` costs ~{ratio:.0f}x more per token than `{cand.model}`. "
                 f"Signals: {task}."),
        recommendation=rec,
        evidence={"current_model": current.model, "candidate_model": cand.model,
                  "price_ratio": round(ratio, 1), "task_hints": sorted(hints),
                  "current_per_1m_in": round(current.input * 1e6, 3),
                  "candidate_per_1m_in": round(cand.input * 1e6, 3)},
        tier="estimated" if spend else "static", monthly_spend_usd=spend,
        savings_low_usd=low, savings_high_usd=high, trainlab_candidate=slm,
    )]


def rule_deprecated(site: CallSite, ctx: RuleContext) -> list[Finding]:
    if not site.model:
        return []
    p = site_price(site, ctx.catalog)
    if not p or not _deprecated_soon(p, ctx.today):
        return []
    d = date.fromisoformat(p.deprecated[:10])
    past = d <= ctx.today
    a = ctx.attribution.get(site.id)
    return [Finding(
        id="", rule="R102", severity="high" if (d - ctx.today).days <= 30 else "medium",
        title=f"`{site.model}` is {'past its' if past else 'near its'} deprecation date ({p.deprecated[:10]})",
        strategy="model", call_site_ids=[site.id],
        summary=("The provider has scheduled this model for retirement. Calls will fail after the "
                 "shutdown date, and older models are usually priced above their successors."),
        recommendation="Migrate to the provider's current model; run `trainlab experiment` to compare quality first.",
        evidence={"model": p.model, "deprecation_date": p.deprecated[:10],
                  "per_1m_in": round(p.input * 1e6, 3), "per_1m_out": round(p.output * 1e6, 3)},
        tier="estimated" if a else "static", monthly_spend_usd=a.monthly_spend_usd if a else None,
    )]


def rule_caching(site: CallSite, ctx: RuleContext) -> list[Finding]:
    tokens = site.system_prompt_tokens or 0
    if site.kind != "chat" or tokens < CACHE_MIN_TOKENS:
        return []
    fam = family(site)
    p = site_price(site, ctx.catalog)
    needs_marker = fam in ("anthropic", "bedrock")
    missing_marker = needs_marker and not site.uses_cache_control
    if not (missing_marker or site.prompt_dynamic_prefix):
        return []
    a = ctx.attribution.get(site.id)
    spend = a.monthly_spend_usd if a else None
    discount = (1 - p.cache_read / p.input) if (p and p.cache_read) else 0.5
    high = low = None
    if spend:
        ms = ctx.spend.by_model.get(base_model(site.model) or "")
        in_share = ms.input_cost_share if ms else 0.75
        static_frac = min(0.9, tokens / (tokens + 300))
        high = spend * in_share * static_frac * discount
        low = high * 0.5
    parts = []
    if site.prompt_dynamic_prefix:
        parts.append("dynamic content (e.g. retrieved documents) is placed before the static "
                     "instructions, so the prompt prefix changes on every request and cannot be cached")
    if missing_marker:
        parts.append("no `cache_control` breakpoint is set, so the provider never caches it")
    rec = []
    if site.prompt_dynamic_prefix:
        rec.append("Move static instructions to the start of the prompt and put per-request content after them.")
    if missing_marker:
        rec.append("Add `cache_control: {\"type\": \"ephemeral\"}` on the static system block.")
    else:
        rec.append("OpenAI caches identical prefixes of 1,024+ tokens automatically once the order is fixed.")
    return [Finding(
        id="", rule="R103", strategy="cache", call_site_ids=[site.id],
        severity="high" if (spend and spend >= 500) or tokens >= 2000 else "medium",
        title=f"~{tokens:,}-token system prompt is not being cached",
        summary="The static prompt is re-billed at full price on every call: " + "; ".join(parts) + ".",
        recommendation=" ".join(rec),
        evidence={"system_prompt_tokens_est": tokens, "dynamic_prefix": site.prompt_dynamic_prefix,
                  "cache_control": site.uses_cache_control,
                  "cache_read_discount": round(discount, 2)},
        tier="estimated" if spend else "static", monthly_spend_usd=spend,
        savings_low_usd=low, savings_high_usd=high,
    )]


def rule_batch(site: CallSite, ctx: RuleContext) -> list[Finding]:
    if site.kind != "chat" or not site.in_loop or site.params.get("stream") is True:
        return []
    fam = family(site)
    if fam not in ("openai", "azure", "anthropic", "bedrock", "gemini"):
        return []
    name = (site.function or "").lower()
    offline = any(w in name for w in OFFLINE_WORDS)
    p = site_price(site, ctx.catalog)
    discount = (1 - p.batch_input / p.input) if (p and p.batch_input) else 0.5
    a = ctx.attribution.get(site.id)
    spend = a.monthly_spend_usd if a else None
    high = spend * discount if spend else None
    return [Finding(
        id="", rule="R104", strategy="batch", call_site_ids=[site.id],
        severity="medium" if offline else "low",
        title=f"Sequential calls in a loop{' in `' + site.function + '`' if site.function else ''}",
        summary=("Requests are sent one at a time inside a loop. If results are not needed "
                 "immediately, the provider Batch API processes them at a discount."),
        recommendation=(f"If this workload tolerates up to 24h latency, submit it through the Batch API "
                        f"(~{discount:.0%} cheaper). Otherwise, run requests concurrently to cut wall time."),
        evidence={"function": site.function, "looks_offline": offline, "batch_discount": round(discount, 2)},
        tier="estimated" if spend else "static", monthly_spend_usd=spend,
        savings_low_usd=0.0 if spend else None, savings_high_usd=high,
    )]


def rule_output_cap(site: CallSite, ctx: RuleContext) -> list[Finding]:
    if site.kind != "chat" or site.confidence != "high" or family(site) in ("anthropic", "bedrock"):
        return []
    if any(k in site.params for k in ("max_tokens", "max_completion_tokens", "max_output_tokens",
                                      "_kwargs_passthrough", "maxTokens")):
        return []
    return [Finding(
        id="", rule="R105", strategy="config", call_site_ids=[site.id], severity="low",
        title="No output token limit",
        summary="Without a cap, an unexpected long generation (or a reasoning model) is billed in full.",
        recommendation="Set `max_completion_tokens` (or `max_tokens`) to a safe upper bound for this task.",
        evidence={}, tier="static",
    )]


def rule_retry(site: CallSite, ctx: RuleContext) -> list[Finding]:
    if not site.retry_unbounded:
        return []
    return [Finding(
        id="", rule="R106", strategy="reliability", call_site_ids=[site.id], severity="medium",
        title=f"Unbounded retries around an LLM call{' in `' + site.function + '`' if site.function else ''}",
        summary=("The retry decorator has no stop condition. During a provider outage or a "
                 "persistent error, this retries forever and every attempt is billed."),
        recommendation="Add a stop condition, e.g. `@retry(stop=stop_after_attempt(3), wait=wait_exponential())`.",
        evidence={"function": site.function}, tier="static",
    )]


SITE_RULES = [rule_premium_model, rule_deprecated, rule_caching, rule_batch, rule_output_cap, rule_retry]


def coverage_findings(sites: list[CallSite], coverage: dict, spend: SpendSummary) -> list[Finding]:
    out: list[Finding] = []
    unresolved = [s for s in sites if s.kind in ("chat", "embedding") and not s.model]
    if unresolved:
        out.append(Finding(
            id="", rule="R901", strategy="coverage", severity="info",
            call_site_ids=[s.id for s in unresolved],
            title=f"{len(unresolved)} call site(s) choose the model at runtime",
            summary=("The model comes from configuration, an argument, or an environment variable "
                     "without a default, so spend cannot be linked to these lines statically."),
            recommendation="Add `trainlab.instrument()` (profile 2) to tag live calls with their source location.",
            evidence={"examples": [f"{s.path}:{s.line} ({s.model_expr or 'no model argument'})"
                                   for s in unresolved[:5]]},
        ))
    if coverage.get("unattributed_usd"):
        out.append(Finding(
            id="", rule="R902", strategy="coverage", severity="info", call_site_ids=[],
            title=f"${coverage['unattributed_usd']:,.0f}/month of spend has no matching call site",
            summary=("These models appear in billing but not in the scanned code. They may be called "
                     "from another repository, a notebook, a different service, or a runtime-configured model."),
            recommendation="Scan the other repositories that share this API key/project, or instrument at runtime.",
            evidence={"by_model": {k: round(v, 2) for k, v in coverage["unattributed_by_model"].items()}},
            tier="estimated",
        ))
    if spend.unpriced_models:
        out.append(Finding(
            id="", rule="R903", strategy="coverage", severity="info", call_site_ids=[],
            title=f"{len(spend.unpriced_models)} billed model(s) not in the price catalog",
            summary="Their cost could not be computed from tokens (and the export had no cost column).",
            recommendation="Run `trainlab prices update`, or import a billing export that includes cost.",
            evidence={"models": spend.unpriced_models},
        ))
    return out
