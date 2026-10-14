"""Normalize usage records into monthly spend per model."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from ..pricing import PriceCatalog
from ..schema import UsageRecord

_PREFIX = re.compile(r"^(openai|anthropic|azure|azure_ai|bedrock|gemini|vertex_ai|litellm|groq|"
                     r"together_ai|mistral|deepseek|openrouter|models)/", re.I)
_DATE = re.compile(r"-(20\d{2}-?\d{2}-?\d{2}|latest)$")
_REGION = re.compile(r"^(us|eu|apac|au|jp|global)\.")


def base_model(name: str | None) -> str | None:
    """Normalize a model name so code and billing names meet: 'openai/gpt-4o-2024-08-06' -> 'gpt-4o'."""
    if not name:
        return None
    m = name.strip().lower()
    while _PREFIX.match(m):
        m = _PREFIX.sub("", m, count=1)
    m = _REGION.sub("", m)
    m = re.sub(r"-v\d+(:\d+)?$", "", m)       # bedrock -v1:0
    m = _DATE.sub("", m)
    m = re.sub(r"-\d{8}$", "", m)             # claude-...-20250929
    m = re.sub(r"-\d{4}(-preview)?$", "", m)  # gpt-4-0613, gpt-4-1106-preview, gpt-3.5-turbo-0125
    return _canonical_family(m)


_VENDOR = re.compile(r"^(anthropic|meta|amazon|cohere|mistral|ai21)\.")


def _canonical_family(m: str) -> str:
    """Make code names and billing names meet:
    Claude3.5Sonnet / claude-3.5-sonnet / anthropic.claude-3-5-sonnet / claude-sonnet-3-5 -> claude-sonnet-3-5
    """
    m = _VENDOR.sub("", m)
    if "claude" in m:
        m = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", "-", m)   # claude3.5sonnet -> claude-3.5-sonnet
        m = m.replace(".", "-").replace("_", "-")
        m = re.sub(r"-+", "-", m)
        tier = re.search(r"\b(opus|sonnet|haiku|instant)\b", m)
        ver = re.search(r"\d+(?:-\d+)?", m)
        if tier and ver:
            rest = re.sub(rf"\b{tier.group(1)}\b|{re.escape(ver.group(0))}|claude", "", m).strip("-")
            m = "-".join(x for x in ("claude", tier.group(1), ver.group(0), rest) if x)
    return m


@dataclass
class ModelSpend:
    provider: str
    model: str            # as billed (most common spelling)
    base: str
    monthly_cost: float = 0.0
    monthly_requests: float = 0.0
    monthly_input_tokens: float = 0.0
    monthly_output_tokens: float = 0.0
    monthly_cached_tokens: float = 0.0
    input_cost_share: float = 0.75      # share of cost from input tokens
    priced: bool = True                 # False if neither billed cost nor catalog price
    batch_share: float = 0.0
    projects: dict[str, float] = field(default_factory=dict)


@dataclass
class SpendSummary:
    days: int = 0
    start: str | None = None
    end: str | None = None
    monthly_total: float = 0.0
    by_model: dict[str, ModelSpend] = field(default_factory=dict)   # key: base model
    by_project: dict[str, float] = field(default_factory=dict)
    unpriced_models: list[str] = field(default_factory=list)
    # model -> catalog entry it was priced from, when that is another host's listing (no billed cost)
    approx_priced: dict[str, str] = field(default_factory=dict)
    # (code_path, code_line, base model) -> [monthly_cost, monthly_requests]; from traces only
    by_location: dict[tuple, list] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    # billing sources that covered the same model-days as a preferred source and were not counted
    overlaps: list[str] = field(default_factory=list)
    # API key (id or name, as the source reports it) -> monthly cost; and base models seen per key
    by_api_key: dict[str, float] = field(default_factory=dict)
    key_models: dict[str, set] = field(default_factory=dict)

    @property
    def has_usage(self) -> bool:
        return bool(self.by_model)


TRACE_SOURCES = ("langfuse", "otel")


def _is_trace(source: str) -> bool:
    return source.startswith(TRACE_SOURCES)


def _channel(r: UsageRecord) -> str:
    """Who billed it: the same model bought directly, on Azure, on Bedrock or through OpenRouter is
    different traffic, so it never counts as an overlap."""
    from ..scanner.providers import maker_of
    p = (r.provider or "").lower()
    if r.model.lower().startswith("azure/"):
        return "azure"
    if p in ("", "unknown", "openai_compatible", "litellm"):
        return maker_of(r.model) or p
    return p


def _source_rank(source: str, records: list[UsageRecord]) -> tuple:
    """Which billing source to trust when two cover the same model and day (lower wins)."""
    has_cost = any(r.cost_usd is not None for r in records)
    is_api = source.startswith(("openai_usage", "anthropic_usage", "openrouter_usage", "openrouter_api"))
    return (0 if has_cost else 1 if is_api else 2, source)


def _dedupe_billing(billing: list[UsageRecord]) -> tuple[list[UsageRecord], list[str]]:
    """The same traffic often arrives twice (usage API + billing CSV of the same month).
    Count each (day, model) from one source only: billed cost first, then usage APIs, then CSVs."""
    by_source: dict[str, list[UsageRecord]] = {}
    for r in billing:
        by_source.setdefault(r.source, []).append(r)
    if len(by_source) < 2:
        return billing, []
    rank = {src: _source_rank(src, rs) for src, rs in by_source.items()}
    owner: dict[tuple, str] = {}
    def key_of(r: UsageRecord) -> tuple:
        return (r.date, base_model(r.model) or r.model, _channel(r))

    for r in billing:
        key = key_of(r)
        cur = owner.get(key)
        if cur is None or rank[r.source] < rank[cur]:
            owner[key] = r.source
    kept, dropped = [], {}
    for r in billing:
        if owner[key_of(r)] == r.source:
            kept.append(r)
        else:
            dropped.setdefault(r.source, set()).add(owner[key_of(r)])
    notes = [f"{src} overlaps {', '.join(sorted(win))} for the same models and days; counted once"
             for src, win in sorted(dropped.items())]
    return kept, notes


def _period(records: list[UsageRecord]) -> tuple[int, str | None, str | None]:
    dates = sorted({r.date for r in records if re.match(r"\d{4}-\d{2}-\d{2}", r.date or "")})
    if not dates:
        return 30, None, None
    d0, d1 = date.fromisoformat(dates[0]), date.fromisoformat(dates[-1])
    return (d1 - d0).days + 1, dates[0], dates[-1]


def _record_cost(r: UsageRecord, catalog: PriceCatalog) -> tuple[float, float, bool]:
    """(cost, input-part cost, priced)"""
    price = catalog.lookup(r.model, r.provider)
    if r.cost_usd is not None:
        cost = r.cost_usd
        if price:
            est_in = price.cost(r.input_tokens, 0, r.cached_input_tokens, r.cache_write_tokens, bool(r.batch))
            est_total = price.cost(r.input_tokens, r.output_tokens, r.cached_input_tokens,
                                   r.cache_write_tokens, bool(r.batch))
            return cost, (cost * est_in / est_total) if est_total else cost * 0.75, True
        if r.input_tokens or r.output_tokens:
            return cost, cost * r.input_tokens / max(r.input_tokens + 4 * r.output_tokens, 1), True
        return cost, cost * 0.75, True
    if price:
        return (price.cost(r.input_tokens, r.output_tokens, r.cached_input_tokens, r.cache_write_tokens,
                           bool(r.batch)),
                price.cost(r.input_tokens, 0, r.cached_input_tokens, r.cache_write_tokens, bool(r.batch)), True)
    return 0.0, 0.0, False


def summarize(records: list[UsageRecord], catalog: PriceCatalog) -> SpendSummary:
    """Monthly spend per model.

    - Every source is scaled to 30 days on its own period (30 days of billing + 7 days of traces
      are not one 37-day window).
    - Billing sources (usage APIs, billing CSVs, cloud exports) are authoritative for totals.
      Traces of the same traffic would double count, so for models present in billing, traces
      only decide *where* that spend comes from (their per-location share is applied to the
      billed amount). Models seen only in traces are added as spend.
    """
    s = SpendSummary()
    if not records:
        return s
    s.sources = sorted({r.source for r in records})
    by_source: dict[str, list[UsageRecord]] = {}
    for r in records:
        by_source.setdefault(r.source, []).append(r)
    factors = {src: 30.0 / max(_period(rs)[0], 1) for src, rs in by_source.items()}
    billing, s.overlaps = _dedupe_billing([r for r in records if not _is_trace(r.source)])
    traces = [r for r in records if _is_trace(r.source)]
    s.days, s.start, s.end = _period(billing or traces)

    spelling: dict[str, dict[str, int]] = {}
    raw_in: dict[str, float] = {}
    raw_total: dict[str, float] = {}
    batch_cost: dict[str, float] = {}
    billed_models = {base_model(r.model) or r.model for r in billing}
    billed_cost: set[str] = set()

    def add(r: UsageRecord, f: float) -> None:
        b = base_model(r.model) or r.model
        ms = s.by_model.get(b)
        if ms is None:
            ms = s.by_model[b] = ModelSpend(provider=r.provider, model=r.model, base=b)
        spelling.setdefault(b, {}).setdefault(r.model, 0)
        spelling[b][r.model] += 1
        cost, in_part, priced = _record_cost(r, catalog)
        if r.cost_usd is not None:
            billed_cost.add(b)
        if not priced:
            ms.priced = False
        ms.monthly_cost += cost * f
        ms.monthly_requests += r.requests * f
        ms.monthly_input_tokens += r.input_tokens * f
        ms.monthly_output_tokens += r.output_tokens * f
        ms.monthly_cached_tokens += r.cached_input_tokens * f
        raw_in[b] = raw_in.get(b, 0.0) + in_part * f
        raw_total[b] = raw_total.get(b, 0.0) + cost * f
        if r.batch:
            batch_cost[b] = batch_cost.get(b, 0.0) + cost * f
        if r.project:
            ms.projects[r.project] = ms.projects.get(r.project, 0.0) + cost * f
            s.by_project[r.project] = s.by_project.get(r.project, 0.0) + cost * f
        if r.api_key:
            s.by_api_key[r.api_key] = s.by_api_key.get(r.api_key, 0.0) + cost * f
            s.key_models.setdefault(r.api_key, set()).add(b)

    for r in billing:
        add(r, factors[r.source])
    for r in traces:
        if (base_model(r.model) or r.model) not in billed_models:
            add(r, factors[r.source])

    # Locations from traces, expressed as a share of each model's (billed) monthly spend.
    trace_total: dict[str, float] = {}
    trace_req: dict[str, float] = {}
    loc_raw: dict[tuple, list] = {}
    for r in traces:
        b = base_model(r.model) or r.model
        cost, _, _ = _record_cost(r, catalog)
        f = factors[r.source]
        trace_total[b] = trace_total.get(b, 0.0) + cost * f
        trace_req[b] = trace_req.get(b, 0.0) + r.requests * f
        if r.code_path:
            loc = loc_raw.setdefault((r.code_path, r.code_line, b), [0.0, 0.0])
            loc[0] += cost * f
            loc[1] += r.requests * f
    for (path, line, b), (cost, req) in loc_raw.items():
        ms = s.by_model.get(b)
        if not ms:
            continue
        cost_share = cost / trace_total[b] if trace_total.get(b) else 0.0
        req_share = req / trace_req[b] if trace_req.get(b) else cost_share
        s.by_location[(path, line, b)] = [ms.monthly_cost * cost_share, ms.monthly_requests * req_share]

    for b, ms in s.by_model.items():
        ms.model = max(spelling[b].items(), key=lambda kv: kv[1])[0]
        if raw_total.get(b):
            ms.input_cost_share = min(1.0, raw_in.get(b, 0.0) / raw_total[b])
            ms.batch_share = batch_cost.get(b, 0.0) / raw_total[b]
        if not ms.priced:
            s.unpriced_models.append(ms.model)
        elif b not in billed_cost:
            price = catalog.lookup(ms.model, ms.provider)
            if price and price.approximate:
                s.approx_priced[ms.model] = price.model
        s.monthly_total += ms.monthly_cost
    return s
