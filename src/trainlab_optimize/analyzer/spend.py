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
    sources: list[str] = field(default_factory=list)

    @property
    def has_usage(self) -> bool:
        return bool(self.by_model)


def summarize(records: list[UsageRecord], catalog: PriceCatalog) -> SpendSummary:
    s = SpendSummary()
    if not records:
        return s
    dates = sorted({r.date for r in records if re.match(r"\d{4}-\d{2}-\d{2}", r.date)})
    if dates:
        d0, d1 = date.fromisoformat(dates[0]), date.fromisoformat(dates[-1])
        s.days = (d1 - d0).days + 1
        s.start, s.end = dates[0], dates[-1]
    else:
        s.days = 30
    factor = 30.0 / max(s.days, 1)
    s.sources = sorted({r.source for r in records})

    spelling: dict[str, dict[str, int]] = {}
    raw_in_cost: dict[str, float] = {}
    batch_cost: dict[str, float] = {}
    for r in records:
        b = base_model(r.model) or r.model
        ms = s.by_model.get(b)
        if ms is None:
            ms = s.by_model[b] = ModelSpend(provider=r.provider, model=r.model, base=b)
        spelling.setdefault(b, {}).setdefault(r.model, 0)
        spelling[b][r.model] += 1
        price = catalog.lookup(r.model, r.provider)
        if r.cost_usd is not None:
            cost = r.cost_usd
            if price:
                est_in = price.cost(r.input_tokens, 0, r.cached_input_tokens, r.cache_write_tokens, bool(r.batch))
                est_total = price.cost(r.input_tokens, r.output_tokens, r.cached_input_tokens,
                                       r.cache_write_tokens, bool(r.batch))
                in_part = cost * (est_in / est_total) if est_total else cost * 0.75
            else:
                in_part = cost * 0.75
        elif price:
            cost = price.cost(r.input_tokens, r.output_tokens, r.cached_input_tokens,
                              r.cache_write_tokens, bool(r.batch))
            in_part = price.cost(r.input_tokens, 0, r.cached_input_tokens, r.cache_write_tokens, bool(r.batch))
        else:
            cost = 0.0
            in_part = 0.0
            ms.priced = False
        ms.monthly_cost += cost * factor
        ms.monthly_requests += r.requests * factor
        ms.monthly_input_tokens += r.input_tokens * factor
        ms.monthly_output_tokens += r.output_tokens * factor
        ms.monthly_cached_tokens += r.cached_input_tokens * factor
        raw_in_cost[b] = raw_in_cost.get(b, 0.0) + in_part
        if r.batch:
            batch_cost[b] = batch_cost.get(b, 0.0) + cost
        if r.project:
            ms.projects[r.project] = ms.projects.get(r.project, 0.0) + cost * factor
            s.by_project[r.project] = s.by_project.get(r.project, 0.0) + cost * factor

    for b, ms in s.by_model.items():
        ms.model = max(spelling[b].items(), key=lambda kv: kv[1])[0]
        total_raw = ms.monthly_cost / factor if factor else 0
        if total_raw > 0:
            ms.input_cost_share = min(1.0, raw_in_cost.get(b, 0.0) / total_raw)
            ms.batch_share = batch_cost.get(b, 0.0) / total_raw
        if not ms.priced:
            s.unpriced_models.append(ms.model)
        s.monthly_total += ms.monthly_cost
    return s
