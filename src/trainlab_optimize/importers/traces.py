"""Trace importers: Langfuse observation exports and OpenTelemetry (OTLP JSON) GenAI spans.

Only usage numbers are kept (model, tokens, cost, date, project, code location). Prompt and
completion text in the trace is never read into the workspace.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ..schema import UsageRecord

# OpenTelemetry GenAI semantic conventions (current names first, then legacy names still emitted
# by many instrumentations).
OTEL_MODEL = ("gen_ai.response.model", "gen_ai.request.model", "llm.response.model", "llm.request.model")
OTEL_PROVIDER = ("gen_ai.provider.name", "gen_ai.system", "llm.vendor")
OTEL_IN = ("gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens", "llm.usage.prompt_tokens")
OTEL_OUT = ("gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens", "llm.usage.completion_tokens")
OTEL_CACHED = ("gen_ai.usage.cache_read.input_tokens", "gen_ai.usage.cache_read_input_tokens",
               "gen_ai.usage.cached_tokens")
OTEL_COST = ("gen_ai.usage.cost", "llm.usage.cost")
OTEL_FILE = ("code.file.path", "code.filepath", "trainlab.code.file")
OTEL_LINE = ("code.line.number", "code.lineno", "trainlab.code.line")
OTEL_PROJECT = ("service.name", "trainlab.project", "deployment.environment")

PROVIDER_ALIASES = {"az.ai.openai": "azure", "azure.ai.openai": "azure", "aws.bedrock": "bedrock",
                    "gcp.gemini": "gemini", "gcp.vertex_ai": "gemini", "gcp.gen_ai": "gemini"}


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _day_from_iso(s: str | None) -> str:
    return (s or "")[:10] or "unknown"


def _day_from_nanos(n) -> str:
    try:
        return datetime.fromtimestamp(int(n) / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return "unknown"


class _Agg:
    """Aggregate spans/observations into daily records per (model, project, code location)."""

    def __init__(self, source: str):
        self.source = source
        self.rows: dict[tuple, dict] = defaultdict(lambda: {"in": 0, "out": 0, "cached": 0, "req": 0,
                                                             "cost": 0.0, "has_cost": True})

    def add(self, day, provider, model, tin, tout, cached, cost, project, code_path, code_line):
        key = (day, provider, model, project, code_path, code_line)
        r = self.rows[key]
        r["in"] += int(tin)
        r["out"] += int(tout)
        r["cached"] += int(cached)
        r["req"] += 1
        if cost is None:
            r["has_cost"] = False
        else:
            r["cost"] += cost

    def records(self) -> list[UsageRecord]:
        out = []
        for (day, provider, model, project, path, line), r in self.rows.items():
            out.append(UsageRecord(
                date=day, provider=provider, model=model, input_tokens=r["in"], output_tokens=r["out"],
                cached_input_tokens=r["cached"], requests=r["req"],
                cost_usd=r["cost"] if r["has_cost"] else None, project=project, source=self.source,
                code_path=path, code_line=line))
        return out


def _infer_provider(model: str) -> str:
    m = model.lower()
    return ("anthropic" if "claude" in m and "anthropic." not in m else
            "bedrock" if "anthropic." in m or m.startswith(("amazon.", "meta.")) else
            "gemini" if "gemini" in m else "openai")


# --------------------------------------------------------------------------- Langfuse

def _maybe_json(v):
    if isinstance(v, str) and v[:1] in "{[":
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def parse_langfuse(observations: list[dict], source: str = "langfuse") -> list[UsageRecord]:
    """Langfuse observations (UI JSON/CSV export or /api/public observations).

    Supports the current shape (usageDetails, costDetails, totalCost) and the older one
    (usage{input,output}, calculatedTotalCost, promptTokens/completionTokens).
    """
    agg = _Agg(source)
    for o in observations:
        if not isinstance(o, dict):
            continue
        if str(o.get("type", "GENERATION")).upper() not in ("GENERATION", "EMBEDDING"):
            continue
        model = o.get("model")
        if not model:
            continue
        ud = _maybe_json(o.get("usageDetails")) or {}
        u = _maybe_json(o.get("usage")) or {}
        cd = _maybe_json(o.get("costDetails")) or {}
        md = _maybe_json(o.get("metadata")) or {}
        md = md if isinstance(md, dict) else {}
        tin = _num(ud.get("input") or u.get("input") or o.get("promptTokens"))
        tout = _num(ud.get("output") or u.get("output") or o.get("completionTokens"))
        cached = _num(ud.get("input_cached_tokens") or ud.get("cache_read_input_tokens")
                      or ud.get("input_cache_read") or 0)
        cost = o.get("totalCost")
        if cost is None:
            cost = o.get("calculatedTotalCost")
        if cost is None and isinstance(cd, dict) and cd:
            cost = cd.get("total") or sum(_num(v) for v in cd.values())
        if cost is None and isinstance(u, dict) and u.get("totalCost") is not None:
            cost = u.get("totalCost")
        cost = None if cost in (None, "") else _num(cost)
        path = md.get("code.file.path") or md.get("code.filepath") or md.get("trainlab.code.file")
        line = md.get("code.line.number") or md.get("code.lineno") or md.get("trainlab.code.line")
        agg.add(_day_from_iso(o.get("startTime")), _infer_provider(str(model)), str(model), tin, tout,
                cached, cost, o.get("environment") or None,
                path, int(line) if str(line or "").isdigit() else None)
    return agg.records()


def read_langfuse(path: str | Path) -> list[UsageRecord]:
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig", errors="replace")
    source = f"langfuse:{p.name}"
    if p.suffix.lower() == ".csv":
        import csv
        import io
        return parse_langfuse(list(csv.DictReader(io.StringIO(text))), source)
    if p.suffix.lower() == ".jsonl":
        rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    else:
        data = json.loads(text)
        rows = data.get("data", data) if isinstance(data, dict) else data
    return parse_langfuse(rows, source)


# --------------------------------------------------------------------------- OpenTelemetry

def _attr_value(v: dict):
    for k in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if k in v:
            return v[k]
    return None


def _attrs(lst) -> dict:
    if isinstance(lst, dict):
        return lst
    out = {}
    for a in lst or []:
        if isinstance(a, dict) and "key" in a:
            out[a["key"]] = _attr_value(a.get("value") or {})
    return out


def _first(d: dict, keys):
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    return None


def _iter_spans(doc):
    """Yield (resource_attrs, span) from OTLP JSON, a list of OTLP docs, or flat span lists."""
    if isinstance(doc, list):
        for d in doc:
            yield from _iter_spans(d)
        return
    if not isinstance(doc, dict):
        return
    if "resourceSpans" in doc:
        for rs in doc["resourceSpans"]:
            res = _attrs((rs.get("resource") or {}).get("attributes"))
            for ss in rs.get("scopeSpans", []) or rs.get("instrumentationLibrarySpans", []):
                for span in ss.get("spans", []):
                    yield res, span
    elif "attributes" in doc:
        yield {}, doc


def parse_otel(doc, source: str = "otel") -> list[UsageRecord]:
    agg = _Agg(source)
    for res, span in _iter_spans(doc):
        a = _attrs(span.get("attributes"))
        model = _first(a, OTEL_MODEL)
        if not model:
            continue
        tin, tout = _num(_first(a, OTEL_IN)), _num(_first(a, OTEL_OUT))
        if not (tin or tout) and not _first(a, OTEL_COST):
            continue
        provider = str(_first(a, OTEL_PROVIDER) or _infer_provider(str(model))).lower()
        provider = PROVIDER_ALIASES.get(provider, provider)
        cost = _first(a, OTEL_COST)
        line = _first(a, OTEL_LINE)
        agg.add(_day_from_nanos(span.get("startTimeUnixNano")), provider, str(model), tin, tout,
                _num(_first(a, OTEL_CACHED)), None if cost is None else _num(cost),
                _first(res, OTEL_PROJECT) or _first(a, OTEL_PROJECT),
                _first(a, OTEL_FILE), int(line) if line is not None and str(line).isdigit() else None)
    return agg.records()


def read_otel(path: str | Path) -> list[UsageRecord]:
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig", errors="replace")
    source = f"otel:{p.name}"
    try:
        doc = json.loads(text)
    except ValueError:
        doc = [json.loads(ln) for ln in text.splitlines() if ln.strip()]   # JSONL (file exporter)
    return parse_otel(doc, source)
