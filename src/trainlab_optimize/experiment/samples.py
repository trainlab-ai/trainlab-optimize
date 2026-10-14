"""Samples: real inputs from one call site, to replay against cheaper models.

Where they come from
- `instrument(capture=...)` (opt-in): .trainlab/samples/*.jsonl, with the production output.
- Files the user already has: JSONL (chat messages, prompt/completion, input/expected),
  CSV (one input column, optional expected column), Langfuse observation exports, and
  OpenTelemetry spans that carry prompt content.

A sample is an OpenAI-style message list plus the request parameters that matter for
quality (max_tokens, temperature, JSON mode), and optionally the output production gave
(`reference`) and a known-correct answer (`expected`).
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..redact import redact

PARAM_KEYS = ("max_tokens", "max_completion_tokens", "max_output_tokens", "temperature", "top_p",
              "response_format", "stop", "seed")


@dataclass
class Sample:
    id: str
    messages: list[dict]                       # [{"role": "system"|"user"|"assistant", "content": str}]
    params: dict = field(default_factory=dict)
    reference: str | None = None               # output the production model gave
    expected: str | None = None                # known-correct answer, if the dataset has one
    site: str | None = None                    # "path:line"
    model: str | None = None                   # model that served it in production
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def json_mode(self) -> bool:
        rf = self.params.get("response_format")
        return isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema")

    def prompt_chars(self) -> int:
        return sum(len(m.get("content") or "") for m in self.messages)


# ------------------------------------------------------------------ normalisation
def _text(content) -> str:
    """Message content as text: strings, OpenAI/Anthropic content parts, Gemini parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for k in ("text", "content", "value"):
            if isinstance(content.get(k), str):
                return content[k]
        return _text(content.get("parts")) if "parts" in content else ""
    if isinstance(content, (list, tuple)):
        return "".join(_text(c) for c in content)
    return str(content)


def normalize_messages(raw, system=None) -> list[dict]:
    """OpenAI, Anthropic (separate system), Gemini (contents/parts, role=model), Responses API
    (`input` string or items) -> [{"role", "content"}]."""
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": _text(system)})
    if isinstance(raw, str):
        return out + [{"role": "user", "content": raw}]
    for m in raw or []:
        if isinstance(m, str):
            out.append({"role": "user", "content": m})
            continue
        if not isinstance(m, dict):
            continue
        role = m.get("role") or "user"
        role = {"model": "assistant", "developer": "system", "human": "user", "ai": "assistant"}.get(role, role)
        if role not in ("system", "user", "assistant"):
            continue                                    # tool messages are not replayed
        content = m.get("content") if "content" in m else m.get("parts")
        out.append({"role": role, "content": _text(content)})
    return out


def sample_id(messages: list[dict], params: dict | None = None) -> str:
    blob = json.dumps({"m": messages, "p": {k: v for k, v in (params or {}).items() if k != "seed"}},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def make_sample(messages, params=None, reference=None, expected=None, site=None, model=None, system=None,
                meta=None, do_redact=False) -> Sample | None:
    msgs = normalize_messages(messages, system)
    if not msgs or not any(m["role"] == "user" and m["content"].strip() for m in msgs):
        return None
    # A trailing assistant turn is the answer, not part of the prompt.
    if msgs[-1]["role"] == "assistant" and reference is None:
        reference = msgs.pop()["content"]
    if do_redact:
        msgs = [{"role": m["role"], "content": redact(m["content"]) or ""} for m in msgs]
        reference = redact(reference) if reference else reference
    p = {k: v for k, v in (params or {}).items() if k in PARAM_KEYS and v is not None}
    return Sample(id=sample_id(msgs, p), messages=msgs, params=p, reference=reference,
                  expected=None if expected is None else str(expected), site=site, model=model, meta=meta or {})


# ------------------------------------------------------------------ loaders
def _from_record(d: dict, do_redact: bool) -> Sample | None:
    """One JSON object in any of the supported shapes."""
    if not isinstance(d, dict):
        return None
    if "messages" in d or "contents" in d:                       # trainlab capture, OpenAI chat/eval files
        return make_sample(d.get("messages") or d.get("contents"), d.get("params") or d,
                           reference=d.get("output") or d.get("reference") or d.get("completion"),
                           expected=d.get("expected") or d.get("ideal") or d.get("label"),
                           site=d.get("site"), model=d.get("model"), system=d.get("system"),
                           meta={k: d[k] for k in ("tags", "ts") if k in d}, do_redact=do_redact)
    if "prompt" in d or "input" in d or "question" in d:      # prompt/completion, input/expected
        inp = d.get("prompt") or d.get("input") or d.get("question")
        if isinstance(inp, dict):                                # Langfuse-style input object
            inp = inp.get("messages") or inp.get("prompt") or inp.get("input") or json.dumps(inp)
        out = d.get("completion") or d.get("output")
        if isinstance(out, dict):
            out = _text(out.get("content") or out.get("text") or out.get("choices", [{}])[0].get("message", {}))
        return make_sample(inp, d.get("params") or d.get("modelParameters") or {}, reference=out,
                           expected=d.get("expected") or d.get("ideal") or d.get("answer") or d.get("label"),
                           site=d.get("site"), model=d.get("model"), system=d.get("system"), do_redact=do_redact)
    return None


def _otel_samples(doc, do_redact: bool) -> list[Sample]:
    from ..importers.traces import _attrs, _iter_spans, OTEL_MODEL, OTEL_FILE, OTEL_LINE, _first
    out = []
    for _, span in _iter_spans(doc):
        a = _attrs(span.get("attributes"))
        msgs = a.get("gen_ai.input.messages") or a.get("gen_ai.prompt")
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except ValueError:
                pass
        if not msgs:                                              # older indexed attributes
            i, msgs = 0, []
            while f"gen_ai.prompt.{i}.content" in a:
                msgs.append({"role": a.get(f"gen_ai.prompt.{i}.role", "user"), "content": a[f"gen_ai.prompt.{i}.content"]})
                i += 1
        if not msgs:
            continue
        ref = a.get("gen_ai.output.messages") or a.get("gen_ai.completion") or a.get("gen_ai.completion.0.content")
        if isinstance(ref, str) and ref[:1] in "[{":
            try:
                ref = _text(json.loads(ref))
            except ValueError:
                pass
        path, line = _first(a, OTEL_FILE), _first(a, OTEL_LINE)
        s = make_sample(msgs, {"max_tokens": a.get("gen_ai.request.max_tokens"),
                               "temperature": a.get("gen_ai.request.temperature")},
                        reference=_text(ref) if ref else None, model=_first(a, OTEL_MODEL),
                        site=f"{path}:{line}" if path else None, do_redact=do_redact)
        if s:
            out.append(s)
    return out


def load_samples(path: str | Path, *, do_redact: bool = False, input_column: str | None = None,
                 expected_column: str | None = None, system: str | None = None) -> list[Sample]:
    """Read samples from a file or a directory of capture files. Duplicates are dropped."""
    p = Path(path)
    files = sorted(p.glob("*.jsonl")) + sorted(p.glob("*.json")) if p.is_dir() else [p]
    found: list[Sample] = []
    for f in files:
        suffix = f.suffix.lower()
        if suffix == ".csv":
            found += _csv_samples(f, input_column, expected_column, system, do_redact)
            continue
        text = f.read_text(encoding="utf-8-sig", errors="replace")
        doc = None
        try:
            doc = json.loads(text)
            records = doc if isinstance(doc, list) else [doc]
        except ValueError:
            records = []
            for ln in text.splitlines():
                ln = ln.strip()
                if ln:
                    try:
                        records.append(json.loads(ln))
                    except ValueError:
                        continue
        if any(isinstance(r, dict) and ("resourceSpans" in r or "spans" in r) for r in records[:5]):
            found += _otel_samples(records, do_redact)
            continue
        if isinstance(doc, dict) and isinstance(doc.get("data"), list):
            records = doc["data"]                                   # Langfuse API pages
        for r in records:
            if isinstance(r, dict) and str(r.get("type", "GENERATION")).upper() not in ("GENERATION", "EMBEDDING"):
                continue
            s = _from_record(r, do_redact)
            if s and system and not any(m["role"] == "system" for m in s.messages):
                s.messages.insert(0, {"role": "system", "content": system})
            if s:
                found.append(s)
    seen, out = set(), []
    for s in found:
        if s.id not in seen:
            seen.add(s.id)
            out.append(s)
    return out


def _csv_samples(f: Path, input_column, expected_column, system, do_redact) -> list[Sample]:
    out = []
    with open(f, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return out
    cols = {c.lower().strip(): c for c in rows[0]}
    inp = input_column or next((cols[c] for c in ("input", "prompt", "question", "text", "message", "query")
                                if c in cols), None)
    exp = expected_column or next((cols[c] for c in ("expected", "label", "answer", "ideal", "output", "target")
                                   if c in cols), None)
    if not inp:
        raise ValueError(f"{f.name}: no input column (use --input-column). Columns: {', '.join(rows[0])}")
    for r in rows:
        s = make_sample(r.get(inp) or "", {}, expected=(r.get(exp) or None) if exp else None, system=system,
                        do_redact=do_redact)
        if s:
            out.append(s)
    return out


def select(samples: list[Sample], n: int | None, seed: int = 7, site: str | None = None) -> list[Sample]:
    """Samples for one site (when given), then a seeded random subset of n."""
    if site:
        path, _, line = site.partition(":")
        pick = [s for s in samples if s.site and (s.site == site or (not line and s.site.split(":")[0] == path))]
        samples = pick or [s for s in samples if not s.site]
    if n and len(samples) > n:
        samples = random.Random(seed).sample(samples, n)
    return samples
