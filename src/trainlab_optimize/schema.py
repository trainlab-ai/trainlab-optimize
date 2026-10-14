"""Unified workload schema.

Every adapter (scanner, importers) converts its source into these records.
Analyzer and report only ever see these types, never a third-party format.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


def _from_dict(cls, data: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class CallSite:
    """One place in source code that calls (or configures) an LLM."""

    id: str                      # stable: sha1(path:line:kind)[:12]
    path: str                    # relative to scan root
    line: int
    kind: str                    # chat | embedding | batch | model_config | model_reference
    provider: str                # openai | anthropic | azure | bedrock | gemini | litellm | openai_compatible | unknown
    sdk_call: str                # e.g. client.chat.completions.create
    language: str = "python"
    function: str | None = None
    cell: int | None = None      # notebook cell number (1-based) when path is an .ipynb
    repo: str | None = None      # repository name when several repositories are analysed together
    model: str | None = None     # resolved literal model name, if known
    model_expr: str | None = None  # source expression when model is not a literal
    model_source: str = "unknown"  # literal | constant | env_default | unresolved | missing
    params: dict[str, Any] = field(default_factory=dict)  # max_tokens, temperature, stream, response_format...
    in_loop: bool = False
    loop_kind: str | None = None  # for | comprehension | while
    scope: str = "app"            # app | example (examples/, docs/, notebooks/, benchmarks/ ...)
    is_async: bool = False
    retry_unbounded: bool = False
    has_retry: bool = False
    prompt_fingerprint: str | None = None
    system_prompt_tokens: int | None = None   # estimated static tokens
    prompt_static_tokens: int | None = None   # estimated static tokens across all literal parts
    prompt_dynamic_prefix: bool = False       # interpolation appears before most static text
    prompt_preview: str | None = None         # redacted, truncated
    task_hints: list[str] = field(default_factory=list)  # extraction | classification | summarization | structured_output | ...
    uses_cache_control: bool = False
    is_test: bool = False
    confidence: str = "high"     # high (AST) | heuristic (regex)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def location(self) -> str:
        loc = f"{self.path}:cell {self.cell}:{self.line}" if self.cell else f"{self.path}:{self.line}"
        return f"{self.repo}/{loc}" if self.repo else loc

    @classmethod
    def from_dict(cls, d: dict) -> "CallSite":
        return _from_dict(cls, d)


@dataclass
class UsageRecord:
    """Aggregated usage for one model over one period (usually a day)."""

    date: str                    # YYYY-MM-DD (bucket start)
    provider: str
    model: str
    input_tokens: int = 0        # total input incl. cached
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    requests: int = 0
    cost_usd: float | None = None   # as billed, if the source provides it
    batch: bool | None = None
    project: str | None = None      # project / workspace / api key for team attribution
    source: str = "unknown"
    code_path: str | None = None    # from traces that carry a source location (exact attribution)
    code_line: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "UsageRecord":
        return _from_dict(cls, d)


@dataclass
class Finding:
    id: str
    rule: str
    title: str
    severity: str                # high | medium | low | info
    strategy: str                # model | cache | batch | prompt | config | slm | reliability | coverage
    call_site_ids: list[str]
    summary: str
    recommendation: str
    evidence: dict[str, Any] = field(default_factory=dict)
    tier: str = "static"         # static | estimated | validated | realized
    monthly_spend_usd: float | None = None
    savings_low_usd: float | None = None
    savings_high_usd: float | None = None
    trainlab_candidate: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Finding":
        return _from_dict(cls, d)


@dataclass
class Attribution:
    call_site_id: str
    model: str
    monthly_spend_usd: float
    monthly_requests: float
    method: str                  # exact_trace | exact_model | split_model | unattributed

    def to_dict(self) -> dict:
        return asdict(self)
