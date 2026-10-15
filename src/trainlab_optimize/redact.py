"""Local redaction of secrets and PII before anything is stored or shown in a report."""
from __future__ import annotations

import re

_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"), "<ANTHROPIC_KEY>"),
    (re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{16,}"), "<OPENAI_KEY>"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "<AWS_KEY>"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"), "Bearer <TOKEN>"),
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "<EMAIL>"),
    (re.compile(r"\b(?:\d[ \-]?){13,16}\b"), "<CARD>"),
    (re.compile(r"(?<!\w)\+?\d[\d \-().]{8,}\d(?!\w)"), "<PHONE>"),
]


def redact(text: str | None, max_len: int | None = None, one_line: bool | None = None) -> str | None:
    """Replace secrets and PII with placeholders. Whitespace is kept (prompts are replayed in
    experiments, and their layout matters to the model) unless one_line, which is the default
    when max_len is given, for previews in reports."""
    if text is None:
        return None
    out = text
    for pat, repl in _PATTERNS:
        out = pat.sub(repl, out)
    if one_line if one_line is not None else max_len is not None:
        out = " ".join(out.split())
    if max_len and len(out) > max_len:
        out = out[: max_len - 1] + "…"
    return out
