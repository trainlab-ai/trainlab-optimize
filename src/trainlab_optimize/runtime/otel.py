"""For apps that already trace LLM calls with OpenTelemetry (OpenLLMetry, OpenInference,
Langfuse, Logfire, the official GenAI instrumentations, ...): add the code location and
trainlab tags to those spans instead of creating new ones.

    from opentelemetry import trace
    from trainlab_optimize.runtime.otel import CodeLocationSpanProcessor
    trace.get_tracer_provider().add_span_processor(CodeLocationSpanProcessor())

Exported spans then carry code.file.path / code.line.number / code.function.name, and
`trainlab import` on those spans gives cost per line.
"""
from __future__ import annotations

import os
import sys

from .core import _skip_prefixes, current_tags

_GENAI_HINTS = ("gen_ai.", "llm.", "openinference.span.kind")


def _is_genai(span) -> bool:
    attrs = getattr(span, "attributes", None) or {}
    if any(k.startswith(_GENAI_HINTS) for k in attrs):
        return True
    name = (getattr(span, "name", "") or "").lower()
    return name.startswith(("chat ", "embeddings ", "text_completion ", "generate_content", "invoke_agent")) \
        or any(t in name for t in ("openai.chat", "anthropic.chat", "litellm", "chatcompletion"))


try:
    from opentelemetry.sdk.trace import SpanProcessor as _Base
except Exception:  # noqa: BLE001 - OTel SDK not installed: still importable, never used
    _Base = object


class CodeLocationSpanProcessor(_Base):
    def __init__(self, project_root: str | None = None, include_packages: tuple[str, ...] = (),
                 all_spans: bool = False):
        self.root = os.path.realpath(project_root or os.getcwd())
        self._skip = _skip_prefixes(())
        self._include = tuple(include_packages)
        self.all_spans = all_spans

    def _locate(self):
        f = sys._getframe(1)
        while f is not None:
            fn = f.f_code.co_filename
            mod = str(f.f_globals.get("__name__", ""))
            mine = self._include and mod.startswith(self._include)
            if mine or (not fn.startswith(self._skip) and not fn.startswith("<")):
                path = os.path.realpath(fn)
                if path.startswith(self.root + os.sep):
                    path = os.path.relpath(path, self.root).replace(os.sep, "/")
                return path, f.f_lineno, f.f_code.co_name
            f = f.f_back
        return None

    def on_start(self, span, parent_context=None):
        try:
            if not span.is_recording():
                return
            # Many instrumentations set gen_ai.* attributes at start; when they don't, the span
            # name usually says it. all_spans=True tags everything.
            if not (self.all_spans or _is_genai(span)):
                return
            attrs = getattr(span, "attributes", None) or {}
            if "code.file.path" not in attrs and "code.filepath" not in attrs:
                loc = self._locate()
                if loc:
                    span.set_attribute("code.file.path", loc[0])
                    span.set_attribute("code.line.number", loc[1])
                    span.set_attribute("code.function.name", loc[2])
            for k, v in current_tags().items():
                span.set_attribute(f"trainlab.tag.{k}", str(v)[:100])
        except Exception:  # noqa: BLE001 - never break tracing or the app
            pass

    def on_end(self, span):
        pass

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000):
        return True
