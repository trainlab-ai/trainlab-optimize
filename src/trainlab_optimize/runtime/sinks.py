"""Where recorded calls go. A sink that fails is dropped by the recorder; the app is never affected."""
from __future__ import annotations

import atexit
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


class JsonlSink:
    """Appends one JSON line per call to .trainlab/traces/events-YYYYMMDD-<pid>.jsonl.

    Buffered: written every `flush_every` events or `interval` seconds, and at exit.
    One file per process and day, so workers never interleave lines.
    """

    def __init__(self, directory: str | os.PathLike, flush_every: int = 50, interval: float = 2.0):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        gi = self.dir.parent / ".gitignore"
        try:
            if self.dir.parent.name == ".trainlab" and not gi.exists():
                gi.write_text("*\n")
        except OSError:
            pass
        self.flush_every = flush_every
        self.interval = interval
        self._buf: list[str] = []
        self._lock = threading.Lock()
        self._last = time.monotonic()
        atexit.register(self.flush)
        if hasattr(os, "register_at_fork"):
            # A forked worker (gunicorn --preload, multiprocessing) inherits the parent's
            # unwritten lines; the parent writes those, so the child must not.
            os.register_at_fork(after_in_child=self._after_fork)

    def _after_fork(self) -> None:
        self._lock = threading.Lock()
        self._buf = []

    def path(self) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        return self.dir / f"events-{day}-{os.getpid()}.jsonl"

    def write(self, ev) -> None:
        line = json.dumps(ev.to_json(), separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            self._buf.append(line)
            due = len(self._buf) >= self.flush_every or time.monotonic() - self._last >= self.interval
        if due:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            if not self._buf:
                return
            lines, self._buf = self._buf, []
            self._last = time.monotonic()
        with open(self.path(), "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


class CallbackSink:
    """Hand each call to your own function (metrics, logging, a queue)."""

    def __init__(self, fn):
        self.fn = fn

    def write(self, ev) -> None:
        self.fn(ev.to_json())

    def flush(self) -> None:
        pass


class OTelSink:
    """Emits one OpenTelemetry span per call with GenAI semantic-convention attributes plus the
    code location, on the app's configured tracer provider. Use it when no other GenAI
    instrumentation is creating spans (otherwise add CodeLocationSpanProcessor instead)."""

    def __init__(self, tracer_name: str = "trainlab_optimize"):
        from opentelemetry import trace  # noqa: F401 - fails fast when OTel isn't installed
        self.trace = trace
        self.tracer = trace.get_tracer(tracer_name)

    def write(self, ev) -> None:
        end = time.time_ns()
        start = end - int((ev.ms or 0) * 1e6)
        attrs = {
            "gen_ai.operation.name": "embeddings" if ev.op == "embedding" else "chat",
            "gen_ai.provider.name": ev.provider,
            "gen_ai.request.model": ev.model or "",
            "gen_ai.response.model": ev.model or "",
            "code.file.path": ev.path or "",
            "code.line.number": ev.line or 0,
            "code.function.name": ev.func or "",
        }
        for k, v in (("gen_ai.usage.input_tokens", ev.input_tokens), ("gen_ai.usage.output_tokens", ev.output_tokens),
                     ("gen_ai.usage.cache_read.input_tokens", ev.cached_input_tokens)):
            if v is not None:
                attrs[k] = v
        for k, v in ev.tags.items():
            attrs[f"trainlab.tag.{k}"] = v
        span = self.tracer.start_span(f"{attrs['gen_ai.operation.name']} {ev.model or ''}".strip(),
                                      start_time=start, attributes=attrs)
        if not ev.ok:
            from opentelemetry.trace import Status, StatusCode
            span.set_status(Status(StatusCode.ERROR, ev.error or "error"))
            span.set_attribute("error.type", ev.error or "error")
        span.end(end_time=end)

    def flush(self) -> None:
        pass
