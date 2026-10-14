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

    Write-through: each call is handed to the OS as it is recorded, so nothing is lost when a
    server worker exits without running atexit hooks (gunicorn, uvicorn reloads) or is killed.
    One file per process and day, so workers never interleave lines. Cost: one small write
    per LLM call (a few microseconds).
    """

    def __init__(self, directory: str | os.PathLike, flush_every: int | None = None, interval: float | None = None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        gi = self.dir.parent / ".gitignore"
        try:
            if self.dir.parent.name == ".trainlab" and not gi.exists():
                gi.write_text("*\n")
        except OSError:
            pass
        self._lock = threading.Lock()
        self._fh = None
        self._key: tuple | None = None       # (day, pid) the open file belongs to
        atexit.register(self.close)
        if hasattr(os, "register_at_fork"):
            # A forked worker must not write through the parent's file handle.
            os.register_at_fork(after_in_child=self._after_fork)

    def _after_fork(self) -> None:
        self._lock = threading.Lock()
        self._fh, self._key = None, None

    def path(self) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        return self.dir / f"events-{day}-{os.getpid()}.jsonl"

    def write(self, ev) -> None:
        line = json.dumps(ev.to_json(), separators=(",", ":"), ensure_ascii=False) + "\n"
        key = (datetime.now(timezone.utc).strftime("%Y%m%d"), os.getpid())
        with self._lock:
            if self._fh is None or key != self._key:
                if self._fh is not None:
                    try:
                        self._fh.close()
                    except OSError:
                        pass
                self._fh = open(self.dir / f"events-{key[0]}-{key[1]}.jsonl", "a", encoding="utf-8")
                self._key = key
            self._fh.write(line)
            self._fh.flush()

    def flush(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None


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
