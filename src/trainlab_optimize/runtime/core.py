"""Runtime recorder: one event per LLM call, with the line of the customer's code that made it.

Design rules
- Never break or slow the app: every hook path is wrapped; a failure inside trainlab disables
  that piece and logs one warning; the app's own call, result and exceptions pass through
  unchanged.
- Never store prompt or completion text, keys or headers. Only counts, names and locations.
- One event per logical call: a framework that calls another instrumented SDK (LiteLLM calling
  the OpenAI SDK) is recorded once, by the outermost hook.
"""
from __future__ import annotations

import contextvars
import functools
import inspect
import os
import random
import sys
import sysconfig
import threading
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

SCHEMA = 1

_tags: contextvars.ContextVar[dict] = contextvars.ContextVar("trainlab_tags", default={})
_active: contextvars.ContextVar[bool] = contextvars.ContextVar("trainlab_active", default=False)


@dataclass
class CallEvent:
    """One LLM call. Token fields are None when the provider didn't report them."""
    provider: str
    model: str | None
    op: str                               # chat | embedding | image | other
    sdk: str                              # e.g. openai.chat.completions.create
    ts: str = ""
    input_tokens: int | None = None       # total input, cached included
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    input_tokens_estimated: bool = False   # streams without usage: counted from request text length
    output_tokens_estimated: bool = False  # streams without usage: counted from text length
    stream_incomplete: bool = False       # the app stopped reading before the end: usage may be missing
    stream: bool = False
    batch: bool = False
    ms: float | None = None
    ok: bool = True
    error: str | None = None              # exception type name only, never its message
    path: str | None = None
    line: int | None = None
    func: str | None = None
    service: str | None = None
    tags: dict = field(default_factory=dict)
    weight: float = 1.0                   # 1 / sample rate

    def to_json(self) -> dict:
        d = asdict(self)
        d["v"] = SCHEMA
        return {k: v for k, v in d.items() if v not in (None, {}, False) or k in ("ok", "provider", "op")}


class Recorder:
    def __init__(self, root: str, service: str | None, tags: dict, sinks: list, sample_rate: float,
                 extra_skip: tuple[str, ...] = (), include_packages: tuple[str, ...] = ()):
        self.root = os.path.realpath(root)
        self.service = service
        self.base_tags = dict(tags or {})
        self.sinks = list(sinks)
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.enabled = True
        self._warned: set[str] = set()
        self._skip = _skip_prefixes(extra_skip)
        # The app's own packages count as "user code" even when installed into site-packages
        self._include = tuple(f"{p}." for p in include_packages) + tuple(include_packages)
        self._lock = threading.Lock()
        self.counts = {"recorded": 0, "dropped": 0, "errors": 0}

    # -- public helpers used by patches
    def locate(self) -> tuple[str | None, int | None, str | None]:
        """First stack frame outside trainlab, installed packages and the standard library."""
        try:
            f = sys._getframe(2)
            skip = self._skip
            while f is not None:
                fn = f.f_code.co_filename
                mine = bool(self._include) and str(f.f_globals.get("__name__", "")).startswith(self._include) \
                    and not str(f.f_globals.get("__name__", "")).startswith("trainlab_optimize")
                if mine or (not fn.startswith(skip) and not fn.startswith("<frozen")):
                    path = os.path.realpath(fn) if not fn.startswith("<") else fn
                    if path.startswith(self.root + os.sep):
                        path = os.path.relpath(path, self.root).replace(os.sep, "/")
                    return path, f.f_lineno, f.f_code.co_name
                f = f.f_back
        except Exception:  # noqa: BLE001
            self._warn("locate", "could not read the call stack")
        return None, None, None

    def sampled(self) -> bool:
        return self.sample_rate >= 1.0 or random.random() < self.sample_rate

    def emit(self, ev: CallEvent) -> None:
        try:
            ev.ts = ev.ts or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            ev.service = ev.service or self.service
            merged = {**self.base_tags, **ev.tags}
            ev.tags = {str(k): str(v)[:100] for k, v in list(merged.items())[:20]}
            ev.weight = 1.0 / self.sample_rate if 0 < self.sample_rate < 1 else 1.0
            for s in list(self.sinks):
                try:
                    s.write(ev)
                except Exception as exc:  # noqa: BLE001
                    self._warn(f"sink:{type(s).__name__}", f"{type(s).__name__} failed ({type(exc).__name__}); disabled")
                    self.sinks.remove(s)
            self.counts["recorded"] += 1
        except Exception:  # noqa: BLE001
            self.counts["errors"] += 1
            self._warn("emit", "could not record a call")

    def flush(self) -> None:
        for s in list(self.sinks):
            try:
                s.flush()
            except Exception:  # noqa: BLE001
                pass

    def _warn(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            warnings.warn(f"trainlab instrument: {msg}. The app is unaffected.", RuntimeWarning, stacklevel=3)


def _skip_prefixes(extra: tuple[str, ...]) -> tuple[str, ...]:
    paths = set()
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        try:
            p = sysconfig.get_paths().get(key)
            if p:
                paths.add(os.path.realpath(p))
                paths.add(p)
        except Exception:  # noqa: BLE001
            pass
    for p in sys.path:
        if p and ("site-packages" in p or "dist-packages" in p):
            paths.add(p)
            paths.add(os.path.realpath(p))
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # trainlab_optimize/
    paths.add(here)
    paths.add(os.path.realpath(here))
    paths.update(extra)
    return tuple(sorted(paths, key=len, reverse=True))


# ------------------------------------------------------------------ tags
class tag:  # noqa: N801 - used like a function: with trainlab.tag(feature="x"): / @trainlab.tag(...)
    """Attach tags (feature, team, customer tier, ...) to every LLM call made inside.

        with trainlab.tag(feature="invoice-extract", team="billing"):
            client.chat.completions.create(...)

        @trainlab.tag(feature="support-bot")
        async def answer(q): ...
    """

    def __init__(self, **tags):
        self.tags = tags
        self._tokens: list = []

    def __enter__(self):
        self._tokens.append(_tags.set({**_tags.get(), **self.tags}))
        return self

    def __exit__(self, *exc):
        _tags.reset(self._tokens.pop())
        return False

    def __call__(self, fn):
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(*a, **kw):
                with tag(**self.tags):
                    return await fn(*a, **kw)
            return awrapper

        @functools.wraps(fn)
        def wrapper(*a, **kw):
            with tag(**self.tags):
                return fn(*a, **kw)
        return wrapper


def current_tags() -> dict:
    return dict(_tags.get())


# ------------------------------------------------------------------ call wrapping
def wrap_call(recorder_ref, orig, describe, extract, *, wrap_stream=None, is_stream=None, sdk_name: str):
    """Wrap one SDK method.

    describe(args, kwargs) -> (provider, model, op)       before the call
    extract(result, event)                                fills tokens/model from the result
    is_stream(args, kwargs, result) -> bool; wrap_stream(result, event, finish) -> proxy
    """
    if inspect.iscoroutinefunction(orig):
        @functools.wraps(orig)
        async def awrapper(*args, **kwargs):
            rec = recorder_ref()
            if rec is None or not rec.enabled or _active.get():
                return await orig(*args, **kwargs)
            d = _describe(describe, args, kwargs)
            if d is None or not rec.sampled():       # not an LLM call (e.g. S3 via botocore), or sampled out
                return await orig(*args, **kwargs)
            ev, t0, token = _begin(rec, d, sdk_name)
            try:
                result = await orig(*args, **kwargs)
            except BaseException as exc:
                _active.reset(token)
                _fail(rec, ev, t0, exc)
                raise
            _active.reset(token)
            return _finish_or_stream(rec, ev, t0, result, args, kwargs, extract, is_stream, wrap_stream)
        awrapper.__trainlab_original__ = orig
        return awrapper

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        rec = recorder_ref()
        if rec is None or not rec.enabled or _active.get():
            return orig(*args, **kwargs)
        d = _describe(describe, args, kwargs)
        if d is None or not rec.sampled():
            return orig(*args, **kwargs)
        ev, t0, token = _begin(rec, d, sdk_name)
        try:
            result = orig(*args, **kwargs)
        except BaseException as exc:
            _active.reset(token)
            _fail(rec, ev, t0, exc)
            raise
        _active.reset(token)
        if inspect.iscoroutine(result):
            # async method hidden behind a sync decorator (the OpenAI SDK does this): the real
            # call happens when the app awaits, so record then.
            return _await_then(rec, ev, t0, result, args, kwargs, extract, is_stream, wrap_stream)
        return _finish_or_stream(rec, ev, t0, result, args, kwargs, extract, is_stream, wrap_stream)
    wrapper.__trainlab_original__ = orig
    return wrapper


def _describe(describe, args, kwargs):
    try:
        return describe(args, kwargs)
    except Exception:  # noqa: BLE001
        return ("unknown", None, "chat")


def _begin(rec: Recorder, d, sdk_name):
    token = _active.set(True)
    provider, model, op = d
    try:
        path, line, func = rec.locate()
    except Exception:  # noqa: BLE001
        path = line = func = None
    ev = CallEvent(provider=provider, model=model, op=op, sdk=sdk_name, path=path, line=line, func=func,
                   tags=dict(_tags.get()))            # captured now: a stream may finish outside the tag block
    return ev, time.perf_counter(), token


async def _await_then(rec, ev, t0, coro, args, kwargs, extract, is_stream, wrap_stream):
    token = _active.set(True)
    try:
        result = await coro
    except BaseException as exc:
        _active.reset(token)
        _fail(rec, ev, t0, exc)
        raise
    _active.reset(token)
    return _finish_or_stream(rec, ev, t0, result, args, kwargs, extract, is_stream, wrap_stream)


def _fail(rec: Recorder, ev: CallEvent, t0: float, exc: BaseException) -> None:
    try:
        ev.ok, ev.error, ev.ms = False, type(exc).__name__, round((time.perf_counter() - t0) * 1000, 1)
        rec.emit(ev)
    except Exception:  # noqa: BLE001
        pass


def _finish_or_stream(rec, ev, t0, result, args, kwargs, extract, is_stream, wrap_stream):
    try:
        if is_stream is not None and wrap_stream is not None and is_stream(args, kwargs, result):
            ev.stream = True
            ev._req = kwargs          # kept until the stream ends, for input estimates; never stored

            def finish(error: BaseException | None = None):
                ev.ms = round((time.perf_counter() - t0) * 1000, 1)
                if error is not None:
                    ev.ok, ev.error = False, type(error).__name__
                rec.emit(ev)
            return wrap_stream(result, ev, finish)
        extract(result, ev)
        ev.ms = round((time.perf_counter() - t0) * 1000, 1)
        rec.emit(ev)
    except Exception:  # noqa: BLE001
        rec._warn(f"extract:{ev.sdk}", f"could not read usage from {ev.sdk}")
    return result


def num(x) -> int | None:
    try:
        return int(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def get(obj, *names, default=None):
    """obj.a.b or obj['a']['b'], whichever exists."""
    cur = obj
    for n in names:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(n)
        else:
            cur = getattr(cur, n, None)
    return default if cur is None else cur
