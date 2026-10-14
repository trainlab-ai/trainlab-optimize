"""Runtime recorder: one event per LLM call, with the line of the customer's code that made it.

Design rules
- Never break or slow the app: every hook path is wrapped; a failure inside trainlab disables
  that piece and logs one warning; the app's own call, result and exceptions pass through
  unchanged.
- Never store prompt or completion text, keys or headers. Only counts, names and locations.
- One event per logical call: a framework that calls another instrumented SDK (LiteLLM calling
  the OpenAI SDK), or an SDK whose HTTP request the HTTP layer also sees, is recorded once,
  by the outermost hook. Inner layers may only fill in token counts the outer one lacks.
- Work handed to a thread or an asyncio task keeps the app line (and tags) that started it.
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
from dataclasses import dataclass, field
from datetime import datetime, timezone

SCHEMA = 1

_tags: contextvars.ContextVar[dict] = contextvars.ContextVar("trainlab_tags", default={})
# The CallEvent being recorded by the outermost hook in this context (None when no LLM call is
# in progress). Inner hooks see it and don't record a second event.
_active: contextvars.ContextVar = contextvars.ContextVar("trainlab_active", default=None)
SKIP = object()      # _active value for a call that was sampled out: inner layers record nothing
# (path, line, function) of the app line that handed work to the current thread / task.
_origin: contextvars.ContextVar = contextvars.ContextVar("trainlab_origin", default=None)

# The same, per thread: survives libraries that run work inside their own copied context
# (LangChain's ContextThreadPoolExecutor does `copy_context().run(fn)` on the pool thread).
_thread_origin = threading.local()

# Frames of these stdlib files mean "we are inside a thread or event loop started elsewhere".
_BOUNDARY = ("threading.py", "concurrent/futures/thread.py", "asyncio/events.py", "asyncio/base_events.py",
             "asyncio/runners.py", "concurrent\\futures\\thread.py", "asyncio\\events.py", "asyncio\\base_events.py")


@dataclass
class CallEvent:
    """One LLM call. Token fields are None when the provider didn't report them."""
    provider: str
    model: str | None
    op: str                               # chat | embedding | image | other
    sdk: str                              # e.g. openai.chat.completions.create, http:httpx
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
        d = {"v": SCHEMA}
        for k in _FIELDS:
            v = self.__dict__[k]
            if v is None or v is False or v == {}:
                if k in ("ok",):
                    d[k] = v
                continue
            d[k] = dict(v) if k == "tags" else v
        return d


_FIELDS = [f for f in CallEvent.__dataclass_fields__]
_ts_cache = [0, ""]


def _timestamp() -> str:
    """ISO-8601 UTC with milliseconds; the second-level prefix is formatted once per second."""
    now = time.time()
    sec = int(now)
    if sec != _ts_cache[0]:
        _ts_cache[0], _ts_cache[1] = sec, time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(sec))
    return f"{_ts_cache[1]}.{int((now - sec) * 1000):03d}Z"


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
        self._kind: dict[str, int] = {}   # filename -> 0 library, 1 app, 2 thread/loop boundary
        self._relpath: dict[str, str] = {}
        self.counts = {"recorded": 0, "dropped": 0, "errors": 0}

    # -- stack walking
    def _classify(self, fn: str) -> int:
        k = self._kind.get(fn)
        if k is None:
            if fn.endswith(_BOUNDARY) and fn.startswith(self._skip):
                k = 2
            elif fn.startswith(self._skip) or fn.startswith("<"):
                k = 0
            else:
                k = 1
            if len(self._kind) < 20000:
                self._kind[fn] = k
        return k

    def _rel(self, fn: str) -> str:
        path = self._relpath.get(fn)
        if path is None:
            path = os.path.realpath(fn)
            if path.startswith(self.root + os.sep):
                path = os.path.relpath(path, self.root).replace(os.sep, "/")
            if len(self._relpath) < 20000:
                self._relpath[fn] = path
        return path

    def locate(self, depth: int = 2) -> tuple[str | None, int | None, str | None]:
        """The app line responsible for this call: the first stack frame outside trainlab, installed
        packages and the standard library. If the stack first crosses into a thread or event loop
        that was started from somewhere else, the line that started that work is used instead."""
        try:
            f = sys._getframe(depth)
            crossed = False
            include = self._include
            while f is not None:
                fn = f.f_code.co_filename
                k = self._classify(fn)
                if include and k == 0:
                    mod = str(f.f_globals.get("__name__", ""))
                    if mod.startswith(include) and not mod.startswith("trainlab_optimize"):
                        k = 1
                if k == 2:
                    crossed = True
                elif k == 1:
                    origin = _origin.get() or getattr(_thread_origin, "loc", None)
                    if crossed and origin is not None:
                        return origin
                    return self._rel(fn), f.f_lineno, f.f_code.co_name
                f = f.f_back
            origin = _origin.get() or getattr(_thread_origin, "loc", None)
            if origin is not None:
                return origin
        except Exception:  # noqa: BLE001
            self._warn("locate", "could not read the call stack")
        return None, None, None

    def sampled(self) -> bool:
        return self.sample_rate >= 1.0 or random.random() < self.sample_rate

    def emit(self, ev: CallEvent) -> None:
        try:
            ev.ts = ev.ts or _timestamp()
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

    Tags follow the work into threads (ThreadPoolExecutor, threading.Thread) and asyncio tasks.
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


# ------------------------------------------------------------------ handing work to threads / tasks
def carry(recorder_ref, fn, always: bool = False):
    """Wrap fn so that, run in another thread, it keeps the app line and tags of the code that
    handed it over. Returns fn itself when nothing needs carrying (unless always: pool work
    items must not inherit whatever the pool thread was started with)."""
    rec = recorder_ref()
    if rec is None or not rec.enabled:
        return fn
    try:
        loc = rec.locate(depth=3)
    except Exception:  # noqa: BLE001
        loc = None
    tags = _tags.get()
    if (loc is None or loc[0] is None) and not tags and not always:
        return fn

    here = loc if loc and loc[0] else None

    def carried(*a, **kw):
        t1 = _origin.set(here)
        t2 = _tags.set(tags)
        prev = getattr(_thread_origin, "loc", None)
        _thread_origin.loc = here
        try:
            return fn(*a, **kw)
        finally:
            _thread_origin.loc = prev
            _tags.reset(t2)
            _origin.reset(t1)
    carried.__wrapped__ = fn
    return carried


def origin_context(recorder_ref, context=None):
    """A context for a new asyncio task that remembers the app line creating it."""
    rec = recorder_ref()
    if rec is None or not rec.enabled:
        return context
    try:
        loc = rec.locate(depth=3)
    except Exception:  # noqa: BLE001
        return context
    if not loc or loc[0] is None:
        return context
    ctx = context if context is not None else contextvars.copy_context()
    ctx.run(_origin.set, loc)
    return ctx


# ------------------------------------------------------------------ call wrapping
def wrap_call(recorder_ref, orig, describe, extract, *, wrap_stream=None, is_stream=None, sdk_name: str):
    """Wrap one SDK method.

    describe(args, kwargs) -> (provider, model, op), or None for calls that aren't LLM calls
    extract(result, event)                                fills tokens/model from the result
    is_stream(args, kwargs, result) -> bool; wrap_stream(result, event, finish) -> proxy
    """
    if inspect.iscoroutinefunction(orig):
        @functools.wraps(orig)
        async def awrapper(*args, **kwargs):
            rec = recorder_ref()
            if rec is None or not rec.enabled or _active.get() is not None:
                return await orig(*args, **kwargs)
            d = _describe(describe, args, kwargs)
            if d is None:                            # not an LLM call (e.g. S3 via botocore)
                return await orig(*args, **kwargs)
            if not rec.sampled():
                token = _active.set(SKIP)
                try:
                    return _quiet(await orig(*args, **kwargs), args, kwargs, is_stream)
                finally:
                    _active.reset(token)
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
        if rec is None or not rec.enabled or _active.get() is not None:
            return orig(*args, **kwargs)
        d = _describe(describe, args, kwargs)
        if d is None:
            return orig(*args, **kwargs)
        if not rec.sampled():
            token = _active.set(SKIP)
            try:
                return _quiet(orig(*args, **kwargs), args, kwargs, is_stream)
            finally:
                _active.reset(token)
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


class _Quiet:
    """A sampled-out stream: requests it sends lazily (first item, `with` entry) stay unrecorded."""

    def __init__(self, inner):
        self.__dict__["_inner"] = inner

    def _run(self, fn, *a):
        token = _active.set(SKIP)
        try:
            return fn(*a)
        finally:
            _active.reset(token)

    def __iter__(self):
        self.__dict__["_it"] = iter(self._inner)
        return self

    def __next__(self):
        it = self.__dict__.get("_it") or self.__iter__()._it
        return self._run(next, it)

    def __aiter__(self):
        self.__dict__["_ait"] = self._inner.__aiter__()
        return self

    async def __anext__(self):
        it = self.__dict__.get("_ait") or self.__aiter__()._ait
        token = _active.set(SKIP)
        try:
            return await it.__anext__()
        finally:
            _active.reset(token)

    def __enter__(self):
        r = self._run(self._inner.__enter__)
        return self if r is self._inner else r

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)

    async def __aenter__(self):
        token = _active.set(SKIP)
        try:
            r = await self._inner.__aenter__()
        finally:
            _active.reset(token)
        return self if r is self._inner else r

    async def __aexit__(self, *exc):
        return await self._inner.__aexit__(*exc)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _quiet(result, args, kwargs, is_stream):
    try:
        if is_stream is not None and is_stream(args, kwargs, result) and not isinstance(result, dict):
            return _Quiet(result)
    except Exception:  # noqa: BLE001
        pass
    return result


def _describe(describe, args, kwargs):
    try:
        return describe(args, kwargs)
    except Exception:  # noqa: BLE001
        return ("unknown", None, "chat")


def _begin(rec: Recorder, d, sdk_name, depth: int = 3):
    provider, model, op = d
    try:
        path, line, func = rec.locate(depth=depth)
    except Exception:  # noqa: BLE001
        path = line = func = None
    ev = CallEvent(provider=provider, model=model, op=op, sdk=sdk_name, path=path, line=line, func=func,
                   tags=dict(_tags.get()))            # captured now: a stream may finish outside the tag block
    token = _active.set(ev)
    return ev, time.perf_counter(), token


async def _await_then(rec, ev, t0, coro, args, kwargs, extract, is_stream, wrap_stream):
    token = _active.set(ev)
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


def fill(ev: CallEvent, **values) -> None:
    """Set token fields that are still unknown (used by inner layers, e.g. the HTTP layer)."""
    for k, v in values.items():
        if v is not None and getattr(ev, k, None) is None:
            setattr(ev, k, v)


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
        snapshot = (ev.input_tokens, ev.output_tokens, ev.cached_input_tokens, ev.cache_write_tokens,
                    ev.reasoning_tokens)
        extract(result, ev)
        body = ev.__dict__.pop("_http_body", None)
        if body is not None and (ev.input_tokens is None or ev.output_tokens is None):
            try:
                from .http import fill_from_body
                fill_from_body(ev, body)
            except Exception:  # noqa: BLE001
                pass
        # The SDK result is authoritative; values the HTTP layer saw fill only what it left empty.
        for name, before in zip(("input_tokens", "output_tokens", "cached_input_tokens", "cache_write_tokens",
                                 "reasoning_tokens"), snapshot):
            if getattr(ev, name) is None and before is not None:
                setattr(ev, name, before)
        ev.ms = round((time.perf_counter() - t0) * 1000, 1)
        if ev.__dict__.get("_defer"):
            ev._t0 = t0                       # recorded by the HTTP layer when the body has been read
            return result
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
