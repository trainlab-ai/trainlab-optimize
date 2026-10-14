"""SDK hooks. Each wraps one public method; the original is kept for uninstrument().

Covered: OpenAI SDK (and every provider used through it: Azure, OpenRouter, DashScope,
DeepSeek, Groq, Together, Moonshot, Zhipu, xAI, ...), Anthropic SDK (incl. Bedrock/Vertex
clients), LiteLLM, AWS Bedrock Runtime via botocore, Google Gen AI SDK (Gemini / Vertex).
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import json
import math
import sys

from ..scanner.providers import maker_of, provider_from_model_prefix, provider_from_url
from .core import _active, carry, get, num, origin_context, wrap_call

_applied: list[tuple[object, str, object]] = []      # (owner, attribute, original)
patched: set[str] = set()


# ------------------------------------------------------------------ stream proxies
class _StreamProxy:
    """Transparent wrapper over a sync stream/iterator: same items, same methods; reads usage
    from items as the app consumes them and records once when the stream ends or is closed."""

    def __init__(self, inner, ev, finish, on_item, on_end=None):
        self.__dict__.update(_inner=inner, _ev=ev, _finish=finish, _on_item=on_item, _on_end=on_end,
                             _it=None, _done=False)

    def _close(self, error=None, complete=False):
        if self._done:
            return
        self.__dict__["_done"] = True
        if not complete and error is None:
            self._ev.stream_incomplete = True
        try:
            if self._on_end:
                self._on_end(self._inner, self._ev)
            _estimate_output(self._ev)
        except Exception:  # noqa: BLE001
            pass
        self._finish(error)

    def __iter__(self):
        return self

    def __next__(self):
        token = _active.set(self._ev)        # a lazily sent request belongs to this call
        try:
            if self._it is None:
                self.__dict__["_it"] = iter(self._inner)
            item = next(self._it)
        except StopIteration:
            self._close(complete=True)
            raise
        except BaseException as exc:
            self._close(exc)
            raise
        finally:
            _active.reset(token)
        try:
            self._on_item(item, self._ev)
        except Exception:  # noqa: BLE001
            pass
        return item

    def __enter__(self):
        if hasattr(self._inner, "__enter__"):
            token = _active.set(self._ev)
            try:
                self._inner.__enter__()
            finally:
                _active.reset(token)
        return self

    def __exit__(self, *exc):
        try:
            return self._inner.__exit__(*exc) if hasattr(self._inner, "__exit__") else False
        finally:
            self._close(exc[1] if exc and exc[1] else None)

    def close(self):
        try:
            if hasattr(self._inner, "close"):
                return self._inner.close()
        finally:
            self._close()

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __del__(self):
        try:
            self._close()
        except Exception:  # noqa: BLE001
            pass


class _AsyncStreamProxy(_StreamProxy):
    def __aiter__(self):
        return self

    async def __anext__(self):
        token = _active.set(self._ev)
        try:
            if self._it is None:
                self.__dict__["_it"] = self._inner.__aiter__()
            item = await self._it.__anext__()
        except StopAsyncIteration:
            self._close(complete=True)
            raise
        except BaseException as exc:
            self._close(exc)
            raise
        finally:
            _active.reset(token)
        try:
            self._on_item(item, self._ev)
        except Exception:  # noqa: BLE001
            pass
        return item

    async def __aenter__(self):
        if hasattr(self._inner, "__aenter__"):
            token = _active.set(self._ev)
            try:
                await self._inner.__aenter__()
            finally:
                _active.reset(token)
        return self

    async def __aexit__(self, *exc):
        try:
            return await self._inner.__aexit__(*exc) if hasattr(self._inner, "__aexit__") else False
        finally:
            self._close(exc[1] if exc and exc[1] else None)

    async def aclose(self):
        try:
            if hasattr(self._inner, "aclose"):
                return await self._inner.aclose()
            if hasattr(self._inner, "close"):
                r = self._inner.close()
                if hasattr(r, "__await__"):
                    await r
        finally:
            self._close()


def _proxy(result, ev, finish, on_item, on_end=None):
    cls = _AsyncStreamProxy if hasattr(result, "__aiter__") and not hasattr(result, "__iter__") else _StreamProxy
    return cls(result, ev, finish, on_item, on_end)


def _text_chars(x, depth=0) -> int:
    """Characters of text in a request (messages / input / contents); counted, never kept."""
    if depth > 6 or x is None:
        return 0
    if isinstance(x, str):
        return len(x)
    if isinstance(x, dict):
        return sum(_text_chars(v, depth + 1) for k, v in x.items() if k in ("content", "text", "input", "parts"))
    if isinstance(x, (list, tuple)):
        return sum(_text_chars(v, depth + 1) for v in x)
    return 0


def _estimate_output(ev) -> None:
    chars = getattr(ev, "_chars", 0)
    if ev.output_tokens is None and chars:
        ev.output_tokens = math.ceil(chars / 4)
        ev.output_tokens_estimated = True
    req = ev.__dict__.pop("_req", None)
    if ev.input_tokens is None and req:
        n = sum(_text_chars(req.get(k)) for k in ("messages", "input", "system", "contents"))
        if n:
            ev.input_tokens = math.ceil(n / 4) + 4 * len(req.get("messages") or [])
            ev.input_tokens_estimated = True


def _add_chars(ev, text) -> None:
    if text:
        ev._chars = getattr(ev, "_chars", 0) + len(text)
        cap = ev.__dict__.get("_cap_text")
        if cap is not None:
            cap.append(text)


# ------------------------------------------------------------------ raw-response wrappers
def _raw_kind(result) -> str | None:
    """Raw-response wrappers (`.with_raw_response`, `.with_streaming_response`, and their
    variants across SDK versions). If the body has been read, parse() is cached and safe to
    call ("legacy"); if not, it must not be touched ("unread") - the HTTP layer reads the usage
    as the app consumes the body."""
    n = type(result).__name__
    if "APIResponse" not in n or not hasattr(result, "parse"):
        return None
    hr = getattr(result, "http_response", None)
    if hr is not None and getattr(hr, "_content", None) is not None:
        return "legacy"
    return "unread"


def _raw_aware(extract, is_stream):
    def ex(result, ev):
        kind = _raw_kind(result)
        if kind == "unread":
            ev._defer = True                  # the HTTP layer records it once the app has read the body
            return
        if kind == "legacy":
            if ev.stream:
                return
            result = result.parse()
        extract(result, ev)

    def st(args, kwargs, result):
        if _raw_kind(result):
            return False
        return is_stream(args, kwargs, result)
    return ex, st


# ------------------------------------------------------------------ OpenAI SDK
_provider_cache: dict[str, str] = {}


def _openai_provider(resource) -> str:
    client = getattr(resource, "_client", None)
    if client is None:
        return "openai"
    if "Azure" in type(client).__name__:
        return "azure"
    base = str(getattr(client, "base_url", "") or "")
    p = _provider_cache.get(base)
    if p is None:
        p = provider_from_url(base) or ("openai" if not base or "api.openai.com" in base else "openai_compatible")
        if len(_provider_cache) < 1000:
            _provider_cache[base] = p
    return p


def _openai_chat_usage(u, ev) -> None:
    if u is None:
        return
    d = u if isinstance(u, dict) else getattr(u, "__dict__", None)
    if isinstance(d, dict) and not isinstance(u, dict):
        # pydantic models: read fields from __dict__ (missing attributes are slow to probe)
        u = {k: (v.__dict__ if hasattr(v, "__dict__") and not isinstance(v, (int, str)) else v)
             for k, v in d.items()}
    ev.input_tokens = num(get(u, "prompt_tokens")) if get(u, "prompt_tokens") is not None else num(get(u, "input_tokens"))
    ev.output_tokens = num(get(u, "completion_tokens")) if get(u, "completion_tokens") is not None \
        else num(get(u, "output_tokens"))
    ev.cached_input_tokens = num(get(u, "prompt_tokens_details", "cached_tokens")) or \
        num(get(u, "input_tokens_details", "cached_tokens"))
    ev.reasoning_tokens = num(get(u, "completion_tokens_details", "reasoning_tokens")) or \
        num(get(u, "output_tokens_details", "reasoning_tokens"))
    cw = num(get(u, "cache_creation_input_tokens"))      # LiteLLM reports Anthropic cache writes here
    if cw:
        ev.cache_write_tokens = cw


def _openai_specs():
    try:
        from openai.resources.chat import completions as chat
        from openai.resources import embeddings, responses
    except Exception:  # noqa: BLE001
        return []

    def describe(op):
        def d(args, kwargs):
            return _openai_provider(args[0]), kwargs.get("model"), op
        return d

    def extract(result, ev):
        ev.model = get(result, "model") or ev.model
        _openai_chat_usage(get(result, "usage"), ev)

    def is_stream(args, kwargs, result):
        return kwargs.get("stream") is True

    def chat_item(item, ev):
        ev.model = get(item, "model") or ev.model
        if get(item, "usage") is not None:
            _openai_chat_usage(get(item, "usage"), ev)
        for ch in get(item, "choices", default=[]) or []:
            _add_chars(ev, get(ch, "delta", "content"))

    def responses_item(item, ev):
        t = get(item, "type", default="")
        if t == "response.output_text.delta":
            _add_chars(ev, get(item, "delta"))
        resp = get(item, "response")
        if resp is not None:
            ev.model = get(resp, "model") or ev.model
            if get(resp, "usage") is not None:
                _openai_chat_usage(get(resp, "usage"), ev)

    def stream_wrapper(on_item):
        return lambda result, ev, finish: _proxy(result, ev, finish, on_item)

    specs = []
    for owner in (chat.Completions, chat.AsyncCompletions):
        for name in ("create", "parse"):
            if hasattr(owner, name):
                specs.append((owner, name, dict(describe=describe("chat"), extract=extract, is_stream=is_stream,
                                                wrap_stream=stream_wrapper(chat_item),
                                                sdk_name=f"openai.chat.completions.{name}")))
    for owner in (responses.Responses, responses.AsyncResponses):
        for name in ("create", "parse"):
            if hasattr(owner, name):
                specs.append((owner, name, dict(describe=describe("chat"), extract=extract, is_stream=is_stream,
                                                wrap_stream=stream_wrapper(responses_item),
                                                sdk_name=f"openai.responses.{name}")))
    try:
        from openai.resources import completions as legacy

        def text_item(item, ev):
            ev.model = get(item, "model") or ev.model
            if get(item, "usage") is not None:
                _openai_chat_usage(get(item, "usage"), ev)
            for ch in get(item, "choices", default=[]) or []:
                _add_chars(ev, get(ch, "text"))
        for owner in (legacy.Completions, legacy.AsyncCompletions):
            specs.append((owner, "create", dict(describe=describe("chat"), extract=extract, is_stream=is_stream,
                                                wrap_stream=stream_wrapper(text_item),
                                                sdk_name="openai.completions.create")))
    except Exception:  # noqa: BLE001
        pass
    for owner in (embeddings.Embeddings, embeddings.AsyncEmbeddings):
        specs.append((owner, "create", dict(describe=describe("embedding"), extract=extract,
                                            sdk_name="openai.embeddings.create")))
    return specs


# ------------------------------------------------------------------ Anthropic SDK
def _anthropic_provider(resource) -> str:
    client = getattr(resource, "_client", None)
    name = type(client).__name__ if client is not None else ""
    if "Bedrock" in name:
        return "bedrock"
    if "Vertex" in name:
        return "vertex"
    base = str(getattr(client, "base_url", "") or "")
    return provider_from_url(base) or "anthropic"


def _anthropic_usage(u, ev) -> None:
    if u is None:
        return
    inp, cr, cw = num(get(u, "input_tokens")), num(get(u, "cache_read_input_tokens")), \
        num(get(u, "cache_creation_input_tokens"))
    if inp is not None or cr or cw:
        ev.input_tokens = (inp or 0) + (cr or 0) + (cw or 0)
        ev.cached_input_tokens = cr
        ev.cache_write_tokens = cw
    out = num(get(u, "output_tokens"))
    if out is not None:
        ev.output_tokens = out


def _anthropic_specs():
    try:
        from anthropic.resources import messages as m
    except Exception:  # noqa: BLE001
        return []
    owners = [m.Messages, m.AsyncMessages]
    try:
        from anthropic.resources.beta import messages as bm
        owners += [bm.Messages, bm.AsyncMessages]
    except Exception:  # noqa: BLE001
        pass

    def describe(args, kwargs):
        return _anthropic_provider(args[0]), kwargs.get("model"), "chat"

    def extract(result, ev):
        ev.model = get(result, "model") or ev.model
        _anthropic_usage(get(result, "usage"), ev)

    def item(event, ev):
        t = get(event, "type", default="")
        if t == "message_start":
            ev.model = get(event, "message", "model") or ev.model
            _anthropic_usage(get(event, "message", "usage"), ev)
        elif t == "message_delta":
            out = num(get(event, "usage", "output_tokens"))
            if out is not None:
                ev.output_tokens = out
        elif t == "content_block_delta":
            _add_chars(ev, get(event, "delta", "text"))

    def stream_manager(result, ev, finish):
        """messages.stream(): a context manager yielding a MessageStream."""
        return _ManagerProxy(result, ev, finish)

    specs = []
    for owner in owners:
        sdk = "anthropic.beta.messages" if "beta" in owner.__module__ else "anthropic.messages"
        specs.append((owner, "create", dict(describe=describe, extract=extract,
                                            is_stream=lambda a, k, r: k.get("stream") is True,
                                            wrap_stream=lambda r, ev, fin: _proxy(r, ev, fin, item),
                                            sdk_name=f"{sdk}.create")))
        if hasattr(owner, "stream"):
            specs.append((owner, "stream", dict(describe=describe, extract=extract,
                                                is_stream=lambda a, k, r: True, wrap_stream=stream_manager,
                                                sdk_name=f"{sdk}.stream")))
    return specs


class _ManagerProxy:
    """Anthropic MessageStreamManager / AsyncMessageStreamManager: record usage from the final
    message snapshot when the `with` block ends."""

    def __init__(self, inner, ev, finish):
        self.__dict__.update(_inner=inner, _ev=ev, _finish=finish, _stream=None, _done=False)

    def _record(self, exc=None):
        if self._done:
            return
        self.__dict__["_done"] = True
        try:
            snap = getattr(self._stream, "current_message_snapshot", None)
            if snap is not None:
                self._ev.model = get(snap, "model") or self._ev.model
                _anthropic_usage(get(snap, "usage"), self._ev)
        except Exception:  # noqa: BLE001
            pass
        self._finish(exc)

    def __enter__(self):
        token = _active.set(self._ev)        # messages.stream() sends its request here
        try:
            self.__dict__["_stream"] = self._inner.__enter__()
        finally:
            _active.reset(token)
        return self._stream

    def __exit__(self, *exc):
        try:
            return self._inner.__exit__(*exc)
        finally:
            self._record(exc[1] if exc and exc[1] else None)

    async def __aenter__(self):
        token = _active.set(self._ev)
        try:
            self.__dict__["_stream"] = await self._inner.__aenter__()
        finally:
            _active.reset(token)
        return self._stream

    async def __aexit__(self, *exc):
        try:
            return await self._inner.__aexit__(*exc)
        finally:
            self._record(exc[1] if exc and exc[1] else None)

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ------------------------------------------------------------------ LiteLLM
def _litellm_specs():
    try:
        import litellm
        import litellm.main as lm
    except Exception:  # noqa: BLE001
        return []

    def describe(op):
        def d(args, kwargs):
            model = kwargs.get("model") or (args[0] if args else None)
            prov = provider_from_model_prefix(model) or maker_of(model) or "litellm"
            return prov, model, op
        return d

    def extract(result, ev):
        ev.model = get(result, "model") or ev.model
        _openai_chat_usage(get(result, "usage"), ev)

    def chunk(item, ev):
        if get(item, "usage") is not None:
            _openai_chat_usage(get(item, "usage"), ev)
        for ch in get(item, "choices", default=[]) or []:
            _add_chars(ev, get(ch, "delta", "content"))

    specs = []
    for owner in (litellm, lm):
        for name, op in (("completion", "chat"), ("acompletion", "chat"), ("embedding", "embedding"),
                         ("aembedding", "embedding")):
            if hasattr(owner, name):
                specs.append((owner, name, dict(describe=describe(op), extract=extract,
                                                is_stream=lambda a, k, r: k.get("stream") is True,
                                                wrap_stream=lambda r, ev, fin: _proxy(r, ev, fin, chunk),
                                                sdk_name=f"litellm.{name}")))
    return specs


# ------------------------------------------------------------------ AWS Bedrock (botocore)
_BEDROCK_OPS = {"Converse": "chat", "ConverseStream": "chat", "InvokeModel": "chat",
                "InvokeModelWithResponseStream": "chat"}


def _bedrock_specs():
    specs = []
    owners = []
    try:
        from botocore.client import BaseClient
        owners.append((BaseClient, "botocore"))
    except Exception:  # noqa: BLE001
        pass
    try:
        from aiobotocore.client import AioBaseClient
        owners.append((AioBaseClient, "aiobotocore"))
    except Exception:  # noqa: BLE001
        pass

    def describe(args, kwargs):
        client, op = args[0], (args[1] if len(args) > 1 else kwargs.get("operation_name"))
        if get(client, "meta", "service_model", "service_name") != "bedrock-runtime" or op not in _BEDROCK_OPS:
            return None                                   # every other AWS call passes straight through
        params = args[2] if len(args) > 2 else kwargs.get("api_params", {})
        return "bedrock", (params or {}).get("modelId"), _BEDROCK_OPS[op]

    def extract(result, ev):
        u = (result or {}).get("usage")
        if u:
            cr, cw = num(u.get("cacheReadInputTokens")), num(u.get("cacheWriteInputTokens"))
            ev.input_tokens = (num(u.get("inputTokens")) or 0) + (cr or 0) + (cw or 0)
            ev.output_tokens = num(u.get("outputTokens"))
            ev.cached_input_tokens, ev.cache_write_tokens = cr, cw
            return
        h = get(result, "ResponseMetadata", "HTTPHeaders", default={}) or {}
        if "x-amzn-bedrock-input-token-count" in h:
            ev.input_tokens = num(h.get("x-amzn-bedrock-input-token-count"))
            ev.output_tokens = num(h.get("x-amzn-bedrock-output-token-count"))

    def is_stream(args, kwargs, result):
        return isinstance(result, dict) and ("stream" in result and args[1] == "ConverseStream"
                                             or args[1] == "InvokeModelWithResponseStream")

    def event(item, ev):
        meta = (item or {}).get("metadata") if isinstance(item, dict) else None
        if meta and meta.get("usage"):
            u = meta["usage"]
            cr, cw = num(u.get("cacheReadInputTokens")), num(u.get("cacheWriteInputTokens"))
            ev.input_tokens = (num(u.get("inputTokens")) or 0) + (cr or 0) + (cw or 0)
            ev.output_tokens = num(u.get("outputTokens"))
            ev.cached_input_tokens, ev.cache_write_tokens = cr, cw
        chunk = (item or {}).get("chunk") if isinstance(item, dict) else None
        if chunk and chunk.get("bytes"):
            try:
                m = json.loads(chunk["bytes"]).get("amazon-bedrock-invocationMetrics")
                if m:
                    ev.input_tokens = num(m.get("inputTokenCount"))
                    ev.output_tokens = num(m.get("outputTokenCount"))
            except (ValueError, AttributeError):
                pass

    def wrap_stream(result, ev, finish):
        key = "stream" if "stream" in result else "body"
        result[key] = _proxy(result[key], ev, finish, event)
        return result

    for owner, lib in owners:
        specs.append((owner, "_make_api_call", dict(describe=describe, extract=extract, is_stream=is_stream,
                                                    wrap_stream=wrap_stream, sdk_name=f"{lib}.bedrock-runtime")))
    return specs


# ------------------------------------------------------------------ Google Gen AI SDK
def _genai_specs():
    try:
        from google.genai import models
    except Exception:  # noqa: BLE001
        return []

    def describe(op):
        def d(args, kwargs):
            api = get(args[0], "_api_client")
            prov = "vertex" if get(api, "vertexai") else "gemini"
            return prov, kwargs.get("model"), op
        return d

    def usage(meta, ev):
        if meta is None:
            return
        ev.input_tokens = num(get(meta, "prompt_token_count"))
        ev.output_tokens = num(get(meta, "candidates_token_count"))
        ev.cached_input_tokens = num(get(meta, "cached_content_token_count"))
        ev.reasoning_tokens = num(get(meta, "thoughts_token_count"))

    def extract(result, ev):
        ev.model = get(result, "model_version") or ev.model
        usage(get(result, "usage_metadata"), ev)

    def item(chunk, ev):
        usage(get(chunk, "usage_metadata"), ev)

    specs = []
    for owner in (models.Models, models.AsyncModels):
        specs.append((owner, "generate_content", dict(describe=describe("chat"), extract=extract,
                                                      sdk_name="google.genai.models.generate_content")))
        if hasattr(owner, "generate_content_stream"):
            specs.append((owner, "generate_content_stream", dict(
                describe=describe("chat"), extract=extract, is_stream=lambda a, k, r: True,
                wrap_stream=lambda r, ev, fin: _proxy(r, ev, fin, item),
                sdk_name="google.genai.models.generate_content_stream")))
        if hasattr(owner, "embed_content"):
            specs.append((owner, "embed_content", dict(describe=describe("embedding"), extract=lambda r, ev: None,
                                                       sdk_name="google.genai.models.embed_content")))
    return specs


# ------------------------------------------------------------------ work handed to threads / tasks
def _threadpool_specs():
    from concurrent.futures import thread as cft

    def factory(recorder_ref, orig):
        def submit(self, fn, /, *args, **kwargs):
            try:
                fn = carry(recorder_ref, fn, always=True)
            except Exception:  # noqa: BLE001
                pass
            return orig(self, fn, *args, **kwargs)
        submit.__trainlab_original__ = orig
        return submit
    return [(cft.ThreadPoolExecutor, "submit", {"factory": factory})]


def _thread_specs():
    import threading

    def factory(recorder_ref, orig):
        def start(self):
            try:
                if not self.__dict__.get("_trainlab_carried"):
                    run = carry(recorder_ref, self.run)
                    if run is not self.run:
                        self.run = run
                    self.__dict__["_trainlab_carried"] = True
            except Exception:  # noqa: BLE001
                pass
            return orig(self)
        start.__trainlab_original__ = orig
        return start
    return [(threading.Thread, "start", {"factory": factory})]


def _asyncio_specs():
    if sys.version_info < (3, 11):
        return []                                  # create_task(context=) arrived in 3.11
    from asyncio import base_events

    def factory(recorder_ref, orig):
        def create_task(self, coro, *args, **kwargs):
            try:
                ctx = origin_context(recorder_ref, kwargs.get("context"))
                if ctx is not None:
                    kwargs["context"] = ctx
            except Exception:  # noqa: BLE001
                pass
            return orig(self, coro, *args, **kwargs)
        create_task.__trainlab_original__ = orig
        return create_task
    return [(base_events.BaseEventLoop, "create_task", {"factory": factory})]


def _http(kind):
    def specs():
        from . import http
        return http.httpx_specs(kind) if kind.startswith("httpx") else getattr(http, f"{kind}_specs")()
    return specs


SDKS = {"openai": _openai_specs, "anthropic": _anthropic_specs, "litellm": _litellm_specs,
        "bedrock": _bedrock_specs, "google-genai": _genai_specs,
        "http-httpx": _http("httpx"), "http-httpx2": _http("httpx2"), "http-requests": _http("requests"),
        "http-aiohttp": _http("aiohttp"),
        "ctx-threadpool": _threadpool_specs, "ctx-thread": _thread_specs, "ctx-asyncio": _asyncio_specs}
SDK_NAMES = ["openai", "anthropic", "litellm", "bedrock", "google-genai"]
_SDK_MODULE = {"openai": "openai", "anthropic": "anthropic", "litellm": "litellm", "bedrock": "botocore.client",
               "google-genai": "google.genai", "http-httpx": "httpx", "http-httpx2": "httpx2",
               "http-requests": "requests", "http-aiohttp": "aiohttp",
               "ctx-threadpool": "concurrent.futures.thread", "ctx-thread": "threading",
               "ctx-asyncio": "asyncio.base_events"}


def _patch_sdk(recorder_ref, name: str) -> int:
    try:
        specs = SDKS[name]()
    except Exception:  # noqa: BLE001
        return 0
    n = 0
    for owner, attr, spec in specs:
        if name in ("openai", "anthropic") and spec.get("is_stream"):
            spec["extract"], spec["is_stream"] = _raw_aware(spec["extract"], spec["is_stream"])
        orig = getattr(owner, attr, None)
        if orig is None or getattr(orig, "__trainlab_original__", None) is not None:
            continue
        if "factory" in spec:
            wrapped = spec["factory"](recorder_ref, orig)
        else:
            wrapped = wrap_call(recorder_ref, orig, spec["describe"], spec["extract"],
                                is_stream=spec.get("is_stream"), wrap_stream=spec.get("wrap_stream"),
                                sdk_name=spec["sdk_name"])
        setattr(owner, attr, wrapped)
        _applied.append((owner, attr, orig))
        n += 1
    if n:
        patched.add(name)
    return n


class _LazyLoader(importlib.abc.Loader):
    def __init__(self, inner, after):
        self.inner, self.after = inner, after

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        try:
            self.after()
        except Exception:  # noqa: BLE001
            pass

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _PostImportHook(importlib.abc.MetaPathFinder):
    """Patch an SDK right after the app imports it, so instrument() never imports SDKs itself
    (LiteLLM alone takes seconds to import)."""

    def __init__(self, recorder_ref, pending: dict[str, str]):
        self.recorder_ref, self.pending = recorder_ref, pending      # module name -> sdk name

    def find_spec(self, fullname, path=None, target=None):
        sdk = self.pending.get(fullname)
        if sdk is None:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:  # noqa: BLE001
                continue
            if spec is not None:
                break
        else:
            return None
        if spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec

        def after():
            if self.pending.pop(fullname, None) is not None:
                _patch_sdk(self.recorder_ref, sdk)
        spec.loader = _LazyLoader(spec.loader, after)
        return spec


_hook: _PostImportHook | None = None


def apply(recorder_ref, only: list[str] | None = None) -> list[str]:
    """Patch SDKs the app has already imported now; the rest when (if) the app imports them.
    Returns the names patched now."""
    global _hook
    done, pending = [], {}
    for name in SDKS:
        family = "http" if name.startswith("http-") else "context" if name.startswith("ctx-") else name
        if only is not None and family not in only:
            continue
        mod = _SDK_MODULE[name]
        if mod in sys.modules:
            if _patch_sdk(recorder_ref, name):
                done.append(name)
            continue
        try:
            if importlib.util.find_spec(mod.split(".")[0]) is None:
                continue
        except (ImportError, ValueError):
            continue
        pending[mod] = name
    if pending:
        if _hook is not None and _hook in sys.meta_path:
            sys.meta_path.remove(_hook)
        _hook = _PostImportHook(recorder_ref, pending)
        sys.meta_path.insert(0, _hook)
    return done


def pending() -> list[str]:
    return sorted(_hook.pending.values()) if _hook is not None and _hook in sys.meta_path else []


def remove() -> None:
    global _hook
    if _hook is not None and _hook in sys.meta_path:
        sys.meta_path.remove(_hook)
    _hook = None
    patched.clear()
    while _applied:
        owner, attr, orig = _applied.pop()
        try:
            setattr(owner, attr, orig)
        except Exception:  # noqa: BLE001
            pass
