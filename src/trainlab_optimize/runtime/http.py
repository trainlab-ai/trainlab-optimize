"""HTTP-level layer: LLM calls made through httpx, httpx2, requests or aiohttp.

Covers SDKs without a dedicated hook (Mistral, Cohere, Groq, Together, Ollama, Hugging Face,
DashScope, Zhipu, Azure AI Inference, Voyage, Fireworks, ...), framework integrations that
call APIs directly, and apps posting to LLM APIs by hand.

- A request is a candidate only if it is a POST to a known LLM API path; a response is
  recorded only if it is LLM-shaped (usage fields or LLM stream chunks), so an app's own
  `/api/chat` route is not mistaken for an LLM call.
- When an SDK hook is already recording the call, this layer creates nothing: it only fills
  in token counts that hook could not read (e.g. from raw-response wrappers).
- Streams are read alongside the app (the bytes it consumes are parsed line by line, nothing
  is kept beyond the current line) and recorded when the app finishes or closes them.
"""
from __future__ import annotations

import json
import math
import re
import time

from ..scanner.providers import provider_from_url
from .core import SKIP, CallEvent, _active, _begin, num

# path -> (provider family, op)
_ROUTES = [
    (re.compile(r"/chat/completions$"), "openai", "chat"),
    (re.compile(r"/v\d+/completions$|^/completions$|/inference/v1/completions$"), "openai", "chat"),
    (re.compile(r"/embeddings$"), "openai", "embedding"),
    (re.compile(r"/v1/responses$|/openai/responses$"), "openai", "chat"),
    (re.compile(r"/v1/messages$"), "anthropic", "chat"),
    (re.compile(r":(generateContent|streamGenerateContent)$"), "gemini", "chat"),
    (re.compile(r":(embedContent|batchEmbedContents)$"), "gemini", "embedding"),
    (re.compile(r"^/v[12]/chat$"), "cohere", "chat"),
    (re.compile(r"^/v[12]/embed$"), "cohere", "embedding"),
    (re.compile(r"/api/(chat|generate)$"), "ollama", "chat"),
    (re.compile(r"/api/embed(dings)?$"), "ollama", "embedding"),
    (re.compile(r"/services/aigc/(text|multimodal)-generation/generation$"), "dashscope", "chat"),
    (re.compile(r"/services/embeddings/text-embedding/text-embedding$"), "dashscope", "embedding"),
]
_USAGE_MARKS = (b'"usage', b'"usageMetadata"', b'"prompt_eval_count"', b'"billed_units"', b'"meta"')
_MAX_LINE = 4 << 20
_MAX_BODY = 8 << 20


_TAILS = ("completions", "embeddings", "responses", "messages", "Content", "Contents", "/chat", "/embed",
          "/generate", "generation", "text-embedding")


def match(method: str, url) -> tuple[str, str] | None:
    if method != "POST" and str(method).upper() != "POST":
        return None
    path = getattr(url, "path", None)
    if path is None:
        from urllib.parse import urlparse
        path = urlparse(str(url)).path
    path = (path or "").rstrip("/")
    if not path.endswith(_TAILS):                   # fast exit for every non-LLM request
        return None
    for rx, fam, op in _ROUTES:
        if rx.search(path):
            return fam, op
    return None


def _provider(url, family: str) -> str:
    u = str(url)
    p = provider_from_url(u)
    if p and p != "openai_compatible":
        return p
    if family == "openai":
        return "azure" if "/openai/deployments/" in u else ("openai" if "api.openai.com" in u else
                                                            p or "openai_compatible")
    return family


def _model_from(url, body: dict | None) -> str | None:
    if isinstance(body, dict) and isinstance(body.get("model"), str):
        return body["model"]
    u = str(url)
    m = re.search(r"/models/([^/:?]+):", u) or re.search(r"/deployments/([^/?]+)/", u)
    return m.group(1) if m else None


def _json_body(content) -> dict | None:
    if not content:
        return None
    try:
        if isinstance(content, str):
            content = content.encode()
        if len(content) > _MAX_BODY or content[:1] not in (b"{", b"["):
            return None
        d = json.loads(content)
        return d if isinstance(d, dict) else None
    except (ValueError, TypeError):
        return None


# ------------------------------------------------------------------ usage from any provider's JSON
def _texts(d: dict) -> list[str]:
    """Answer text pieces in one response or stream chunk, any supported provider."""
    out: list[str] = []
    for ch in d.get("choices") or []:
        if isinstance(ch, dict):
            delta = ch.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                out.append(delta["content"])
            msg = ch.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                out.append(msg["content"])
            if isinstance(ch.get("text"), str):
                out.append(ch["text"])
    delta = d.get("delta")
    if isinstance(delta, dict):
        if isinstance(delta.get("text"), str):
            out.append(delta["text"])
        msg = delta.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), dict):
            out.append(msg["content"].get("text") or "")
    if d.get("type") == "response.output_text.delta" and isinstance(d.get("delta"), str):
        out.append(d["delta"])
    msg = d.get("message")
    if isinstance(msg, dict):
        if isinstance(msg.get("content"), str):
            out.append(msg["content"])                     # Ollama chat
        elif isinstance(msg.get("content"), list) and "role" in msg and d.get("type") != "message_start":
            out += [c.get("text") or "" for c in msg["content"] if isinstance(c, dict)]   # Cohere v2
    if isinstance(d.get("content"), list) and d.get("type") == "message":
        out += [c.get("text") or "" for c in d["content"] if isinstance(c, dict) and c.get("type") == "text"]
    if isinstance(d.get("response"), str):
        out.append(d["response"])                          # Ollama generate
    for c in d.get("candidates") or []:
        for part in ((c or {}).get("content") or {}).get("parts") or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                out.append(part["text"])
    if isinstance(d.get("text"), str) and d.get("event_type") in (None, "text-generation") and "meta" in d \
            or d.get("event_type") == "text-generation":
        out.append(d.get("text") or "")                    # Cohere v1
    out_obj = d.get("output")
    if isinstance(out_obj, dict):                          # DashScope
        if isinstance(out_obj.get("text"), str):
            out.append(out_obj["text"])
        for ch in out_obj.get("choices") or []:
            m = (ch or {}).get("message") or {}
            if isinstance(m.get("content"), str):
                out.append(m["content"])
    return out


def apply_json(d, ev: CallEvent, overwrite: bool) -> bool:
    """Read usage (and model) from one response or stream chunk of any supported provider.
    Returns True if the object looked like an LLM response."""
    if isinstance(d, list):
        return any([apply_json(x, ev, overwrite) for x in d])
    if not isinstance(d, dict):
        return False
    vals: dict = {}
    llm = False
    model = d.get("model") or d.get("modelVersion")
    if not model and isinstance(d.get("message"), dict):
        model = d["message"].get("model")
    if not model and isinstance(d.get("response"), dict):
        model = d["response"].get("model")
    u = d.get("usage")
    if not isinstance(u, dict):
        u = None
    for alt in (("x_groq", "usage"), ("response", "usage"), ("message", "usage"), ("delta", "usage")):
        if u is None and isinstance(d.get(alt[0]), dict) and isinstance(d[alt[0]].get(alt[1]), dict):
            u = d[alt[0]][alt[1]]
    if isinstance(u, dict):
        llm = True
        if "prompt_tokens" in u or "completion_tokens" in u:
            vals["input_tokens"] = num(u.get("prompt_tokens"))
            vals["output_tokens"] = num(u.get("completion_tokens"))
            vals["cached_input_tokens"] = num((u.get("prompt_tokens_details") or {}).get("cached_tokens")) \
                or num(u.get("prompt_cache_hit_tokens"))
            vals["reasoning_tokens"] = num((u.get("completion_tokens_details") or {}).get("reasoning_tokens"))
            if u.get("cache_creation_input_tokens"):
                vals["cache_write_tokens"] = num(u.get("cache_creation_input_tokens"))
        elif "billed_units" in u or "tokens" in u:                       # Cohere v2
            t = u.get("billed_units") or u.get("tokens") or {}
            vals["input_tokens"], vals["output_tokens"] = num(t.get("input_tokens")), num(t.get("output_tokens"))
        elif "input_tokens" in u or "output_tokens" in u or "inputTokens" in u:
            cr = num(u.get("cache_read_input_tokens")) or num(u.get("cacheReadInputTokens"))
            cw = num(u.get("cache_creation_input_tokens")) or num(u.get("cacheWriteInputTokens"))
            inp = num(u.get("input_tokens")) if "input_tokens" in u else num(u.get("inputTokens"))
            if inp is not None:
                vals["input_tokens"] = inp + (cr or 0) + (cw or 0)
            out = num(u.get("output_tokens")) if "output_tokens" in u else num(u.get("outputTokens"))
            if out is not None:
                vals["output_tokens"] = out
            vals["cached_input_tokens"] = cr or num((u.get("input_tokens_details") or {}).get("cached_tokens"))
            vals["cache_write_tokens"] = cw
            vals["reasoning_tokens"] = num((u.get("output_tokens_details") or {}).get("reasoning_tokens"))
        elif "total_tokens" in u:                                         # embeddings (Voyage, DashScope)
            vals["input_tokens"] = num(u.get("total_tokens"))
    um = d.get("usageMetadata")
    if isinstance(um, dict):                                              # Gemini / Vertex
        llm = True
        vals["input_tokens"] = num(um.get("promptTokenCount"))
        vals["output_tokens"] = num(um.get("candidatesTokenCount"))
        vals["cached_input_tokens"] = num(um.get("cachedContentTokenCount"))
        vals["reasoning_tokens"] = num(um.get("thoughtsTokenCount"))
    meta = d.get("meta") or (d.get("response") or {}).get("meta") if isinstance(d.get("response"), dict) \
        else d.get("meta")
    if isinstance(meta, dict) and isinstance(meta.get("billed_units"), dict):   # Cohere v1
        llm = True
        bu = meta["billed_units"]
        vals["input_tokens"], vals["output_tokens"] = num(bu.get("input_tokens")), num(bu.get("output_tokens"))
    if "prompt_eval_count" in d or "eval_count" in d:                    # Ollama
        llm = True
        vals["input_tokens"], vals["output_tokens"] = num(d.get("prompt_eval_count")), num(d.get("eval_count"))
    if not llm and (d.get("choices") is not None or d.get("candidates") is not None
                    or d.get("type", "").startswith(("message_", "content_block", "response.", "message-", "content-"))
                    or ("done" in d and ("message" in d or "response" in d)) or "embeddings" in d
                    or d.get("object") in ("list", "chat.completion.chunk", "text_completion")):
        llm = True
    if model and isinstance(model, str) and (overwrite or ev.model is None):
        ev.model = model
    vals = {k: v for k, v in vals.items() if v is not None}
    if overwrite:
        for k, v in vals.items():
            setattr(ev, k, v)
    else:
        # Fill what the SDK hook left empty; values filled from an earlier chunk of this same
        # stream may be updated by later chunks (Anthropic: output_tokens 1, then the real count).
        mine = ev.__dict__.setdefault("_http_filled", set())
        for k, v in vals.items():
            if getattr(ev, k, None) is None or k in mine:
                setattr(ev, k, v)
                mine.add(k)
    if overwrite:
        parts = _texts(d)
        ev._chars = getattr(ev, "_chars", 0) + sum(len(t) for t in parts)
        cap = ev.__dict__.get("_cap_text")
        if cap is not None:
            cap.extend(parts)
    return llm


def fill_from_body(ev: CallEvent, body: bytes) -> None:
    try:
        apply_json(json.loads(body), ev, overwrite=False)
    except ValueError:
        pass


class _Tap:
    """Parses a response body as it streams by: SSE (`data: {...}`), NDJSON, or one JSON
    document. Holds at most one line (or, for a plain JSON body, the body up to 8 MB)."""

    def __init__(self, ev: CallEvent, content_type: str, own: bool, on_done):
        self.ev, self.own, self.on_done = ev, own, on_done
        ct = (content_type or "").lower()
        self.whole = "json" in ct and "ndjson" not in ct and "stream" not in ct
        self.buf = b""
        self.llm = False
        self.done = False

    def feed(self, chunk) -> None:
        if self.done or not chunk:
            return
        try:
            if isinstance(chunk, str):
                chunk = chunk.encode()
            self.buf += chunk
            if self.whole:
                if len(self.buf) > _MAX_BODY:
                    self.buf, self.whole = b"", False
                return
            while b"\n" in self.buf:
                line, self.buf = self.buf.split(b"\n", 1)
                self._line(line)
            if len(self.buf) > _MAX_LINE:
                self.buf = b""
        except Exception:  # noqa: BLE001
            self.buf = b""

    def _line(self, line: bytes) -> None:
        line = line.strip()
        if line.startswith(b"data:"):
            line = line[5:].strip()
        if not line or line[:1] not in (b"{", b"["):
            return
        if not self.own and not any(m in line for m in _USAGE_MARKS):
            return                       # filling an SDK's event: only usage matters
        try:
            d = json.loads(line)
        except ValueError:
            return
        self.llm |= apply_json(d, self.ev, overwrite=self.own)

    def close(self, error=None, complete=True) -> None:
        if self.done:
            return
        self.done = True
        try:
            if self.whole and self.buf and not self.own and not self.ev.__dict__.get("_defer"):
                # Filling an SDK's event from a complete body: the SDK usually reads usage itself,
                # so keep a reference and parse only if it didn't (see core._finish_or_stream).
                self.ev._http_body = self.buf
                self.buf = b""
            elif self.whole and self.buf:
                try:
                    self.llm |= apply_json(json.loads(self.buf), self.ev, overwrite=self.own)
                except ValueError:
                    pass
            elif self.buf:
                self._line(self.buf)
            self.buf = b""
            if self.own:
                if not complete and error is None:
                    self.ev.stream_incomplete = True
                chars = getattr(self.ev, "_chars", 0)
                if self.ev.output_tokens is None and chars and self.ev.op == "chat":
                    self.ev.output_tokens = math.ceil(chars / 4)
                    self.ev.output_tokens_estimated = True
        except Exception:  # noqa: BLE001
            pass
        self.on_done(self.llm, error)


# ------------------------------------------------------------------ shared request bookkeeping
class _Call:
    """One HTTP request to an LLM API: either recorded here (own) or filling an SDK's event."""

    def __init__(self, recorder_ref, family_op, url, body, client_name):
        self.rec = recorder_ref()
        outer = _active.get()
        self.own = outer is None
        self.t0 = time.perf_counter()
        family, op = family_op
        if self.own:
            ev, self.t0, self.token = _begin(self.rec, (_provider(url, family), _model_from(url, body), op),
                                             f"http:{client_name}", depth=4)
            self.ev = ev
            if isinstance(body, dict):
                self.ev._req = body
                cap = self.rec.capture
                if cap is not None and cap.wants(ev):
                    ev._cap_text, ev._cap_req = [], body
        else:
            self.ev, self.token = outer, None

    def release(self):
        if self.token is not None:
            _active.reset(self.token)
            self.token = None

    def failed(self, exc):
        self.release()
        if self.own:
            self.ev.ok, self.ev.error = False, type(exc).__name__
            self.ev.ms = round((time.perf_counter() - self.t0) * 1000, 1)
            self.rec.emit(self.ev)

    def done(self, llm: bool, error=None, status: int | None = None):
        if not self.own:
            ev = self.ev
            if ev.__dict__.pop("_defer", False):  # an SDK raw/streaming response, read only now
                ev.ms = round((time.perf_counter() - ev.__dict__.pop("_t0", self.t0)) * 1000, 1)
                if error is not None:
                    ev.ok, ev.error = False, type(error).__name__
                self.rec.emit(ev)
            return
        ev = self.ev
        ev.ms = round((time.perf_counter() - self.t0) * 1000, 1)
        if status is not None and status >= 400:
            ev.ok, ev.error = False, f"HTTP {status}"
            self.rec.emit(ev)
            return
        if error is not None:
            ev.ok, ev.error = False, type(error).__name__
        if not llm and error is None:
            return                         # not an LLM response after all: record nothing
        req = ev.__dict__.pop("_req", None)
        from .core import _capture_stream
        _capture_stream(self.rec, ev)
        if ev.input_tokens is None and isinstance(req, dict) and ev.op == "chat":
            from .patches import _text_chars
            n = sum(_text_chars(req.get(k)) for k in ("messages", "input", "system", "contents", "prompt", "message"))
            if n:
                ev.input_tokens = math.ceil(n / 4) + 4 * len(req.get("messages") or [])
                ev.input_tokens_estimated = True
        self.rec.emit(ev)

    def tap(self, content_type, stream: bool, status: int):
        self.ev.stream = self.ev.stream or (stream and self.own)
        return _Tap(self.ev, content_type, self.own, lambda llm, err: self.done(llm, err, status))


def _skip(recorder_ref) -> bool:
    """Pass through: sampled out here, or inside a call an outer hook sampled out."""
    outer = _active.get()
    if outer is SKIP:
        return True
    return outer is None and not recorder_ref().sampled()


def _usable(recorder_ref):
    rec = recorder_ref()
    return rec is not None and rec.enabled


# ------------------------------------------------------------------ httpx / httpx2
def httpx_specs(module_name: str):
    try:
        import importlib
        hx = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001
        return []

    def body_of(request):
        try:
            return _json_body(request.content)
        except Exception:  # noqa: BLE001 - streaming upload body
            return None

    def attach(call, response, stream):
        status = response.status_code
        if not stream:
            try:
                tap = call.tap(response.headers.get("content-type", ""), False, status)
                tap.feed(response.content)
                tap.close()
            except Exception:  # noqa: BLE001
                pass
            return response
        tap = call.tap(response.headers.get("content-type", ""), True, status)
        if status >= 400:
            tap.close()
            return response
        _wrap_httpx_response(response, tap)
        return response

    def sync_factory(recorder_ref, orig):
        def send(self, request, *args, **kwargs):
            fo = match(request.method, request.url) if _usable(recorder_ref) else None
            if fo is None or _skip(recorder_ref):
                return orig(self, request, *args, **kwargs)
            call = _Call(recorder_ref, fo, request.url,
                         body_of(request) if _active.get() is None else None, module_name)
            try:
                response = orig(self, request, *args, **kwargs)
            except BaseException as exc:
                call.failed(exc)
                raise
            call.release()
            try:
                return attach(call, response, bool(kwargs.get("stream")))
            except Exception:  # noqa: BLE001
                return response
        send.__trainlab_original__ = orig
        return send

    def async_factory(recorder_ref, orig):
        async def send(self, request, *args, **kwargs):
            fo = match(request.method, request.url) if _usable(recorder_ref) else None
            if fo is None or _skip(recorder_ref):
                return await orig(self, request, *args, **kwargs)
            call = _Call(recorder_ref, fo, request.url,
                         body_of(request) if _active.get() is None else None, module_name)
            try:
                response = await orig(self, request, *args, **kwargs)
            except BaseException as exc:
                call.failed(exc)
                raise
            call.release()
            try:
                return attach(call, response, bool(kwargs.get("stream")))
            except Exception:  # noqa: BLE001
                return response
        send.__trainlab_original__ = orig
        return send

    return [(hx.Client, "send", {"factory": sync_factory}), (hx.AsyncClient, "send", {"factory": async_factory})]


def _wrap_httpx_response(response, tap: _Tap) -> None:
    """Tap the decoded bytes of a streamed response (iter_bytes feeds iter_text/iter_lines/read)."""
    orig_iter, orig_aiter = response.iter_bytes, response.aiter_bytes
    orig_close, orig_aclose = response.close, response.aclose

    def iter_bytes(*a, **k):
        ok = False
        try:
            for chunk in orig_iter(*a, **k):
                tap.feed(chunk)
                yield chunk
            ok = True
        finally:
            tap.close(complete=ok)

    async def aiter_bytes(*a, **k):
        ok = False
        try:
            async for chunk in orig_aiter(*a, **k):
                tap.feed(chunk)
                yield chunk
            ok = True
        finally:
            tap.close(complete=ok)

    def close():
        try:
            return orig_close()
        finally:
            tap.close(complete=False)

    async def aclose():
        try:
            return await orig_aclose()
        finally:
            tap.close(complete=False)

    response.iter_bytes, response.aiter_bytes = iter_bytes, aiter_bytes
    response.close, response.aclose = close, aclose


# ------------------------------------------------------------------ requests
def requests_specs():
    try:
        import requests
    except Exception:  # noqa: BLE001
        return []

    def factory(recorder_ref, orig):
        def send(self, request, **kwargs):
            fo = match(request.method, request.url) if _usable(recorder_ref) else None
            if fo is None or _skip(recorder_ref):
                return orig(self, request, **kwargs)
            call = _Call(recorder_ref, fo, request.url,
                         _json_body(request.body) if _active.get() is None else None, "requests")
            try:
                response = orig(self, request, **kwargs)
            except BaseException as exc:
                call.failed(exc)
                raise
            call.release()
            try:
                stream = bool(kwargs.get("stream"))
                tap = call.tap(response.headers.get("content-type", ""), stream, response.status_code)
                if not stream or response._content_consumed:
                    tap.feed(response.content)
                    tap.close()
                else:
                    orig_iter, orig_close = response.iter_content, response.close

                    def iter_content(*a, **k):
                        ok = False
                        try:
                            for chunk in orig_iter(*a, **k):
                                tap.feed(chunk)
                                yield chunk
                            ok = True
                        finally:
                            tap.close(complete=ok)

                    def close():
                        try:
                            return orig_close()
                        finally:
                            tap.close(complete=False)
                    response.iter_content, response.close = iter_content, close
            except Exception:  # noqa: BLE001
                pass
            return response
        send.__trainlab_original__ = orig
        return send

    return [(requests.Session, "send", {"factory": factory})]


# ------------------------------------------------------------------ aiohttp
class _AioContent:
    """Stand-in for aiohttp's StreamReader that taps what the app reads."""

    def __init__(self, inner, tap: _Tap):
        self._inner, self._tap = inner, tap

    def _seen(self, data):
        if data:
            self._tap.feed(data)
        if not data or self._inner.at_eof():
            self._tap.close()
        return data

    async def read(self, n=-1):
        return self._seen(await self._inner.read(n))

    async def readany(self):
        return self._seen(await self._inner.readany())

    async def readline(self):
        return self._seen(await self._inner.readline())

    async def readexactly(self, n):
        return self._seen(await self._inner.readexactly(n))

    async def readchunk(self):
        data, end = await self._inner.readchunk()
        self._seen(data)
        return data, end

    def __aiter__(self):
        return self._lines()

    async def _lines(self):
        while True:
            line = await self.readline()
            if not line:
                return
            yield line

    async def _chunks(self, gen):
        async for x in gen:
            self._seen(x[0] if isinstance(x, tuple) else x)
            yield x
        self._tap.close()

    def iter_chunked(self, n):
        return self._chunks(self._inner.iter_chunked(n))

    def iter_any(self):
        return self._chunks(self._inner.iter_any())

    def iter_chunks(self):
        return self._chunks(self._inner.iter_chunks())

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __del__(self):
        try:
            self._tap.close(complete=False)
        except Exception:  # noqa: BLE001
            pass


def aiohttp_specs():
    try:
        import aiohttp
    except Exception:  # noqa: BLE001
        return []

    def factory(recorder_ref, orig):
        async def _request(self, method, str_or_url, *args, **kwargs):
            fo = match(method, str_or_url) if _usable(recorder_ref) else None
            if fo is None or _skip(recorder_ref):
                return await orig(self, method, str_or_url, *args, **kwargs)
            body = kwargs.get("json") if isinstance(kwargs.get("json"), dict) else _json_body(kwargs.get("data"))
            call = _Call(recorder_ref, fo, str_or_url, body, "aiohttp")
            try:
                response = await orig(self, method, str_or_url, *args, **kwargs)
            except BaseException as exc:
                call.failed(exc)
                raise
            call.release()
            try:
                ct = response.headers.get("content-type", "")
                tap = call.tap(ct, "json" not in ct.lower() or "stream" in ct.lower(), response.status)
                response.content = _AioContent(response.content, tap)
            except Exception:  # noqa: BLE001
                pass
            return response
        _request.__trainlab_original__ = orig
        return _request

    return [(aiohttp.ClientSession, "_request", {"factory": factory})]
