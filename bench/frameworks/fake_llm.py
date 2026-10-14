"""Fake LLM server speaking the wire formats of many providers, over real HTTP.

Every LLM response is logged to the truth file (tokens it reported, whether usage was sent).
Non-LLM requests (model lists, health checks) are logged with api="other" and not counted.

    python fake_llm.py PORT TRUTH_FILE        # env FAKE_REPLY sets the assistant text

Formats: OpenAI chat/completions/embeddings/responses (and every compatible path: Azure,
Groq, Together, Fireworks, Mistral, Zhipu, DeepSeek, HF, Azure AI Inference, Voyage),
Anthropic messages, Gemini/Vertex generateContent (+stream, embed), Cohere v1/v2 chat and
embed, Ollama chat/generate/embed (NDJSON streams), Bedrock converse/invoke (+ AWS
event-stream binary streams), DashScope native generation/embedding.
"""
import base64
import binascii
import json
import os
import random
import re
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

PORT, TRUTH = int(sys.argv[1]), sys.argv[2]
REPLY = os.environ.get("FAKE_REPLY", "ok")
lock = threading.Lock()
rng = random.Random(int(os.environ.get("FAKE_SEED", "42")))


def log(**kw):
    with lock:
        with open(TRUTH, "a") as fh:
            fh.write(json.dumps(kw) + "\n")


def rand(a, b):
    with lock:
        return rng.randint(a, b)


def _aws_header(name, value):
    n, v = name.encode(), value.encode()
    return struct.pack("B", len(n)) + n + b"\x07" + struct.pack(">H", len(v)) + v


def aws_event(event_type, payload: dict) -> bytes:
    """One vnd.amazon.eventstream message."""
    headers = (_aws_header(":event-type", event_type) + _aws_header(":content-type", "application/json")
               + _aws_header(":message-type", "event"))
    body = json.dumps(payload).encode()
    total = 12 + len(headers) + len(body) + 4
    prelude = struct.pack(">II", total, len(headers))
    prelude += struct.pack(">I", binascii.crc32(prelude) & 0xFFFFFFFF)
    msg = prelude + headers + body
    return msg + struct.pack(">I", binascii.crc32(msg) & 0xFFFFFFFF)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    wbufsize = 1 << 16

    def log_message(self, *a):
        pass

    # -------------------------------------------------------------- transport helpers
    def _send(self, code, obj, headers=None, ctype="application/json"):
        b = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(b)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(b)

    def _sse(self, events, ctype="text/event-stream"):
        self._send(200, "".join(events).encode(), ctype=ctype)

    def do_GET(self):
        p = urlparse(self.path).path
        log(api="other", path=p, method="GET")
        if p.endswith("/models") or p.endswith("/api/tags"):
            return self._send(200, {"object": "list", "data": [{"id": "gpt-4o-mini", "object": "model"}],
                                    "models": [{"name": "llama3.2", "model": "llama3.2"}]})
        return self._send(200, {})

    def do_HEAD(self):
        self._send(200, b"")

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(n)
        if self.headers.get("content-encoding") == "gzip":
            import gzip
            raw = gzip.decompress(raw)
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = {}
        self.body, self.raw = body, raw
        p = urlparse(self.path).path
        ua = (self.headers.get("user-agent") or "").lower()
        pin = len(raw) // 4 + rand(50, 1500)
        pout = rand(10, 400)
        cached = pin // 2 if rand(0, 9) < 3 else 0
        try:
            return self.route(p, ua, pin, pout, cached)
        except BrokenPipeError:
            pass

    # -------------------------------------------------------------- routing
    def route(self, p, ua, pin, pout, cached):
        b = self.body
        model = b.get("model")
        if p.endswith("/chat/completions") or p.endswith("/v1/completions") or p == "/completions":
            return self.openai_chat(model or _path_model(p), pin, pout, cached, ua, legacy=p.endswith("completions")
                                    and not p.endswith("chat/completions"))
        if p.endswith("/embeddings") and "dashscope" not in p:
            inp = b.get("input")
            k = len(inp) if isinstance(inp, list) else 1
            log(api="openai-embed", model=model, inp=pin, out=0, stream=False, usage=True)
            return self._send(200, {"object": "list", "model": model, "usage": {"prompt_tokens": pin, "total_tokens": pin},
                                    "data": [{"object": "embedding", "index": i, "embedding": [0.01] * 8}
                                             for i in range(k)]})
        if p.endswith("/responses"):
            return self.openai_responses(model, pin, pout, cached)
        if p.endswith("/v1/messages") and "/model/" not in p:
            return self.anthropic(model, pin, pout, cached)
        if re.search(r":(generateContent|streamGenerateContent)$", p):
            return self.gemini(_gemini_model(p), pin, pout, cached, stream=p.endswith(":streamGenerateContent"))
        if re.search(r":(embedContent|batchEmbedContents|predict)$", p):
            reqs = b.get("requests") or [b]
            log(api="gemini-embed", model=_gemini_model(p), inp=0, out=0, stream=False, usage=False)
            emb = {"values": [0.01] * 8}
            if p.endswith(":predict"):
                return self._send(200, {"predictions": [{"embeddings": {"values": [0.01] * 8,
                                                         "statistics": {"token_count": 5}}}]})
            return self._send(200, {"embeddings": [emb] * len(reqs)} if p.endswith("batchEmbedContents")
                              else {"embedding": emb})
        if p in ("/v2/chat", "/v1/chat"):
            return self.cohere(model or "command-r", pin, pout, v2=p == "/v2/chat")
        if p in ("/v2/embed", "/v1/embed"):
            log(api="cohere-embed", model=model, inp=pin, out=0, stream=False, usage=True)
            texts = b.get("texts") or b.get("inputs") or [""]
            emb = [[0.01] * 8 for _ in texts]
            return self._send(200, {"id": "e", "embeddings": {"float": emb} if p == "/v2/embed" else emb,
                                    "texts": texts, "meta": {"billed_units": {"input_tokens": pin}},
                                    "response_type": "embeddings_by_type" if p == "/v2/embed" else "embeddings_floats"})
        if p in ("/api/chat", "/api/generate"):
            return self.ollama(model, pin, pout, chat=p == "/api/chat")
        if p in ("/api/embed", "/api/embeddings"):
            log(api="ollama-embed", model=model, inp=pin, out=0, stream=False, usage=True)
            inp = b.get("input") or b.get("prompt") or ""
            k = len(inp) if isinstance(inp, list) else 1
            return self._send(200, {"model": model, "embeddings": [[0.01] * 8] * k, "embedding": [0.01] * 8,
                                    "prompt_eval_count": pin})
        if p in ("/api/show", "/api/pull", "/api/ps"):
            log(api="other", path=p)
            return self._send(200, {"modelfile": "", "parameters": "", "template": "", "details": {},
                                    "model_info": {}, "capabilities": ["completion", "tools"], "status": "success"})
        m = re.match(r"^/model/(.+)/(converse|converse-stream|invoke|invoke-with-response-stream)$", p)
        if m:
            from urllib.parse import unquote
            return self.bedrock(unquote(m.group(1)), m.group(2), pin, pout, cached)
        if p.endswith("/text-generation/generation") or p.endswith("/multimodal-generation/generation"):
            return self.dashscope(model, pin, pout)
        if p.endswith("/text-embedding/text-embedding"):
            log(api="dashscope-embed", model=model, inp=pin, out=0, stream=False, usage=True)
            return self._send(200, {"output": {"embeddings": [{"embedding": [0.01] * 8, "text_index": 0}]},
                                    "usage": {"total_tokens": pin}, "request_id": "r"})
        log(api="other", path=p, method="POST")
        return self._send(404, {"error": {"message": f"fake server: no route {p}"}})

    # -------------------------------------------------------------- OpenAI family
    def openai_chat(self, model, pin, pout, cached, ua, legacy=False):
        b = self.body
        usage = {"prompt_tokens": pin, "completion_tokens": pout, "total_tokens": pin + pout,
                 "prompt_tokens_details": {"cached_tokens": cached}}
        if b.get("stream"):
            # OpenAI only sends usage when asked; Groq/Mistral/Together/Fireworks/... always do.
            ask = bool((b.get("stream_options") or {}).get("include_usage"))
            sends_usage = ask or not ua.startswith("openai/python")
            obj = "text_completion" if legacy else "chat.completion.chunk"
            def ch(delta):
                c = {"index": 0, "finish_reason": None}
                c.update({"text": delta} if legacy else {"delta": {"role": "assistant", "content": delta}})
                return c
            pieces = [REPLY[i:i + 4] for i in range(0, len(REPLY), 4)] or [""]
            ev = [f"data: {json.dumps({'id': 'c', 'object': obj, 'created': 1, 'model': model, 'choices': [ch(t)]})}\n\n"
                  for t in pieces]
            last = {"id": "c", "object": obj, "created": 1, "model": model,
                    "choices": [{"index": 0, "finish_reason": "stop", **({"text": ""} if legacy else {"delta": {}})}]}
            if sends_usage and not ask:
                last["usage"] = usage                       # Mistral/Together style: usage on the final chunk
                last["x_groq"] = {"usage": usage}
            ev.append(f"data: {json.dumps(last)}\n\n")
            if ask:
                ev.append(f"data: {json.dumps({'id': 'c', 'object': obj, 'created': 1, 'model': model, 'choices': [], 'usage': usage})}\n\n")
            ev.append("data: [DONE]\n\n")
            log(api="openai-chat", model=model, inp=pin, out=pout, cached=cached, stream=True, usage=sends_usage)
            return self._sse(ev)
        log(api="openai-chat", model=model, inp=pin, out=pout, cached=cached, stream=False, usage=True)
        if legacy:
            return self._send(200, {"id": "x", "object": "text_completion", "created": 1, "model": model,
                                    "choices": [{"index": 0, "text": REPLY, "finish_reason": "stop", "logprobs": None}],
                                    "usage": usage})
        msg = {"role": "assistant", "content": REPLY}
        if b.get("tools") and os.environ.get("FAKE_TOOL_CALL") and not any(
                m.get("role") == "tool" for m in b.get("messages", []) if isinstance(m, dict)):
            t = b["tools"][0]["function"]
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": t["name"], "arguments": os.environ["FAKE_TOOL_CALL"]}}]}
        return self._send(200, {"id": "x", "object": "chat.completion", "created": 1, "model": model,
                                "choices": [{"index": 0, "message": msg,
                                             "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}],
                                "usage": usage})

    def openai_responses(self, model, pin, pout, cached):
        usage = {"input_tokens": pin, "output_tokens": pout, "total_tokens": pin + pout,
                 "input_tokens_details": {"cached_tokens": cached}, "output_tokens_details": {"reasoning_tokens": 0}}
        resp = {"id": "resp_1", "object": "response", "created_at": 1, "status": "completed", "model": model,
                "output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": REPLY, "annotations": []}]}],
                "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "usage": usage}
        log(api="openai-resp", model=model, inp=pin, out=pout, cached=cached, stream=bool(self.body.get("stream")),
            usage=True)
        if self.body.get("stream"):
            evs = [("response.created", {"type": "response.created", "sequence_number": 0,
                                          "response": {**resp, "status": "in_progress", "output": [], "usage": None}}),
                   ("response.output_text.delta", {"type": "response.output_text.delta", "sequence_number": 1,
                                                   "item_id": "msg_1", "output_index": 0, "content_index": 0,
                                                   "delta": REPLY, "logprobs": []}),
                   ("response.completed", {"type": "response.completed", "sequence_number": 2, "response": resp})]
            return self._sse([f"event: {a}\ndata: {json.dumps(d)}\n\n" for a, d in evs])
        return self._send(200, resp)

    # -------------------------------------------------------------- Anthropic
    def anthropic(self, model, pin, pout, cached, stream=None, raw_events=False):
        cw = pin // 5 if rand(0, 9) < 2 else 0
        usage = {"input_tokens": pin - cached - cw, "output_tokens": pout, "cache_read_input_tokens": cached,
                 "cache_creation_input_tokens": cw}
        stream = self.body.get("stream") if stream is None else stream
        log(api="anthropic", model=model, inp=pin, out=pout, cached=cached, cw=cw, stream=bool(stream), usage=True)
        msg = {"id": "msg_1", "type": "message", "role": "assistant", "model": model,
               "content": [{"type": "text", "text": REPLY}], "stop_reason": "end_turn", "stop_sequence": None,
               "usage": usage}
        evs = [{"type": "message_start", "message": {**msg, "content": [], "stop_reason": None,
                                                      "usage": {**usage, "output_tokens": 1}}},
               {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
               {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": REPLY}},
               {"type": "content_block_stop", "index": 0},
               {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": pout}},
               {"type": "message_stop"}]
        if raw_events:
            return msg, evs
        if stream:
            return self._sse([f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in evs])
        return self._send(200, msg)

    # -------------------------------------------------------------- Gemini / Vertex
    def gemini(self, model, pin, pout, cached, stream):
        log(api="gemini", model=model, inp=pin, out=pout, cached=cached, stream=stream, usage=True)
        resp = {"candidates": [{"content": {"role": "model", "parts": [{"text": REPLY}]}, "finishReason": "STOP",
                                "index": 0}],
                "modelVersion": model,
                "usageMetadata": {"promptTokenCount": pin, "candidatesTokenCount": pout,
                                  "cachedContentTokenCount": cached, "totalTokenCount": pin + pout}}
        if stream:
            if "alt=sse" in self.path:
                return self._sse([f"data: {json.dumps(resp)}\n\n"])
            return self._send(200, [resp])                 # JSON array stream (no alt=sse)
        return self._send(200, resp)

    # -------------------------------------------------------------- Cohere
    def cohere(self, model, pin, pout, v2):
        stream = bool(self.body.get("stream"))
        log(api="cohere", model=model, inp=pin, out=pout, stream=stream, usage=True)
        units = {"input_tokens": pin, "output_tokens": pout}
        if v2:
            usage = {"billed_units": units, "tokens": units}
            if stream:
                evs = [{"type": "message-start", "id": "m", "delta": {"message": {"role": "assistant"}}},
                       {"type": "content-start", "index": 0, "delta": {"message": {"content": {"type": "text", "text": ""}}}},
                       {"type": "content-delta", "index": 0, "delta": {"message": {"content": {"text": REPLY}}}},
                       {"type": "content-end", "index": 0},
                       {"type": "message-end", "delta": {"finish_reason": "COMPLETE", "usage": usage}}]
                return self._sse([f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in evs])
            return self._send(200, {"id": "m", "finish_reason": "COMPLETE",
                                    "message": {"role": "assistant", "content": [{"type": "text", "text": REPLY}]},
                                    "usage": usage})
        meta = {"api_version": {"version": "1"}, "billed_units": units, "tokens": units}
        if stream:
            lines = [{"is_finished": False, "event_type": "stream-start", "generation_id": "g"},
                     {"is_finished": False, "event_type": "text-generation", "text": REPLY},
                     {"is_finished": True, "event_type": "stream-end", "finish_reason": "COMPLETE",
                      "response": {"text": REPLY, "generation_id": "g", "finish_reason": "COMPLETE", "meta": meta,
                                   "chat_history": []}}]
            return self._send(200, ("\n".join(json.dumps(x) for x in lines) + "\n").encode(),
                              ctype="application/stream+json")
        return self._send(200, {"text": REPLY, "generation_id": "g", "response_id": "r", "finish_reason": "COMPLETE",
                                "chat_history": [], "meta": meta})

    # -------------------------------------------------------------- Ollama
    def ollama(self, model, pin, pout, chat):
        stream = self.body.get("stream", True)
        log(api="ollama", model=model, inp=pin, out=pout, stream=bool(stream), usage=True)
        base = {"model": model, "created_at": "2026-10-14T00:00:00Z"}
        stats = {"done": True, "done_reason": "stop", "total_duration": 1000, "load_duration": 1,
                 "prompt_eval_count": pin, "prompt_eval_duration": 1, "eval_count": pout, "eval_duration": 1}
        def piece(t, done=False):
            d = dict(base)
            if chat:
                d["message"] = {"role": "assistant", "content": t}
            else:
                d["response"] = t
            d["done"] = done
            return d
        if stream:
            lines = [piece(REPLY), {**piece(""), **stats}]
            return self._send(200, ("\n".join(json.dumps(x) for x in lines) + "\n").encode(),
                              ctype="application/x-ndjson")
        return self._send(200, {**piece(REPLY), **stats})

    # -------------------------------------------------------------- Bedrock
    def bedrock(self, model, op, pin, pout, cached):
        if op in ("converse", "converse-stream"):
            usage = {"inputTokens": pin - cached, "outputTokens": pout, "totalTokens": pin + pout,
                     "cacheReadInputTokens": cached, "cacheWriteInputTokens": 0}
            log(api="bedrock", model=model, inp=pin, out=pout, cached=cached, stream=op.endswith("stream"), usage=True)
            if op == "converse":
                return self._send(200, {"output": {"message": {"role": "assistant", "content": [{"text": REPLY}]}},
                                        "stopReason": "end_turn", "usage": usage, "metrics": {"latencyMs": 5}})
            evs = [aws_event("messageStart", {"role": "assistant"}),
                   aws_event("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": REPLY}}),
                   aws_event("contentBlockStop", {"contentBlockIndex": 0}),
                   aws_event("messageStop", {"stopReason": "end_turn"}),
                   aws_event("metadata", {"usage": usage, "metrics": {"latencyMs": 5}})]
            return self._send(200, b"".join(evs), ctype="application/vnd.amazon.eventstream")
        # InvokeModel: Anthropic-on-Bedrock body
        if op == "invoke":
            msg, _ = self.anthropic(model, pin, pout, 0, stream=False, raw_events=True)
            return self._send(200, msg, headers={"x-amzn-bedrock-input-token-count": str(pin),
                                                 "x-amzn-bedrock-output-token-count": str(pout)})
        msg, evs = self.anthropic(model, pin, pout, 0, stream=True, raw_events=True)
        evs[-1]["amazon-bedrock-invocationMetrics"] = {"inputTokenCount": pin, "outputTokenCount": pout,
                                                       "invocationLatency": 5, "firstByteLatency": 1}
        chunks = [aws_event("chunk", {"bytes": base64.b64encode(json.dumps(e).encode()).decode()}) for e in evs]
        return self._send(200, b"".join(chunks), ctype="application/vnd.amazon.eventstream")

    # -------------------------------------------------------------- DashScope native
    def dashscope(self, model, pin, pout):
        params = self.body.get("parameters") or {}
        sse = "text/event-stream" in (self.headers.get("accept") or "") or self.headers.get("x-dashscope-sse") == "enable"
        log(api="dashscope", model=model, inp=pin, out=pout, stream=sse, usage=True)
        out = ({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": REPLY}}]}
               if params.get("result_format") == "message" else {"text": REPLY, "finish_reason": "stop"})
        resp = {"output": out, "usage": {"input_tokens": pin, "output_tokens": pout, "total_tokens": pin + pout},
                "request_id": "req-1"}
        if sse:
            return self._sse([f"id:1\nevent:result\n:HTTP_STATUS/200\ndata:{json.dumps(resp)}\n\n"])
        return self._send(200, resp)


def _path_model(p):
    m = re.search(r"/deployments/([^/]+)/", p) or re.search(r"/models/([^/]+)/v1/", p)
    return m.group(1) if m else None


def _gemini_model(p):
    m = re.search(r"models/([^/:]+):", p)
    return m.group(1) if m else None


ThreadingHTTPServer.daemon_threads = True
ThreadingHTTPServer.request_queue_size = 2048
srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
print(f"fake LLM on {PORT}", flush=True)
srv.serve_forever()
