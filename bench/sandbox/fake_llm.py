"""Fake LLM server over real HTTP: OpenAI, Anthropic and Gemini wire formats, streaming and errors.

Every response it serves is appended to truth.jsonl (tokens it reported, the caller's
x-truth-feature header, status). That file is the ground truth trainlab is checked against.

    python fake_llm.py PORT TRUTH_FILE
"""
import json
import random
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, TRUTH = int(sys.argv[1]), sys.argv[2]
lock = threading.Lock()
rng = random.Random(42)


def log(**kw):
    with lock:
        with open(TRUTH, "a") as fh:
            fh.write(json.dumps(kw) + "\n")


def rand(a, b):
    with lock:
        return rng.randint(a, b)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    wbufsize = 1 << 16          # headers + body in one TCP write (avoids 40 ms delayed-ACK stalls)

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _sse(self, events):
        body = "".join(events).encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(n)
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = {}
        feat = self.headers.get("x-truth-feature")
        fail = self.headers.get("x-fail")
        path = self.path
        model = body.get("model") or (re.search(r"models/([^:]+):", path) or [None, None])[1]
        api = ("openai-embed" if path.endswith("/embeddings") else "openai-resp" if path.endswith("/responses")
               else "openai-chat" if path.endswith("/chat/completions") else "anthropic" if path.endswith("/messages")
               else "gemini" if ":generateContent" in path or ":streamGenerateContent" in path else "other")
        if fail:
            code = int(fail)
            log(api=api, model=model, status=code, feature=feat)
            if api == "anthropic":
                return self._send(code, {"type": "error", "error": {"type": "api_error", "message": "boom"}})
            return self._send(code, {"error": {"message": "boom", "type": "server_error", "code": code}})
        if model == "slow":
            time.sleep(0.3)
        pin = len(raw) // 4 + rand(50, 1500)
        pout = rand(10, 400)
        cached = pin // 2 if rand(0, 9) < 3 else 0

        if api == "openai-embed":
            inp = body.get("input")
            k = len(inp) if isinstance(inp, list) else 1
            log(api=api, model=model, status=200, inp=pin, out=0, feature=feat, stream=False, usage=True)
            return self._send(200, {"object": "list", "model": model, "usage": {"prompt_tokens": pin, "total_tokens": pin},
                                    "data": [{"object": "embedding", "index": i, "embedding": [0.0] * 4} for i in range(k)]})
        if api == "openai-resp":
            log(api=api, model=model, status=200, inp=pin, out=pout, cached=cached, feature=feat, stream=False, usage=True)
            return self._send(200, {
                "id": "r", "object": "response", "created_at": 1, "status": "completed", "model": model,
                "output": [{"type": "message", "id": "m", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": "ok", "annotations": []}]}],
                "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
                "usage": {"input_tokens": pin, "output_tokens": pout, "total_tokens": pin + pout,
                          "input_tokens_details": {"cached_tokens": cached}, "output_tokens_details": {"reasoning_tokens": 0}}})
        if api == "openai-chat":
            usage = {"prompt_tokens": pin, "completion_tokens": pout, "total_tokens": pin + pout,
                     "prompt_tokens_details": {"cached_tokens": cached}}
            if body.get("stream"):
                inc = bool((body.get("stream_options") or {}).get("include_usage"))
                words = ["tok "] * 6
                ev = [f"data: {json.dumps({'id': 'c', 'object': 'chat.completion.chunk', 'created': 1, 'model': model, 'choices': [{'index': 0, 'delta': {'content': w}, 'finish_reason': None}]})}\n\n" for w in words]
                if inc:
                    ev.append(f"data: {json.dumps({'id': 'c', 'object': 'chat.completion.chunk', 'created': 1, 'model': model, 'choices': [], 'usage': usage})}\n\n")
                ev.append("data: [DONE]\n\n")
                log(api=api, model=model, status=200, inp=pin, out=pout, cached=cached, feature=feat, stream=True, usage=inc)
                return self._sse(ev)
            log(api=api, model=model, status=200, inp=pin, out=pout, cached=cached, feature=feat, stream=False, usage=True)
            return self._send(200, {"id": "x", "object": "chat.completion", "created": 1, "model": model,
                                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                                 "finish_reason": "stop"}], "usage": usage})
        if api == "anthropic":
            cw = pin // 5 if rand(0, 9) < 2 else 0
            base_in = pin - cached - cw
            usage = {"input_tokens": base_in, "output_tokens": pout, "cache_read_input_tokens": cached,
                     "cache_creation_input_tokens": cw}
            log(api=api, model=model, status=200, inp=pin, out=pout, cached=cached, cw=cw, feature=feat,
                stream=bool(body.get("stream")), usage=True)
            if body.get("stream"):
                evs = [("message_start", {"type": "message_start", "message": {
                            "id": "m", "type": "message", "role": "assistant", "model": model, "content": [],
                            "stop_reason": None, "stop_sequence": None, "usage": {**usage, "output_tokens": 1}}}),
                       ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
                       ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}}),
                       ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                       ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                          "usage": {"output_tokens": pout}}),
                       ("message_stop", {"type": "message_stop"})]
                return self._sse([f"event: {a}\ndata: {json.dumps(b)}\n\n" for a, b in evs])
            return self._send(200, {"id": "m", "type": "message", "role": "assistant", "model": model,
                                    "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                                    "stop_sequence": None, "usage": usage})
        if api == "gemini":
            stream = ":streamGenerateContent" in path
            log(api=api, model=model, status=200, inp=pin, out=pout, cached=cached, feature=feat, stream=stream, usage=True)
            resp = {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                    "modelVersion": model,
                    "usageMetadata": {"promptTokenCount": pin, "candidatesTokenCount": pout,
                                      "cachedContentTokenCount": cached, "totalTokenCount": pin + pout}}
            if stream:
                return self._sse([f"data: {json.dumps(resp)}\n\n"])
            return self._send(200, resp)
        self._send(404, {"error": {"message": f"no route {path}"}})


ThreadingHTTPServer.daemon_threads = True
ThreadingHTTPServer.request_queue_size = 2048
srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
print(f"fake LLM on {PORT}", flush=True)
srv.serve_forever()
