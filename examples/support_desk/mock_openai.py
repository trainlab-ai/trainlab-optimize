"""A local stand-in for the OpenAI API, so the support desk runs offline and for free.

It speaks the real wire format (chat completions incl. streaming, JSON mode and tool calls,
embeddings, the Responses API) and answers in a content-aware way: tickets get a plausible
queue, JSON fields, mood and replies. Token counts are about 4 characters per token.

Cheaper models are deliberately a little worse at routing tickets, so `trainlab experiment`
has something to find. Each request's usage is appended to $MOCK_OPENAI_LOG (if set), which
tests use as ground truth.

    python mock_openai.py --port 8787          # then OPENAI_BASE_URL=http://127.0.0.1:8787/v1
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

QUEUE_WORDS = {
    "billing": ["billed", "charge", "invoice", "refund", "vat", "price", "payment", "card", "credit", "seats"],
    "bug": ["stuck", "crash", "closes", "conflict", "search returns", "fails", "error", "re-index", "sync"],
    "account": ["log in", "login", "password", "2fa", "authenticator", "ownership", "merge", "sso", "locked"],
    "shipping": ["order nw-", "delivered", "delivery", "tracking", "customs", "arrived", "desk hub"],
    "feature_request": ["dark mode", "roadmap", "would help", "could you expose", "we'd love", "any plans", "add "],
}
QUEUES = list(QUEUE_WORDS)
# Chance (percent) that a model routes a ticket to the wrong queue.
ROUTING_ERROR = {"gpt-4o": 0, "gpt-4o-mini": 0, "gpt-4.1": 0, "gpt-4.1-mini": 3, "gpt-4.1-nano": 12,
                 "gpt-5-mini": 22, "gpt-5-nano": 35}
# Chance (percent) that a model judges a ticket's urgency differently from gpt-4o: a judgment
# call, so smaller models disagree on it more often, as real ones do.
URGENCY_FLIP = {"gpt-4o-mini": 12, "gpt-4.1-mini": 6, "gpt-4.1-nano": 20, "gpt-5-mini": 8, "gpt-5-nano": 15}
REASONING = re.compile(r"^(gpt-5|o\d)")
_log_lock = threading.Lock()


def tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4)) if text else 0


def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def route_ticket(text: str, model: str) -> str:
    low = text.lower()
    scores = {q: sum(low.count(w) for w in words) for q, words in QUEUE_WORDS.items()}
    if "order nw-" in low or re.search(r"order\s+nw-\d+", low):
        scores["shipping"] += 3
    best = max(QUEUES, key=lambda q: (scores[q], -QUEUES.index(q)))
    if _h(model, text) % 100 < ROUTING_ERROR.get(model, 8):
        best = QUEUES[(QUEUES.index(best) + 1 + _h(text) % 4) % len(QUEUES)]
    return best


def extract(text: str, model: str = "gpt-4o") -> dict:
    low = text.lower()
    order = re.search(r"NW-\d{5}", text)
    product = next((p for p in ["Northwind Sync for Teams", "Northwind Drive", "Northwind Backup",
                                "Northwind Mobile", "Northwind Desk Hub"] if p.lower() in low), None)
    urgent = any(w in low for w in ["locked out", "crash", "declined", "by friday", "hundreds", "two days"])
    subject = re.search(r"Subject: (.+)", text)
    urgency = "high" if urgent else "normal"
    if _h("urgency", model, text) % 100 < URGENCY_FLIP.get(model, 0):
        urgency = "normal" if urgent else "high"
    return {"product": product, "order_id": order.group(0) if order else None,
            "urgency": urgency, "wants_refund": "refund" in low,
            "summary": (subject.group(1) if subject else text[:60]).strip()}


def judge(prompt: str) -> str:
    """Grade two answers: an empty one loses; otherwise the one sharing more words with the request
    wins when the gap is large, and close calls are a tie."""
    def part(tag):
        m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", prompt, re.S)
        return m.group(1) if m else ""
    a, b = (part("field_a"), part("field_b")) if "<field_a>" in prompt else (part("response_a"), part("response_b"))
    if not a.strip() or not b.strip():
        return "B" if not a.strip() else "A"
    req = set(re.findall(r"[a-z]{4,}", part("request").lower()))
    sa = len(req & set(re.findall(r"[a-z]{4,}", a.lower())))
    sb = len(req & set(re.findall(r"[a-z]{4,}", b.lower())))
    return "A" if sa > sb + 3 else "B" if sb > sa + 3 else "TIE"


def mood(text: str) -> str:
    low = text.lower()
    if any(w in low for w in ["locked out", "forever", "hundreds", "keeps failing", "nothing arrived"]):
        return "angry"
    if any(w in low for w in ["still", "twice", "wrong", "doesn't", "didn't", "fails"]):
        return "annoyed"
    return "calm"


def reply_text(user: str) -> str:
    title = re.search(r"## (.+)", user)
    name = re.search(r"Ticket from (.+?):", user)
    return (f"Hi {(name.group(1).split()[0] if name else 'there')},\n\nThanks for getting in touch. "
            f"Our guide '{title.group(1) if title else 'Getting help'}' covers this: follow the steps there and "
            "it should sort things out. If it doesn't, reply to this email and we'll pick it up straight away.\n\n"
            "Northwind Support")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("x-request-id", "req_" + uuid.uuid4().hex[:16])
        self.end_headers()
        self.wfile.write(data)

    def _record(self, endpoint: str, model: str, inp: int, out: int, **extra) -> None:
        path = os.environ.get("MOCK_OPENAI_LOG")
        if not path:
            return
        line = json.dumps({"endpoint": endpoint, "model": model, "in": inp, "out": out, **extra})
        with _log_lock, open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._json(400, {"error": {"message": "invalid JSON", "type": "invalid_request_error"}})
        time.sleep(0.005 + (_h(time.time()) % 20) / 1000)
        path = self.path.split("?")[0]
        if path.endswith("/chat/completions"):
            return self.chat(body)
        if path.endswith("/embeddings"):
            return self.embeddings(body)
        if path.endswith("/responses"):
            return self.responses_api(body)
        return self._json(404, {"error": {"message": f"unknown path {path}", "type": "invalid_request_error"}})

    # ---------------------------------------------------------------- chat completions
    def chat(self, b: dict) -> None:
        model = b.get("model", "")
        if REASONING.match(model) and "max_tokens" in b:
            return self._json(400, {"error": {"message": "Unsupported parameter: 'max_tokens' is not supported with "
                                              "this model. Use 'max_completion_tokens' instead.",
                                              "type": "invalid_request_error", "param": "max_tokens",
                                              "code": "unsupported_parameter"}})
        if REASONING.match(model) and b.get("temperature") not in (None, 1):
            return self._json(400, {"error": {"message": "Unsupported value: 'temperature' does not support "
                                              f"{b['temperature']} with this model. Only the default (1) value is "
                                              "supported.", "type": "invalid_request_error", "param": "temperature",
                                              "code": "unsupported_value"}})
        msgs = b.get("messages", [])
        system = " ".join(_text_of(m.get("content")) for m in msgs if m.get("role") in ("system", "developer"))
        user = " ".join(_text_of(m.get("content")) for m in msgs if m.get("role") == "user")
        prompt_text = json.dumps(msgs) + (json.dumps(b["tools"]) if b.get("tools") else "")
        pin = tokens(prompt_text)
        cap = b.get("max_completion_tokens") or b.get("max_tokens") or 4096
        tool_calls, content = None, None
        if b.get("tools") and not any(m.get("role") == "tool" for m in msgs):
            order = re.search(r"NW-\d{5}", user)
            tool_calls = [{"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                           "function": {"name": b["tools"][0]["function"]["name"],
                                        "arguments": json.dumps({"order_id": order.group(0) if order else ""})}}]
        elif b.get("tools"):
            result = next((json.loads(m["content"]) for m in reversed(msgs) if m.get("role") == "tool"), {})
            content = (f"Your order {result.get('order_id')} is {result.get('status', 'being checked')}"
                       + (f" with {result['carrier']}" if result.get("carrier") else "")
                       + ". We'll email you as soon as it moves.\n\nNorthwind Support")
        elif "<field_a>" in user or "<response_a>" in user:
            content = judge(user)
        elif "Categories:" in system:
            content = route_ticket(user, model)
        elif "return JSON" in system or (b.get("response_format") or {}).get("type") == "json_object":
            content = json.dumps(extract(user, model))
        elif "customer's mood" in system:
            content = mood(user)
        else:
            content = reply_text(user)
        out_text = content if content is not None else json.dumps(tool_calls)
        pout = tokens(out_text)
        finish = "tool_calls" if tool_calls else "stop"
        # Reasoning models think first, inside the same limit (max_completion_tokens), as OpenAI's do:
        # a small limit is used up by reasoning and the visible answer comes back empty.
        reasoning = min((64 + _h(user) % 128) if REASONING.match(model) else 0, cap)
        room = cap - reasoning
        if content is not None and pout > room:
            content, pout, finish = content[: room * 4], room, "length"
        usage = {"prompt_tokens": pin, "completion_tokens": pout + reasoning, "total_tokens": pin + pout + reasoning,
                 "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                 "completion_tokens_details": {"reasoning_tokens": reasoning, "audio_tokens": 0,
                                               "accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0}}
        cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
        self._record("chat", model, pin, pout + reasoning, stream=bool(b.get("stream")),
                     tools=bool(b.get("tools")))
        if b.get("stream"):
            return self._stream_chat(cid, created, model, content, tool_calls, finish, usage,
                                     bool((b.get("stream_options") or {}).get("include_usage")))
        msg = {"role": "assistant", "content": content, "refusal": None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": model,
                         "choices": [{"index": 0, "message": msg, "logprobs": None, "finish_reason": finish}],
                         "usage": usage, "system_fingerprint": "fp_mock"})

    def _stream_chat(self, cid, created, model, content, tool_calls, finish, usage, include_usage):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def send(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                "system_fingerprint": "fp_mock"}
        send({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
        words = re.findall(r"\S+\s*", content or "")
        for i in range(0, len(words), 3):
            send({**base, "choices": [{"index": 0, "delta": {"content": "".join(words[i:i + 3])},
                                       "finish_reason": None}]})
        send({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        if include_usage:
            send({**base, "choices": [], "usage": usage})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    # ---------------------------------------------------------------- embeddings
    def embeddings(self, b: dict) -> None:
        inputs = b.get("input")
        inputs = [inputs] if isinstance(inputs, str) else list(inputs or [])
        data, pin = [], 0
        for i, text in enumerate(inputs):
            pin += tokens(text)
            vec = [0.0] * 64
            for w in re.findall(r"[a-z0-9]+", text.lower()):
                if len(w) > 3:
                    vec[_h(w) % 64] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            data.append({"object": "embedding", "index": i, "embedding": [v / norm for v in vec]})
        self._record("embeddings", b.get("model", ""), pin, 0)
        self._json(200, {"object": "list", "data": data, "model": b.get("model"),
                         "usage": {"prompt_tokens": pin, "total_tokens": pin}})

    # ---------------------------------------------------------------- responses
    def responses_api(self, b: dict) -> None:
        model = b.get("model", "")
        inp = b.get("input")
        text = inp if isinstance(inp, str) else json.dumps(inp)
        pin = tokens((b.get("instructions") or "") + text)
        queues = re.search(r"By queue: (.+)", text)
        out = ("Today's support summary. Volume by queue: " + (queues.group(1) if queues else "steady") + ". "
               "The most common problems were sync issues and billing questions about duplicate charges. "
               "Urgent items are flagged in the CRM. Suggestion: publish a short note on the 8.4.3 update to cut "
               "conflict-copy tickets.")
        cap = b.get("max_output_tokens") or 4096
        pout = min(tokens(out), cap)
        self._record("responses", model, pin, pout)
        rid = "resp_" + uuid.uuid4().hex[:24]
        self._json(200, {
            "id": rid, "object": "response", "created_at": int(time.time()), "status": "completed", "model": model,
            "output": [{"id": "msg_" + uuid.uuid4().hex[:24], "type": "message", "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": out, "annotations": []}]}],
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "error": None,
            "incomplete_details": None, "instructions": b.get("instructions"), "metadata": {},
            "temperature": 1.0, "top_p": 1.0, "max_output_tokens": b.get("max_output_tokens"),
            "usage": {"input_tokens": pin, "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens": pout, "output_tokens_details": {"reasoning_tokens": 0},
                      "total_tokens": pin + pout}})


def start(port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    """Start in a background thread; returns (server, base_url for OPENAI_BASE_URL)."""
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8787)
    a = ap.parse_args()
    srv, url = start(a.port)
    print(f"Mock OpenAI API on {url}  (set OPENAI_BASE_URL={url} OPENAI_API_KEY=mock); Ctrl+C to stop")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.shutdown()
