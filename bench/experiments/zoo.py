"""A simulated model zoo for testing `trainlab experiment` without real API keys.

Speaks the OpenAI (and every OpenAI-compatible path), Azure OpenAI, Anthropic, Gemini and
Bedrock Converse wire formats. Each request's last user message identifies a registered
sample; the model's designed accuracy for that sample's dataset decides, deterministically,
whether it answers correctly. So the harness knows exactly what every model will answer and
can check the experiment engine's verdicts, scores and cost math against the truth.

Also: injects 429 (with Retry-After) and 500 errors on some first attempts, reports token
usage computed from the request and answer text, judges A/B comparisons like a decent judge,
and logs every request it receives.

    python zoo.py PORT LOGFILE
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

LOCK = threading.Lock()
REGISTRY: dict[str, dict] = {}
ATTEMPTS: dict[tuple, int] = {}
JUDGE_MARK = "You are grading two responses"


def h(*parts) -> float:
    """Deterministic uniform [0,1) from the parts."""
    d = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(d[:12], 16) / float(16 ** 12)


def norm(s: str) -> str:
    return " ".join((s or "").split())


def key_of(user_text: str) -> str:
    return hashlib.sha256(norm(user_text).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ answers (also imported by the harness)
def correct(entry: dict, model: str) -> bool:
    acc = entry["profile"].get(model, entry["profile"].get("*", 0.5))
    return h("ok", model, entry["key"]) < acc


def answer(entry: dict, model: str) -> str:
    task, gold = entry["task"], entry["gold"]
    good = correct(entry, model)
    v = h("variant", model, entry["key"])
    if task == "classification":
        labels = entry["labels"]
        label = gold if good else [x for x in labels if x != gold][int(h("wrong", model, entry["key"]) * (len(labels) - 1))]
        return [label, label.capitalize() + ".", f"Label: {label}", f"**{label}**"][int(v * 4)]
    if task == "extraction":
        if good:
            d = dict(gold)
            if v < 0.25:
                return json.dumps(d)
            if v < 0.5:
                return json.dumps(dict(reversed(list(d.items()))), indent=2)
            if v < 0.75:
                return "```json\n" + json.dumps(d) + "\n```"
            return "Here is the extracted data:\n" + json.dumps(d)
        if h("break", model, entry["key"]) < entry.get("json_break", {}).get(model, 0.3):
            return "vendor: " + str(gold.get("vendor")) + ", total: " + str(gold.get("total"))   # not JSON
        d = dict(gold)
        field = sorted(d)[int(h("field", model, entry["key"]) * len(d))]
        d[field] = (d[field] + 1) if isinstance(d[field], (int, float)) else str(d[field]) + " Ltd"
        return json.dumps(d)
    # generation
    if good:
        return gold if v < 0.5 else "In short: " + gold
    return "I'm not certain, but it may relate to " + ["history", "weather", "sports", "music"][int(v * 4)] + "."


def gen_correct(entry: dict, text: str) -> bool:
    return norm(entry["gold"]).casefold() in norm(text).casefold()


def judge(entry: dict, a: str, b: str, judge_model: str) -> str:
    ca, cb = gen_correct(entry, a), gen_correct(entry, b)
    out = "TIE" if ca == cb else ("A" if ca else "B")
    if h("judge-noise", judge_model, entry["key"], a[:40], b[:40]) < 0.01:
        out = ["A", "B", "TIE"][int(h("jn2", entry["key"], a[:20]) * 3)]
    return out


def tokens_in(messages_text: str) -> int:
    return max(1, math.ceil(len(messages_text) / 4))


def tokens_out(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


# ------------------------------------------------------------------ server
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    wbufsize = 1 << 16

    def log_message(self, *a):
        pass

    def _send(self, code, obj, headers=None):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(b)

    def _log(self, **kw):
        with LOCK:
            with open(LOG, "a") as fh:
                fh.write(json.dumps(kw) + "\n")

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        path = urlparse(self.path).path
        if path == "/__register":
            with LOCK:
                for e in body["entries"]:
                    REGISTRY[e["key"]] = e
            return self._send(200, {"ok": len(body["entries"])})
        api, model, msgs = self._parse(path, body)
        if api is None:
            self._log(status=404, path=path)
            return self._send(404, {"error": {"message": f"zoo: no route {path}"}})
        auth = bool(self.headers.get("authorization") or self.headers.get("x-api-key") or self.headers.get("api-key")
                    or self.headers.get("x-goog-api-key") or "AWS4" in (self.headers.get("authorization") or ""))
        user = [m for m in msgs if m[0] == "user"]
        text_all = "".join(t for _, t in msgs)
        is_judge = bool(user) and user[-1][1].startswith(JUDGE_MARK)
        if is_judge:
            req = re.search(r"<request>\n(.*)\n</request>", user[-1][1], re.S)
            a = re.search(r"<response_a>\n(.*)\n</response_a>", user[-1][1], re.S)
            b = re.search(r"<response_b>\n(.*)\n</response_b>", user[-1][1], re.S)
            last_user = req.group(1).rsplit("[user] ", 1)[-1] if req else ""
            entry = REGISTRY.get(key_of(last_user))
        else:
            entry = REGISTRY.get(key_of(user[-1][1])) if user else None
        if entry is None:
            self._log(status=400, api=api, model=model, path=path, reason="unknown sample")
            return self._send(400, {"error": {"message": "zoo: unknown sample"}})
        purpose = "judge" if is_judge else "answer"
        akey = (model, entry["key"], purpose, text_all[-64:] if is_judge else "")
        with LOCK:
            ATTEMPTS[akey] = ATTEMPTS.get(akey, 0) + 1
            attempt = ATTEMPTS[akey]
        if attempt == 1:
            r = h("err", model, entry["key"], purpose, text_all[-32:])
            if r < 0.04:
                self._log(status=429, api=api, model=model, exp=entry["exp"], key=entry["key"], judge=is_judge, auth=auth)
                return self._send(429, {"error": {"message": "rate limited", "type": "rate_limit"}}, {"retry-after": "0"})
            if r < 0.055:
                self._log(status=500, api=api, model=model, exp=entry["exp"], key=entry["key"], judge=is_judge, auth=auth)
                return self._send(500, {"error": {"message": "upstream error"}})
        out = judge(entry, a.group(1) if a else "", b.group(1) if b else "", model) if is_judge else answer(entry, model)
        tin, tout = tokens_in(text_all), tokens_out(out)
        sig = hashlib.sha256(json.dumps(msgs).encode()).hexdigest()[:12]
        self._log(status=200, api=api, model=model, exp=entry["exp"], key=entry["key"], judge=is_judge, auth=auth,
                  input_tokens=tin, output_tokens=tout, host=self.headers.get("host"), sig=sig)
        return self._respond(api, model, out, tin, tout)

    def _parse(self, path, body):
        """(api, model, [(role, text)])"""
        def text(c):
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
            return ""
        m = re.search(r"/deployments/([^/]+)/chat/completions$", path)
        if m or path.endswith("/chat/completions"):
            return ("azure" if m else "openai"), (m.group(1) if m else body.get("model")), \
                [(x.get("role"), text(x.get("content"))) for x in body.get("messages", [])]
        if path.endswith("/v1/messages"):
            msgs = [("system", text(body["system"]))] if body.get("system") else []
            return "anthropic", body.get("model"), msgs + [(x.get("role"), text(x.get("content"))) for x in body.get("messages", [])]
        m = re.search(r"/models/([^/:]+):generateContent$", path)
        if m:
            msgs = [("system", text((body.get("systemInstruction") or {}).get("parts")))] if body.get("systemInstruction") else []
            return "gemini", m.group(1), msgs + [("assistant" if c.get("role") == "model" else "user", text(c.get("parts")))
                                               for c in body.get("contents", [])]
        m = re.match(r"^/model/(.+)/converse$", path)
        if m:
            msgs = [("system", text(body["system"]))] if body.get("system") else []
            return "bedrock", unquote(m.group(1)), msgs + [(x.get("role"), text(x.get("content"))) for x in body.get("messages", [])]
        return None, None, []

    def _respond(self, api, model, out, tin, tout):
        if api in ("openai", "azure"):
            return self._send(200, {"id": "z", "object": "chat.completion", "created": 1, "model": model,
                                    "choices": [{"index": 0, "message": {"role": "assistant", "content": out},
                                                 "finish_reason": "stop"}],
                                    "usage": {"prompt_tokens": tin, "completion_tokens": tout, "total_tokens": tin + tout}})
        if api == "anthropic":
            return self._send(200, {"id": "m", "type": "message", "role": "assistant", "model": model,
                                    "content": [{"type": "text", "text": out}], "stop_reason": "end_turn",
                                    "usage": {"input_tokens": tin, "output_tokens": tout}})
        if api == "gemini":
            return self._send(200, {"candidates": [{"content": {"role": "model", "parts": [{"text": out}]},
                                                    "finishReason": "STOP"}], "modelVersion": model,
                                    "usageMetadata": {"promptTokenCount": tin, "candidatesTokenCount": tout,
                                                      "totalTokenCount": tin + tout}})
        return self._send(200, {"output": {"message": {"role": "assistant", "content": [{"text": out}]}},
                                "stopReason": "end_turn", "usage": {"inputTokens": tin, "outputTokens": tout,
                                                                    "totalTokens": tin + tout},
                                "metrics": {"latencyMs": 3}})


if __name__ == "__main__":
    PORT, LOG = int(sys.argv[1]), sys.argv[2]
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer.request_queue_size = 4096
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"zoo on {PORT}", flush=True)
    srv.serve_forever()
