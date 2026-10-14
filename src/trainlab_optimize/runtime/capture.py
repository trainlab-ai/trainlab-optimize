"""Opt-in capture of real inputs and outputs for experiments (Profile 3).

Off by default. When on, a sample of calls from the chosen call sites is written to
.trainlab/samples/<site>.jsonl (git-ignored), with emails, phone numbers, card numbers and
keys redacted unless redact=False. Nothing is sent anywhere; `trainlab experiment run`
reads these files.

    trainlab.instrument(capture={"sites": ["app/support.py:42"], "sample_rate": 0.05, "max_per_site": 300})
    trainlab.instrument(capture=True)       # every call site, 2% of calls, 200 per site
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
from pathlib import Path

from ..redact import redact
from .core import get

PARAMS = ("max_tokens", "max_completion_tokens", "max_output_tokens", "temperature", "top_p", "response_format",
          "stop", "seed")


class Capture:
    def __init__(self, directory: str | os.PathLike, sites=None, sample_rate: float = 0.02,
                 max_per_site: int = 200, do_redact: bool = True, tags: dict | None = None):
        self.dir = Path(directory)
        self.sites = {str(s) for s in sites} if sites else None
        self.tags = dict(tags or {})
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.max_per_site = max_per_site
        self.redact = do_redact
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_option(cls, option, root: str) -> "Capture | None":
        if not option:
            return None
        d = option if isinstance(option, dict) else {}
        return cls(d.get("path") or os.path.join(root, ".trainlab", "samples"), d.get("sites"),
                   d.get("sample_rate", 1.0 if d.get("sites") else 0.02), d.get("max_per_site", 200 if not d.get("sites") else 500),
                   d.get("redact", True), d.get("tags"))

    def site_key(self, ev) -> str | None:
        return f"{ev.path}:{ev.line}" if ev.path and ev.line else None

    def wants(self, ev) -> bool:
        key = self.site_key(ev)
        if key is None or not ev.ok:
            return False
        if self.sites is not None and key not in self.sites and ev.path not in self.sites:
            return False
        if self.tags and any(str(ev.tags.get(k)) != str(v) for k, v in self.tags.items()):
            return False
        if self._counts.get(key, 0) >= self.max_per_site and key in self._counts:
            return False
        return self.sample_rate >= 1.0 or random.random() < self.sample_rate

    def _file(self, key: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("_")[:150]
        return self.dir / f"{slug}.jsonl"

    def write(self, ev, request: dict, output: str | None) -> None:
        if output is None or not isinstance(request, dict):
            return
        from ..experiment.samples import normalize_messages
        system = request.get("system") or request.get("instructions")
        raw = request.get("messages") or request.get("contents") or request.get("input") or request.get("prompt")
        msgs = normalize_messages(raw, system)
        if not msgs:
            return
        if self.redact:
            msgs = [{"role": m["role"], "content": redact(m["content"]) or ""} for m in msgs]
            output = redact(output) or ""
        key = self.site_key(ev)
        rec = {"v": 1, "site": key, "model": ev.model, "provider": ev.provider, "messages": msgs,
               "params": {k: request[k] for k in PARAMS if request.get(k) is not None and _jsonable(request[k])},
               "output": output, "input_tokens": ev.input_tokens, "output_tokens": ev.output_tokens,
               "tags": ev.tags or None, "ts": ev.ts or None, "redacted": self.redact}
        line = (json.dumps({k: v for k, v in rec.items() if v is not None}, ensure_ascii=False) + "\n").encode()
        with self._lock:
            f = self._file(key)
            if key not in self._counts:
                self.dir.mkdir(parents=True, exist_ok=True)
                gi = self.dir.parent / ".gitignore"
                if self.dir.parent.name == ".trainlab" and not gi.exists():
                    try:
                        gi.write_text("*\n")
                    except OSError:
                        pass
                self._counts[key] = sum(1 for _ in open(f, "rb")) if f.exists() else 0
            if self._counts[key] >= self.max_per_site:
                return
            fd = os.open(f, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
            self._counts[key] += 1


def _jsonable(v) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def result_text(result) -> str | None:
    """The answer text of an SDK result (OpenAI chat/completions/responses, Anthropic, Gemini,
    LiteLLM, Bedrock Converse)."""
    if result is None:
        return None
    t = get(result, "output_text")
    if isinstance(t, str):
        return t
    choices = get(result, "choices")
    if choices:
        ch = choices[0]
        msg = get(ch, "message")
        if msg is not None:
            content = get(msg, "content")
            if isinstance(content, str):
                return content
            calls = get(msg, "tool_calls")
            if calls:
                return json.dumps([{"name": get(c, "function", "name"), "arguments": get(c, "function", "arguments")}
                                   for c in calls])
        if isinstance(get(ch, "text"), str):
            return get(ch, "text")
    content = get(result, "content")
    if isinstance(content, list):
        parts = [get(b, "text") for b in content if get(b, "type") == "text"]
        if parts:
            return "".join(p for p in parts if p)
    t = None
    try:
        t = getattr(result, "text", None)                 # Gemini (property)
    except Exception:  # noqa: BLE001
        t = None
    if isinstance(t, str):
        return t
    msg = get(result, "output", "message", "content")
    if isinstance(msg, list):
        return "".join(get(c, "text") or "" for c in msg)
    return None


def request_of(kwargs: dict) -> dict:
    """The parts of an SDK call's keyword arguments a sample needs."""
    req = {k: kwargs.get(k) for k in ("messages", "system", "instructions", "input", "contents", "prompt") + PARAMS
           if kwargs.get(k) is not None}
    cfg = kwargs.get("config")                           # google-genai GenerateContentConfig
    if cfg is not None:
        si = get(cfg, "system_instruction")
        if si is not None and "system" not in req:
            req["system"] = si if isinstance(si, str) else json.dumps(si, default=str)
        mt = get(cfg, "max_output_tokens")
        if mt:
            req["max_tokens"] = mt
    return req
