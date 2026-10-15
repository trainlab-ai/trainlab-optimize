"""Calling models with the customer's own keys (BYOK). Standard library only.

A model name resolves to a route: which API format, which endpoint, which key variable.
  gpt-*, o1/o3/o4*, openai/...           OpenAI            OPENAI_API_KEY      (OPENAI_BASE_URL)
  claude-*, anthropic/...                Anthropic         ANTHROPIC_API_KEY   (ANTHROPIC_BASE_URL)
  gemini-*, gemini/...                   Gemini API        GEMINI_API_KEY / GOOGLE_API_KEY
  azure/<deployment>                     Azure OpenAI      AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT
  groq/, together_ai/, fireworks_ai/, deepseek/, mistral/, dashscope/, xai/, openrouter/,
  moonshot/, zhipu/, perplexity/, ollama/  OpenAI-compatible endpoint of that provider
  bedrock/... or anthropic.* / amazon.* / meta.* ids    Bedrock Converse (needs boto3)
  anything else                          LiteLLM, if installed
Every base URL can be overridden with TRAINLAB_BASE_<PROVIDER> (e.g. TRAINLAB_BASE_GROQ), so a
gateway or proxy can sit in between. Keys are read from the environment and never written.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

# provider -> (default base URL, key variables, model prefix to strip)
COMPAT = {
    "openai": ("https://api.openai.com/v1", ("OPENAI_API_KEY",)),
    "groq": ("https://api.groq.com/openai/v1", ("GROQ_API_KEY",)),
    "together": ("https://api.together.xyz/v1", ("TOGETHER_API_KEY", "TOGETHERAI_API_KEY")),
    "fireworks": ("https://api.fireworks.ai/inference/v1", ("FIREWORKS_API_KEY",)),
    "deepseek": ("https://api.deepseek.com/v1", ("DEEPSEEK_API_KEY",)),
    "mistral": ("https://api.mistral.ai/v1", ("MISTRAL_API_KEY",)),
    "dashscope": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", ("DASHSCOPE_API_KEY",)),
    "xai": ("https://api.x.ai/v1", ("XAI_API_KEY",)),
    "openrouter": ("https://openrouter.ai/api/v1", ("OPENROUTER_API_KEY",)),
    "moonshot": ("https://api.moonshot.ai/v1", ("MOONSHOT_API_KEY",)),
    "zhipu": ("https://open.bigmodel.cn/api/paas/v4", ("ZHIPUAI_API_KEY",)),
    "perplexity": ("https://api.perplexity.ai", ("PERPLEXITYAI_API_KEY", "PERPLEXITY_API_KEY")),
    "cohere": ("https://api.cohere.ai/compatibility/v1", ("CO_API_KEY", "COHERE_API_KEY")),
    "ollama": ("http://localhost:11434/v1", ()),
}
PREFIX = {"together_ai": "together", "fireworks_ai": "fireworks", "openai": "openai", "groq": "groq",
          "deepseek": "deepseek", "mistral": "mistral", "dashscope": "dashscope", "xai": "xai",
          "openrouter": "openrouter", "moonshot": "moonshot", "zhipu": "zhipu", "perplexity": "perplexity",
          "ollama": "ollama", "ollama_chat": "ollama", "cohere": "cohere", "cohere_chat": "cohere"}
_BEDROCK_ID = re.compile(r"^((global|us|eu|apac|jp|au|ca|us-gov)\.)?(anthropic|amazon|meta|mistral|cohere|ai21|deepseek|qwen|openai|writer)\.")
_REASONING = re.compile(r"^(o\d|gpt-5)", re.I)


@dataclass
class Route:
    api: str                 # openai | anthropic | gemini | azure | bedrock | litellm
    provider: str
    model: str               # id sent to the API
    base: str | None
    key_env: tuple[str, ...]

    def key(self) -> str | None:
        for k in self.key_env:
            if os.environ.get(k):
                return os.environ[k]
        return None

    @property
    def host(self) -> str:
        if not self.base:
            return self.api
        m = re.match(r"https?://([^/]+)", self.base)
        return m.group(1) if m else self.base


@dataclass
class Result:
    text: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int = 0
    latency_ms: float | None = None
    error: str | None = None
    status: int | None = None
    attempts: int = 1
    served_model: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.text is not None


def _base(provider: str, default: str | None) -> str | None:
    env = os.environ.get(f"TRAINLAB_BASE_{provider.upper()}")
    if env:
        return env.rstrip("/")
    if provider == "openai" and os.environ.get("OPENAI_BASE_URL"):
        return os.environ["OPENAI_BASE_URL"].rstrip("/")
    if provider == "anthropic" and os.environ.get("ANTHROPIC_BASE_URL"):
        return os.environ["ANTHROPIC_BASE_URL"].rstrip("/")
    if provider == "ollama" and os.environ.get("OLLAMA_HOST"):
        return os.environ["OLLAMA_HOST"].rstrip("/") + "/v1"
    return default


def resolve(model: str, provider: str | None = None) -> Route:
    """provider: where the call is served today, when known (the call site's provider). It decides
    the route for names that don't say it themselves (llama3-70b-8192 on Groq, an org/model id on
    a self-hosted OpenAI-compatible server)."""
    m = model.strip()
    prefix, _, rest = m.partition("/")
    p = prefix.lower()
    hint = (provider or "").lower()
    if hint == "bedrock" or _BEDROCK_ID.match(m):
        return Route("bedrock", "bedrock", rest if p == "bedrock" else m, _base("bedrock", None), ())
    if hint in COMPAT and hint != "openai" and p not in PREFIX:
        base, keys = COMPAT[hint]
        return Route("openai", hint, m, _base(hint, base), keys)
    if hint == "openai_compatible" and p not in PREFIX and not m.lower().startswith(("claude", "gemini")):
        return Route("openai", "openai_compatible", m, _base("openai_compatible", _base("openai", COMPAT["openai"][0])),
                     ("TRAINLAB_API_KEY", "OPENAI_API_KEY"))
    if p == "azure" and rest:
        base = os.environ.get("TRAINLAB_BASE_AZURE") or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        return Route("azure", "azure", rest, base.rstrip("/") or None, ("AZURE_OPENAI_API_KEY", "AZURE_API_KEY"))
    if p == "anthropic" or (not rest and m.lower().startswith("claude")):
        return Route("anthropic", "anthropic", rest if p == "anthropic" else m,
                     _base("anthropic", "https://api.anthropic.com"), ("ANTHROPIC_API_KEY",))
    if p in ("gemini", "google") or (not rest and m.lower().startswith("gemini")):
        return Route("gemini", "gemini", rest if p in ("gemini", "google") else m,
                     _base("gemini", "https://generativelanguage.googleapis.com"), ("GEMINI_API_KEY", "GOOGLE_API_KEY"))
    if p == "bedrock":
        return Route("bedrock", "bedrock", rest, _base("bedrock", None), ())
    if p in PREFIX and rest:
        prov = PREFIX[p]
        base, keys = COMPAT[prov]
        return Route("openai", prov, rest, _base(prov, base), keys)
    if not rest and re.match(r"^(gpt-|o\d|chatgpt|text-embedding|davinci|babbage)", m, re.I):
        return Route("openai", "openai", m, _base("openai", COMPAT["openai"][0]), ("OPENAI_API_KEY",))
    if hint == "openrouter" or (rest and hint in ("", "unknown") and p in ("mistralai", "meta-llama", "qwen", "z-ai",
                                                                         "moonshotai", "deepseek-ai", "google")):
        base, keys = COMPAT["openrouter"]
        return Route("openai", "openrouter", m, _base("openrouter", base), keys)
    if not rest:
        from ..scanner.providers import maker_of
        mk = maker_of(m)
        if mk in COMPAT:
            base, keys = COMPAT[mk]
            return Route("openai", mk, m, _base(mk, base), keys)
    return Route("litellm", "litellm", m, None, ())


# ------------------------------------------------------------------ request building
def _system_and_turns(messages: list[dict]) -> tuple[str, list[dict]]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [m for m in messages if m["role"] != "system"]
    return system, turns


def _max_tokens(params: dict, default: int | None = None) -> int | None:
    for k in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        if isinstance(params.get(k), int):
            return params[k]
    return default


def build_request(route: Route, messages: list[dict], params: dict) -> tuple[str, dict, dict]:
    """(url, headers, body) for one call, in the route's API format."""
    key = route.key() or ""
    mt = _max_tokens(params)
    if route.api in ("openai", "azure"):
        body: dict = {"model": route.model, "messages": messages}
        reasoning = bool(_REASONING.match(route.model))
        if mt is not None:
            body["max_completion_tokens" if reasoning else "max_tokens"] = mt
        if params.get("temperature") is not None and not reasoning:
            body["temperature"] = params["temperature"]
        for k in ("top_p", "stop", "seed", "response_format"):
            if params.get(k) is not None and not (reasoning and k == "top_p"):
                body[k] = params[k]
        if route.api == "azure":
            ver = os.environ.get("OPENAI_API_VERSION", "2024-10-21")
            url = f"{route.base}/openai/deployments/{route.model}/chat/completions?api-version={ver}"
            return url, {"api-key": key, "content-type": "application/json"}, body
        hdr = {"content-type": "application/json"}
        if key:
            hdr["authorization"] = f"Bearer {key}"
        return f"{route.base}/chat/completions", hdr, body
    if route.api == "anthropic":
        system, turns = _system_and_turns(messages)
        body = {"model": route.model, "max_tokens": mt or 1024, "messages": turns}
        if system:
            body["system"] = system
        if params.get("temperature") is not None:
            body["temperature"] = params["temperature"]
        if params.get("stop"):
            body["stop_sequences"] = params["stop"] if isinstance(params["stop"], list) else [params["stop"]]
        return (f"{route.base}/v1/messages",
                {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}, body)
    if route.api == "gemini":
        system, turns = _system_and_turns(messages)
        body = {"contents": [{"role": "model" if t["role"] == "assistant" else "user", "parts": [{"text": t["content"]}]}
                             for t in turns]}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        gc: dict = {}
        if mt is not None:
            gc["maxOutputTokens"] = mt
        if params.get("temperature") is not None:
            gc["temperature"] = params["temperature"]
        rf = params.get("response_format")
        if isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema"):
            gc["responseMimeType"] = "application/json"
        if gc:
            body["generationConfig"] = gc
        return (f"{route.base}/v1beta/models/{route.model}:generateContent",
                {"x-goog-api-key": key, "content-type": "application/json"}, body)
    raise ValueError(f"no HTTP request format for {route.api}")


def parse_response(route: Route, d: dict) -> Result:
    if route.api in ("openai", "azure"):
        ch = (d.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        text = msg.get("content")
        if text is None and msg.get("tool_calls"):
            text = json.dumps([tc.get("function") for tc in msg["tool_calls"]])
        u = d.get("usage") or {}
        return Result(text=text, input_tokens=u.get("prompt_tokens"), output_tokens=u.get("completion_tokens"),
                      cached_input_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                      served_model=d.get("model"))
    if route.api == "anthropic":
        text = "".join(b.get("text", "") for b in d.get("content") or [] if b.get("type") == "text")
        u = d.get("usage") or {}
        cr, cw = u.get("cache_read_input_tokens") or 0, u.get("cache_creation_input_tokens") or 0
        inp = u.get("input_tokens")
        return Result(text=text, input_tokens=None if inp is None else inp + cr + cw,
                      output_tokens=u.get("output_tokens"), cached_input_tokens=cr, served_model=d.get("model"))
    if route.api == "gemini":
        cands = d.get("candidates") or [{}]
        parts = ((cands[0] or {}).get("content") or {}).get("parts") or []
        u = d.get("usageMetadata") or {}
        out = (u.get("candidatesTokenCount") or 0) + (u.get("thoughtsTokenCount") or 0)
        return Result(text="".join(p.get("text", "") for p in parts), input_tokens=u.get("promptTokenCount"),
                      output_tokens=out if u else None, cached_input_tokens=u.get("cachedContentTokenCount") or 0,
                      served_model=d.get("modelVersion"))
    raise ValueError(route.api)


# ------------------------------------------------------------------ calling
# 52x: Cloudflare in front of the provider (520 unknown error, 522/524 timeouts), seen on api.openai.com
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529, 530}


def call(route: Route, messages: list[dict], params: dict, *, timeout: float = 120, max_attempts: int = 5,
         sleep=time.sleep) -> Result:
    """One model call with retries (429/5xx/network) and exponential backoff with jitter."""
    if route.api == "bedrock":
        return _call_bedrock(route, messages, params)
    if route.api == "litellm":
        return _call_litellm(route, messages, params)
    url, headers, body = build_request(route, messages, params)
    data = json.dumps(body).encode()
    attempt, last = 0, None
    while attempt < max_attempts:
        attempt += 1
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read() or b"{}")
            r = parse_response(route, payload)
            r.latency_ms, r.attempts, r.status = round((time.perf_counter() - t0) * 1000, 1), attempt, 200
            return r
        except urllib.error.HTTPError as e:
            status = e.code
            try:
                detail = json.loads(e.read() or b"{}")
                msg = (detail.get("error") or {}).get("message") if isinstance(detail.get("error"), dict) \
                    else detail.get("error") or detail.get("message")
            except (ValueError, AttributeError):
                msg = None
            last = Result(text=None, error=f"HTTP {status}" + (f": {str(msg)[:160]}" if msg else ""), status=status,
                          attempts=attempt)
            if status not in RETRY_STATUS:
                return last
            ra = e.headers.get("retry-after") if e.headers else None
            wait = float(ra) if ra and re.match(r"^\d+(\.\d+)?$", ra) else min(30.0, 0.5 * 2 ** (attempt - 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = Result(text=None, error=f"{type(e).__name__}: {getattr(e, 'reason', e)}"[:200], attempts=attempt)
            wait = min(30.0, 0.5 * 2 ** (attempt - 1))
        except ValueError as e:
            return Result(text=None, error=f"bad response: {e}"[:200], attempts=attempt)
        if attempt < max_attempts:
            sleep(wait * (0.75 + random.random() * 0.5))
    return last or Result(text=None, error="no attempt made")


def _call_bedrock(route: Route, messages: list[dict], params: dict) -> Result:
    try:
        import boto3
    except ImportError:
        return Result(text=None, error="Bedrock needs boto3 (pip install boto3)")
    system, turns = _system_and_turns(messages)
    kw = {"modelId": route.model,
          "messages": [{"role": t["role"], "content": [{"text": t["content"]}]} for t in turns]}
    if system:
        kw["system"] = [{"text": system}]
    inf = {}
    if _max_tokens(params) is not None:
        inf["maxTokens"] = _max_tokens(params)
    if params.get("temperature") is not None:
        inf["temperature"] = params["temperature"]
    if inf:
        kw["inferenceConfig"] = inf
    client = boto3.client("bedrock-runtime", endpoint_url=route.base) if route.base else boto3.client("bedrock-runtime")
    t0 = time.perf_counter()
    try:
        r = client.converse(**kw)
    except Exception as e:  # noqa: BLE001
        return Result(text=None, error=f"{type(e).__name__}: {e}"[:200])
    u = r.get("usage") or {}
    text = "".join(c.get("text", "") for c in ((r.get("output") or {}).get("message") or {}).get("content", []))
    cr = u.get("cacheReadInputTokens") or 0
    return Result(text=text, input_tokens=(u.get("inputTokens") or 0) + cr + (u.get("cacheWriteInputTokens") or 0),
                  output_tokens=u.get("outputTokens"), cached_input_tokens=cr,
                  latency_ms=round((time.perf_counter() - t0) * 1000, 1), status=200)


def _call_litellm(route: Route, messages: list[dict], params: dict) -> Result:
    try:
        import litellm
    except ImportError:
        return Result(text=None, error=f"no built-in route for '{route.model}'; install litellm to call it")
    t0 = time.perf_counter()
    try:
        r = litellm.completion(model=route.model, messages=messages,
                               **{k: v for k, v in params.items() if k in ("max_tokens", "temperature", "stop",
                                                                           "response_format", "top_p")})
    except Exception as e:  # noqa: BLE001
        return Result(text=None, error=f"{type(e).__name__}: {e}"[:200])
    u = r.get("usage") or {}
    return Result(text=r["choices"][0]["message"]["content"], input_tokens=u.get("prompt_tokens"),
                  output_tokens=u.get("completion_tokens"), latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                  status=200, served_model=r.get("model"))
