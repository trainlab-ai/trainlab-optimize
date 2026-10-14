"""Price catalog lookup and cost calculation."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib import resources

_DATE_SUFFIX = re.compile(r"-(20\d{2}-?\d{2}-?\d{2}|\d{4}|latest|preview)$")
_PREFIXES = ("openai/", "anthropic/", "azure/", "azure_ai/", "bedrock/", "gemini/",
             "vertex_ai/", "mistral/", "groq/", "together_ai/", "fireworks_ai/",
             "deepseek/", "openrouter/", "models/")


@dataclass(frozen=True)
class Price:
    model: str
    provider: str
    mode: str
    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None
    batch_input: float | None = None
    batch_output: float | None = None
    max_input_tokens: int | None = None
    deprecated: str | None = None
    via: str = "exact"          # "exact" | "name": matched by model name under another host's listing

    @property
    def approximate(self) -> bool:
        """Price borrowed from a host other than the model's maker (e.g. Groq Llama priced from Azure)."""
        return self.via == "name" and self.provider != "openrouter" and self.provider not in maker_providers(self.model)

    def cost(self, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0,
             cache_write_tokens: int = 0, batch: bool = False) -> float:
        uncached = max(input_tokens - cached_input_tokens - cache_write_tokens, 0)
        in_rate = (self.batch_input if batch and self.batch_input else self.input)
        out_rate = (self.batch_output if batch and self.batch_output else self.output)
        cr = self.cache_read if self.cache_read is not None else self.input
        cw = self.cache_write if self.cache_write is not None else self.input
        return uncached * in_rate + cached_input_tokens * cr + cache_write_tokens * cw + output_tokens * out_rate

    def blended_per_1k(self, in_out_ratio: float = 3.0) -> float:
        """Cost of 1k tokens at a typical input:output ratio. Used for ranking only."""
        total = in_out_ratio + 1
        return 1000 * (self.input * in_out_ratio + self.output) / total


# Our provider ids -> the provider prefix used in the price catalog
PROVIDER_PREFIX = {
    "together": "together_ai", "fireworks": "fireworks_ai", "zhipu": "zai", "vertex": "vertex_ai",
    "nvidia": "nvidia_nim", "qianfan": "qianfan", "huggingface": "huggingface",
}
# Who makes (and sells) each model family: their own list price is the reference price.
_MAKERS = [
    (r"^(gpt|o\d|chatgpt|text-embedding|davinci|babbage|codex|computer-use)", {"openai", "text-completion-openai"}),
    (r"^claude", {"anthropic"}),
    (r"^(gemini|gemma|text-embedding-00|embedding-00)", {"gemini"}),
    (r"^(mistral|mixtral|codestral|ministral|pixtral|magistral|devstral|open-mistral|voxtral)", {"mistral"}),
    (r"^deepseek", {"deepseek"}),
    (r"^grok", {"xai"}),
    (r"^(qwen|qwq|qvq)", {"dashscope", "qwencloud", "qwen_ai_platform"}),
    (r"^(kimi|moonshot)", {"moonshot"}),
    (r"^(glm|chatglm|cogview)", {"zai"}),
    (r"^minimax", {"minimax"}),
    (r"^(command|embed-|c4ai|aya)", {"cohere", "cohere_chat"}),
    (r"^doubao", {"volcengine"}),
    (r"^(hunyuan|hy\d)", {"tencent"}),
    (r"^(sonar|r1-1776)", {"perplexity"}),
    (r"^voyage", {"voyage"}),
    (r"^jamba", {"ai21"}),
]


def maker_providers(name: str) -> set[str]:
    n = _ROUTER_SUFFIX.sub("", name.lower()).rsplit("/", 1)[-1]
    n = re.sub(r"^((us|eu|apac|au|jp|global)\.)?[a-z0-9]+\.(?=[a-z])", "", n) if re.match(
        r"^((us|eu|apac|au|jp|global)\.)?(anthropic|meta|amazon|cohere|mistral|ai21|deepseek|qwen|openai|"
        r"google|moonshotai|minimax|zai)\.", n) else n
    for pat, makers in _MAKERS:
        if re.match(pat, n):
            return makers
    return set()
_NAME_DATE = re.compile(r"-(20\d{2}-?\d{2}-?\d{2}|\d{2}-20\d{2}|2\d{5}|\d{4,5}|latest|preview|instruct|versatile|specdec|instant)$")
_ROUTER_SUFFIX = re.compile(r"(:free|:beta|:extended|:thinking|:nitro|:floor|:online|@latest)$")


def _VENDOR_NORM(name: str) -> str:
    """claude-opus-4-1-20250805-v1:0 -> claude-opus-4-1"""
    name = re.sub(r"-v\d+(:\d+)?$", "", name)
    return re.sub(r"-(20\d{6}|\d{4}-\d{2}-\d{2})$", "", name)


def _name_keys(name: str) -> list[str]:
    """Normalised forms of a model's own name, without host or vendor path."""
    n = _ROUTER_SUFFIX.sub("", name.strip().lower()).rsplit("/", 1)[-1]
    if re.match(r"^(us|eu|apac|au|jp|global)\.", n):
        n = n.split(".", 1)[1]
    n = re.sub(r"^(anthropic|meta|amazon|cohere|mistral|ai21|deepseek|qwen|openai|google|moonshot(ai)?|"
               r"minimax|zai|writer|twelvelabs)\.", "", n)
    n = _VENDOR_NORM(n)
    keys = [n]
    for _ in range(2):
        stripped = _NAME_DATE.sub("", keys[-1])
        if stripped == keys[-1]:
            break
        keys.append(stripped)
    keys += [k.replace(".", "-") for k in keys if "." in k]
    return keys


def _provider_rank(provider: str, key: str) -> int:
    if provider in maker_providers(key):
        return 0
    if provider == "openrouter":
        return 1
    if provider.startswith(("vertex_ai", "bedrock", "azure")):
        return 2
    return 3


class PriceCatalog:
    def __init__(self, data: dict):
        self.meta = data.get("_meta", {})
        self._models: dict[str, dict] = data["models"]
        self._lower = {k.lower(): k for k in self._models}
        self._nidx: dict[str, str] | None = None

    @classmethod
    def load(cls, path: str | None = None) -> "PriceCatalog":
        if path:
            with open(path) as fh:
                return cls(json.load(fh))
        return _default_catalog()

    def __len__(self) -> int:
        return len(self._models)

    def _price(self, key: str) -> Price:
        e = self._models[key]
        return Price(model=key, provider=e["provider"], mode=e["mode"], input=e["in"],
                     output=e.get("out", 0.0), cache_read=e.get("cache_read"),
                     cache_write=e.get("cache_write"), batch_input=e.get("batch_in"),
                     batch_output=e.get("batch_out"), max_input_tokens=e.get("max_in"),
                     deprecated=e.get("deprecated"))

    def lookup(self, model: str | None, provider: str | None = None) -> Price | None:
        """Resolve a model name as written in code or billing to a catalog price.

        Order: exact name (with the provider's own catalog prefix first), then known spellings
        (date suffixes, regional prefixes), then the model's own name under any host,
        preferring the maker's list price, then OpenRouter, then clouds, then other hosts.
        """
        if not model:
            return None
        cands = list(self._candidates(model, provider))
        for cand in cands:
            key = self._lower.get(cand.lower())
            if key:
                return self._price(key)
        idx = self._name_index()
        for cand in cands:
            for nk in _name_keys(cand):
                key = idx.get(nk)
                if key:
                    return replace(self._price(key), via="name")
        return None

    def _name_index(self) -> dict[str, str]:
        if self._nidx is None:
            best: dict[str, tuple] = {}
            for k, e in self._models.items():
                rank = (_provider_rank(e["provider"], k), len(k), k)
                for nk in _name_keys(k):
                    if nk not in best or rank < best[nk]:
                        best[nk] = rank
            self._nidx = {nk: r[2] for nk, r in best.items()}
        return self._nidx

    def _candidates(self, model: str, provider: str | None):
        m = model.strip()
        if provider and provider not in ("openai", "anthropic", "unknown", "openai_compatible", "litellm"):
            yield f"{PROVIDER_PREFIX.get(provider, provider)}/{m}"
        yield m
        bare = m
        for p in _PREFIXES:
            if bare.lower().startswith(p):
                bare = bare[len(p):]
        if bare != m:
            yield bare
        # Bedrock regional prefixes: us.anthropic.claude-... -> anthropic.claude-...
        if re.match(r"^(us|eu|apac|au|jp|global)\.", bare):
            yield bare.split(".", 1)[1]
        # Strip date / version suffixes: gpt-4o-2024-08-06 -> gpt-4o
        stripped = bare
        for _ in range(2):
            new = _DATE_SUFFIX.sub("", stripped)
            if new == stripped:
                break
            stripped = new
            yield stripped
        # Anthropic dot versions: claude-3.5-sonnet -> claude-3-5-sonnet
        if "." in stripped and "claude" in stripped:
            yield stripped.replace(".", "-")

    def models(self, provider: str | None = None, mode: str | None = None):
        for k, e in self._models.items():
            if provider and e["provider"] != provider:
                continue
            if mode and e["mode"] != mode:
                continue
            yield self._price(k)


def user_catalog_path():
    import os
    from pathlib import Path
    env = os.environ.get("TRAINLAB_PRICES")
    return Path(env) if env else Path.home() / ".cache" / "trainlab" / "prices.json"


@lru_cache(maxsize=1)
def _default_catalog() -> PriceCatalog:
    """User cache (from `trainlab prices update`) if present and newer, else the bundled copy."""
    bundled = json.loads(resources.files("trainlab_optimize.catalog").joinpath("prices.json").read_text())
    try:
        user = json.loads(user_catalog_path().read_text())
        if user.get("_meta", {}).get("built_at", "") > bundled.get("_meta", {}).get("built_at", ""):
            return PriceCatalog(user)
    except (OSError, ValueError):
        pass
    return PriceCatalog(bundled)
