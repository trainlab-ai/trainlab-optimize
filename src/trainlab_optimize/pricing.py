"""Price catalog lookup and cost calculation."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
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


class PriceCatalog:
    def __init__(self, data: dict):
        self.meta = data.get("_meta", {})
        self._models: dict[str, dict] = data["models"]
        self._lower = {k.lower(): k for k in self._models}

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
        """Resolve a model name as written in code or billing to a catalog price."""
        if not model:
            return None
        for cand in self._candidates(model, provider):
            key = self._lower.get(cand.lower())
            if key:
                return self._price(key)
        return None

    def _candidates(self, model: str, provider: str | None):
        m = model.strip()
        yield m
        if provider in ("azure", "bedrock", "gemini", "mistral", "groq", "deepseek"):
            yield f"{provider}/{m}"
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


@lru_cache(maxsize=1)
def _default_catalog() -> PriceCatalog:
    text = resources.files("trainlab_optimize.catalog").joinpath("prices.json").read_text()
    return PriceCatalog(json.loads(text))
