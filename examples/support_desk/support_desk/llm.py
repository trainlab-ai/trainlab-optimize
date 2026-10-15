"""OpenAI clients and an optional trainlab tag helper.

The app has no hard dependency on trainlab: `trainlab trace -- python run.py` records every
call without any code change. When trainlab is installed, `tag()` labels calls by feature and
team so the report can split cost that way; without it, `tag()` does nothing.
"""
from __future__ import annotations

from contextlib import nullcontext

from openai import AsyncOpenAI, OpenAI

try:
    from trainlab_optimize import tag
except ImportError:                       # trainlab not installed: tags are a no-op
    def tag(**_labels):
        return nullcontext()

_client: OpenAI | None = None
_aclient: AsyncOpenAI | None = None


def client() -> OpenAI:
    """Reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment, like any OpenAI app."""
    global _client
    if _client is None:
        _client = OpenAI(max_retries=3, timeout=60)
    return _client


def aclient() -> AsyncOpenAI:
    global _aclient
    if _aclient is None:
        _aclient = AsyncOpenAI(max_retries=3, timeout=60)
    return _aclient
