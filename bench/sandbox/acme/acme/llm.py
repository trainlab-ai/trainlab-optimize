"""Acme's LLM layer: real SDK clients pointed at BASE (the fake server in the sandbox)."""
import os

import anthropic
import openai
from google import genai
from google.genai import types

BASE = os.environ.get("LLM_BASE", "http://127.0.0.1:8765")

oa = openai.OpenAI(base_url=f"{BASE}/v1", api_key="sk-sandbox", max_retries=0)
aoa = openai.AsyncOpenAI(base_url=f"{BASE}/v1", api_key="sk-sandbox", max_retries=0)
an = anthropic.Anthropic(base_url=BASE, api_key="sk-ant-sandbox", max_retries=0)
aan = anthropic.AsyncAnthropic(base_url=BASE, api_key="sk-ant-sandbox", max_retries=0)
gm = genai.Client(api_key="g-sandbox", http_options=types.HttpOptions(base_url=BASE))


def h(feature):
    return {"x-truth-feature": feature}
