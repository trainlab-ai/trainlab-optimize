"""Langfuse's drop-in OpenAI wrapper."""
from tl import BASE
from langfuse.openai import openai

openai.OpenAI(base_url=f"{BASE}/v1").chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])  # @call
