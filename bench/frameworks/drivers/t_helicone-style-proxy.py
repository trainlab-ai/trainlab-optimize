"""OpenAI SDK pointed at a router/gateway (OpenRouter-style model names)."""
from tl import BASE
import openai

c = openai.OpenAI(base_url=f"{BASE}/api/v1", api_key="or-key")
c.chat.completions.create(model="anthropic/claude-3.5-sonnet", messages=[{"role": "user", "content": "hi"}])  # @call
c.chat.completions.create(model="qwen/qwen-2.5-72b-instruct", messages=[{"role": "user", "content": "hi"}])  # @call
