from tl import BASE
import openai

# Fireworks recommends its OpenAI-compatible endpoint; the native fireworks-ai SDK targets the same API.
c = openai.OpenAI(base_url=f"{BASE}/inference/v1", api_key="f")
c.chat.completions.create(model="accounts/fireworks/models/llama-v3p1-70b-instruct", messages=[{"role": "user", "content": "hi"}])  # @call
try:
    from fireworks.client import Fireworks
    f = Fireworks(api_key="f", base_url=f"{BASE}/inference/v1")
    f.chat.completions.create(model="accounts/fireworks/models/llama-v3p1-8b-instruct", messages=[{"role": "user", "content": "hi"}])  # @call
except ImportError:
    pass
