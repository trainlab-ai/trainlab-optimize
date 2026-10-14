from tl import BASE
from guidance import assistant, gen, models, user

lm = models.OpenAI("gpt-4o-mini", base_url=f"{BASE}/v1", api_key="k")
with user():
    lm += "hi"
with assistant():
    lm += gen("answer", max_tokens=10)  # @call
