from tl import BASE
from txtai import LLM

llm = LLM("openai/gpt-4o-mini", method="litellm", api_base=f"{BASE}/v1")
llm("hi")  # @call
