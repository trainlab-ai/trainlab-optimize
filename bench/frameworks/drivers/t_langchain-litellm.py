from tl import BASE
from langchain_litellm import ChatLiteLLM

ChatLiteLLM(model="openai/gpt-4o-mini", api_base=f"{BASE}/v1", api_key="k").invoke("hi")  # @call
