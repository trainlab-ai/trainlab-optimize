from tl import BASE
from smolagents import LiteLLMModel, OpenAIServerModel
from smolagents.models import ChatMessage

msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
OpenAIServerModel(model_id="gpt-4o-mini", api_base=f"{BASE}/v1", api_key="k").generate(msgs)  # @call
LiteLLMModel(model_id="openai/gpt-4o-mini", api_base=f"{BASE}/v1", api_key="k").generate(msgs)  # @call
