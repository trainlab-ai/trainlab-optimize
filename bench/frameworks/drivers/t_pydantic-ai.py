from tl import BASE
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

oa = Agent(OpenAIChatModel("gpt-4o", provider=OpenAIProvider(base_url=f"{BASE}/v1", api_key="k")))
oa.run_sync("hi")  # @call
an = Agent(AnthropicModel("claude-sonnet-4-5", provider=AnthropicProvider(api_key="k", base_url=BASE)))
an.run_sync("hi")  # @call
