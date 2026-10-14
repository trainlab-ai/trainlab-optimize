import asyncio

from tl import BASE
from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

k = Kernel()
k.add_service(OpenAIChatCompletion(ai_model_id="gpt-4o-mini", async_client=AsyncOpenAI(base_url=f"{BASE}/v1", api_key="k")))
asyncio.run(k.invoke_prompt("hi"))  # @call
