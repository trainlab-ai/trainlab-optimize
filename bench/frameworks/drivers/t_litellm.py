import asyncio

from tl import BASE
import litellm

litellm.completion(model="openai/gpt-4o-mini", api_base=f"{BASE}/v1", messages=[{"role": "user", "content": "hi"}])  # @call
litellm.completion(model="anthropic/claude-haiku-4-5", api_base=BASE, max_tokens=50, messages=[{"role": "user", "content": "hi"}])  # @call
for _ in litellm.completion(model="openai/gpt-4o-mini", api_base=f"{BASE}/v1", stream=True, stream_options={"include_usage": True}, messages=[{"role": "user", "content": "hi"}]):  # @call
    pass
litellm.embedding(model="openai/text-embedding-3-small", api_base=f"{BASE}/v1", input=["a"])  # @call


async def main():
    await litellm.acompletion(model="openai/gpt-4o", api_base=f"{BASE}/v1", messages=[{"role": "user", "content": "hi"}])  # @call
asyncio.run(main())
