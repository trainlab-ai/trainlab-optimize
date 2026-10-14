import asyncio

from tl import BASE
import openai

c = openai.OpenAI(base_url=f"{BASE}/v1")
c.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])  # @call
for _ in c.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], stream=True, stream_options={"include_usage": True}):  # @call
    pass
c.responses.create(model="gpt-5-mini", input="hi")  # @call
c.embeddings.create(model="text-embedding-3-small", input=["a", "b"])  # @call
c.completions.create(model="gpt-3.5-turbo-instruct", prompt="hi")  # @call


async def main():
    a = openai.AsyncOpenAI(base_url=f"{BASE}/v1")
    await a.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])  # @call
    async for _ in await a.responses.create(model="gpt-5-mini", input="hi", stream=True):  # @call
        pass
asyncio.run(main())
