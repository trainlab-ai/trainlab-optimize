import asyncio

from tl import BASE
try:
    from mistralai import Mistral
except ImportError:
    from mistralai.client import Mistral

c = Mistral(api_key="m", server_url=BASE)
c.chat.complete(model="mistral-large-latest", messages=[{"role": "user", "content": "hi"}])  # @call
for _ in c.chat.stream(model="mistral-small-latest", messages=[{"role": "user", "content": "hi"}]):  # @call
    pass
c.embeddings.create(model="mistral-embed", inputs=["a"])  # @call


async def main():
    await c.chat.complete_async(model="mistral-large-latest", messages=[{"role": "user", "content": "hi"}])  # @call
asyncio.run(main())
