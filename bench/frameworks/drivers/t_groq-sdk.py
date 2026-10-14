import asyncio

from tl import BASE
import groq

c = groq.Groq(base_url=BASE)
c.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": "hi"}])  # @call
for _ in c.chat.completions.create(model="llama-3.1-8b-instant", stream=True, messages=[{"role": "user", "content": "hi"}]):  # @call
    pass


async def main():
    await groq.AsyncGroq(base_url=BASE).chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": "hi"}])  # @call
asyncio.run(main())
