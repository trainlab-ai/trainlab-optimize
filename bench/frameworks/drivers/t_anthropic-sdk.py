import asyncio

from tl import BASE
import anthropic

c = anthropic.Anthropic(base_url=BASE)
c.messages.create(model="claude-sonnet-4-5", max_tokens=50, messages=[{"role": "user", "content": "hi"}])  # @call
with c.messages.stream(model="claude-haiku-4-5", max_tokens=50, messages=[{"role": "user", "content": "hi"}]) as s:  # @call
    s.get_final_text()
for _ in c.messages.create(model="claude-haiku-4-5", max_tokens=50, stream=True, messages=[{"role": "user", "content": "hi"}]):  # @call
    pass


async def main():
    a = anthropic.AsyncAnthropic(base_url=BASE)
    await a.messages.create(model="claude-opus-4-1", max_tokens=50, messages=[{"role": "user", "content": "hi"}])  # @call
asyncio.run(main())
