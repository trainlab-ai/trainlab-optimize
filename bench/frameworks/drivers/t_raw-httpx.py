import asyncio

from tl import BASE
import httpx

httpx.post(f"{BASE}/v1/chat/completions", json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]})  # @call
with httpx.Client() as c:
    c.post(f"{BASE}/v1/embeddings", json={"model": "text-embedding-3-small", "input": "a"})  # @call


async def main():
    async with httpx.AsyncClient() as c:
        await c.post(f"{BASE}/v1/chat/completions", json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})  # @call
asyncio.run(main())
