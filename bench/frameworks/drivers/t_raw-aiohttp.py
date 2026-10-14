import asyncio

from tl import BASE
import aiohttp


async def main():
    async with aiohttp.ClientSession() as s:
        async with s.post(f"{BASE}/v1/chat/completions", json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}) as r:  # @call
            await r.json()
asyncio.run(main())
