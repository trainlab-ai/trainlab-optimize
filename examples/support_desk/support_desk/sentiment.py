"""Score customer sentiment for every ticket concurrently with the async client."""
from __future__ import annotations

import asyncio

from .llm import aclient, tag

SYSTEM = "Rate the customer's mood in this support ticket. Reply with one word: calm, annoyed or angry."


async def _one(ticket: dict, gate: asyncio.Semaphore) -> str:
    async with gate:
        r = await aclient().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=3,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": ticket["body"]}],
        )
    word = (r.choices[0].message.content or "").strip().lower().strip(".")
    return word if word in ("calm", "annoyed", "angry") else "calm"


async def _all(tickets: list[dict], concurrency: int) -> list[str]:
    gate = asyncio.Semaphore(concurrency)
    with tag(feature="sentiment", team="support-ops"):
        return await asyncio.gather(*(_one(t, gate) for t in tickets))


def score_all(tickets: list[dict], concurrency: int = 8) -> list[str]:
    return asyncio.run(_all(tickets, concurrency))
