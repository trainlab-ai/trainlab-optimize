"""The daily support run: triage, extract, score mood, answer, then write the digest."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from .agent import answer_shipping
from .digest import daily_digest
from .extract import extract_fields
from .kb import KnowledgeBase
from .reply import draft_reply
from .sentiment import score_all
from .triage import classify


def _step(name: str):
    print(f"  {name} ...", flush=True)
    return time.perf_counter()


def process(tickets: list[dict], workers: int = 8, show_reply: bool = False) -> dict:
    t0 = _step(f"triage + field extraction for {len(tickets)} tickets ({workers} threads)")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        labels = list(pool.map(classify, tickets))
        fields = list(pool.map(extract_fields, tickets))
    correct = sum(1 for t, l in zip(tickets, labels) if l == t["label"])
    print(f"    routed {correct}/{len(tickets)} to the right queue ({time.perf_counter() - t0:.1f}s)")

    t0 = _step("sentiment (async, 8 at a time)")
    moods = score_all(tickets)
    print(f"    {moods.count('angry')} angry, {moods.count('annoyed')} annoyed ({time.perf_counter() - t0:.1f}s)")

    t0 = _step("help-centre index (embeddings)")
    kb = KnowledgeBase()

    t0 = _step("replies: shipping agent with order lookup, everything else from the help centre")

    def answer(pair):
        ticket, label = pair
        if label == "shipping":
            return answer_shipping(ticket)
        return draft_reply(ticket, kb.search(f"{ticket['subject']} {ticket['body']}"))

    with ThreadPoolExecutor(max_workers=max(1, workers // 2)) as pool:
        replies = list(pool.map(answer, zip(tickets, labels)))
    print(f"    {len(replies)} replies drafted ({time.perf_counter() - t0:.1f}s)")
    if show_reply and replies:
        print("\n    Example reply to " + tickets[0]["id"] + ":\n    " + replies[0].replace("\n", "\n    ") + "\n")

    _step("daily digest (Responses API)")
    digest = daily_digest(tickets, labels, moods, fields)
    return {"labels": labels, "fields": fields, "moods": moods, "replies": replies, "digest": digest,
            "triage_accuracy": correct / len(tickets) if tickets else None}
