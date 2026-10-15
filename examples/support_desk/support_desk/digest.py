"""A daily digest for the support lead, written with the Responses API."""
from __future__ import annotations

from collections import Counter

from .llm import client, tag

INSTRUCTIONS = ("You write a daily digest for the head of support. Use plain sentences, no more than 150 words: "
                "volume by queue, the most common problems, anything urgent, and one suggestion.")


def daily_digest(tickets: list[dict], labels: list[str], moods: list[str], fields: list[dict]) -> str:
    by_queue = Counter(labels)
    angry = [t["subject"] for t, m in zip(tickets, moods) if m == "angry"]
    urgent = [f.get("summary", "") for f in fields if f.get("urgency") == "high"]
    lines = [f"{len(tickets)} tickets today. By queue: " + ", ".join(f"{q} {n}" for q, n in by_queue.most_common()),
             "Subjects: " + "; ".join(t["subject"] for t in tickets[:40]),
             "Angry customers: " + ("; ".join(angry[:10]) or "none"),
             "Marked urgent: " + ("; ".join(urgent[:10]) or "none")]
    with tag(feature="daily-digest", team="support"):
        r = client().responses.create(model="gpt-4.1", instructions=INSTRUCTIONS, input="\n".join(lines),
                                      max_output_tokens=400)
    return r.output_text
