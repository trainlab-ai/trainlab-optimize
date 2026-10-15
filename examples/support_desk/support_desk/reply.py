"""Draft a reply grounded in help-centre articles, streamed token by token."""
from __future__ import annotations

from .llm import client, tag

SYSTEM = ("You are a support agent at Northwind Cloud. Write a short, friendly reply (under 120 words) that "
          "answers the customer using only the help-centre articles provided. If the articles don't cover it, "
          "say you've passed it to the right team. Sign off as 'Northwind Support'.")


def draft_reply(ticket: dict, articles: list[tuple[str, str]], on_token=None) -> str:
    context = "\n\n".join(f"## {title}\n{body}" for title, body in articles)
    parts = []
    with tag(feature="reply-draft", team="support"):
        stream = client().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=250,
            stream=True,
            stream_options={"include_usage": True},
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": f"Help-centre articles:\n{context}\n\nTicket from "
                                                  f"{ticket['customer']}:\nSubject: {ticket['subject']}\n\n"
                                                  f"{ticket['body']}"}],
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
                if on_token:
                    on_token(chunk.choices[0].delta.content)
    return "".join(parts)
