"""Ticket routing classifier (demo) via LiteLLM. Model comes from an env var with a
premium default; classification needs only a one-word answer."""
import os

import litellm

ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-4.1")

LABELS = ["billing", "shipping", "technical", "account", "other"]


def classify_ticket(subject: str, body: str) -> str:
    prompt = (
        "Classify the support ticket into one of the following categories: "
        + ", ".join(LABELS)
        + ". Reply with the category label only.\n\n"
        + f"Subject: {subject}\nBody: {body}"
    )
    resp = litellm.completion(
        model=ROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
    )
    return resp.choices[0].message.content.strip().lower()
