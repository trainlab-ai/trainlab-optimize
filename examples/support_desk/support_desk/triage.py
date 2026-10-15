"""Route each ticket to one of five queues."""
from __future__ import annotations

from .data import QUEUES
from .llm import client, tag

SYSTEM = ("You route customer support tickets for Northwind Cloud, a file-sync company. "
          f"Categories: {', '.join(QUEUES)}. "
          "billing = charges, invoices, refunds, payments. bug = the product misbehaves. "
          "account = login, 2FA, ownership, SSO. shipping = hardware orders and deliveries. "
          "feature_request = asks for something the product doesn't do. "
          "Reply with the category name only.")


def classify(ticket: dict) -> str:
    with tag(feature="triage", team="support-ops"):
        r = client().chat.completions.create(
            model="gpt-4o",
            temperature=0,
            max_tokens=5,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": f"Subject: {ticket['subject']}\n\n{ticket['body']}"}],
        )
    label = (r.choices[0].message.content or "").strip().lower().strip(".")
    return label if label in QUEUES else "bug"
