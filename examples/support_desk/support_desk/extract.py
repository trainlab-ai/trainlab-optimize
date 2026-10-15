"""Pull structured fields out of a ticket for the CRM."""
from __future__ import annotations

import json

from .llm import client, tag

SYSTEM = ("Extract fields from the support ticket and return JSON with exactly these keys: "
          '"product" (string or null), "order_id" (string like NW-12345 or null), '
          '"urgency" ("low", "normal" or "high"), "wants_refund" (true or false), '
          '"summary" (one sentence, at most 20 words).')


def extract_fields(ticket: dict) -> dict:
    with tag(feature="crm-extract", team="support-ops"):
        r = client().chat.completions.create(
            model="gpt-4o",
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": f"Subject: {ticket['subject']}\n\n{ticket['body']}"}],
        )
    try:
        return json.loads(r.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return {}
