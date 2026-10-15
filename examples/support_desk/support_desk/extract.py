"""Pull structured fields out of a ticket for the CRM.

Two versions of the prompt, to show what an experiment's examples are for:

  v1  the first draft. It never says what a "product" is or what counts as wanting a refund, so
      gpt-4o itself answers "SSO" or "Starter" as the product, and cheaper models guess
      differently. `trainlab experiment show <id> --examples` makes that visible.
  v2  (default) the same fields, defined: the product is one of Northwind's five products, a
      refund means asking for money back, urgency has rules, the summary names the customer.
"""
from __future__ import annotations

import json

from .data import PRODUCTS
from .llm import client, tag

PROMPTS = {
    "v1": ("Extract fields from the support ticket and return JSON with exactly these keys: "
           '"product" (string or null), "order_id" (string like NW-12345 or null), '
           '"urgency" ("low", "normal" or "high"), "wants_refund" (true or false), '
           '"summary" (one sentence, at most 20 words).'),
    "v2": ("Extract fields from a Northwind Cloud support ticket. Return JSON with exactly these keys:\n"
           f'- "product": which Northwind product the ticket is about, exactly one of {json.dumps(PRODUCTS)}, '
           'or null if none is named. Plans (Starter, Pro, Business, Enterprise) and features (SSO, 2FA, API, '
           'search) are not products. Hardware orders are "Northwind Desk Hub".\n'
           '- "order_id": the order number like NW-12345, or null.\n'
           '- "urgency": "high" if the customer is locked out, blocked from working, losing data or money, or '
           'has a deadline; "low" for feature requests and questions; otherwise "normal".\n'
           '- "wants_refund": true only if the customer asks for money back (a refund, credit or '
           'reimbursement); a question about a charge alone is false.\n'
           '- "summary": one sentence of at most 20 words that names the customer and the specific problem '
           'or request.'),
}
PROMPT_VERSION = "v2"          # run.py --prompt v1 switches back to the first draft


def extract_fields(ticket: dict) -> dict:
    with tag(feature="crm-extract", team="support-ops"):
        r = client().chat.completions.create(
            model="gpt-4o",
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": PROMPTS[PROMPT_VERSION]},
                      {"role": "user", "content": f"Subject: {ticket['subject']}\n\n{ticket['body']}"}],
        )
    try:
        return json.loads(r.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return {}
