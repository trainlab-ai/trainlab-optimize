"""Shipping questions: a small tool-calling agent that looks the order up before answering."""
from __future__ import annotations

import json

from .data import ORDERS
from .llm import client, tag

AGENT_MODEL = "gpt-4o-mini"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "lookup_order",
        "description": "Get the shipping status of a Northwind hardware order.",
        "parameters": {"type": "object",
                       "properties": {"order_id": {"type": "string", "description": "Order id like NW-12345"}},
                       "required": ["order_id"]},
    },
}]

SYSTEM = ("You answer shipping questions for Northwind Cloud hardware orders. Always call lookup_order first, "
          "then reply in under 80 words with the status and the next step. Sign off as 'Northwind Support'.")


def lookup_order(order_id: str) -> dict:
    return ORDERS.get(order_id.strip().upper(), {"order_id": order_id, "status": "not found"})


def answer_shipping(ticket: dict, max_turns: int = 3) -> str:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Subject: {ticket['subject']}\n\n{ticket['body']}"}]
    with tag(feature="shipping-agent", team="support"):
        for _ in range(max_turns):
            r = client().chat.completions.create(model=AGENT_MODEL, temperature=0, max_tokens=300,
                                                 messages=messages, tools=TOOLS)
            msg = r.choices[0].message
            if not msg.tool_calls:
                return msg.content or ""
            messages.append({"role": "assistant", "content": msg.content,
                             "tool_calls": [{"id": tc.id, "type": "function",
                                             "function": {"name": tc.function.name,
                                                          "arguments": tc.function.arguments}}
                                            for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = lookup_order(args.get("order_id", "")) if tc.function.name == "lookup_order" else {}
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return "We're checking on your order and will update you shortly.\n\nNorthwind Support"
