import json

from . import clients


def extract(invoice_text: str) -> dict:
    msg = clients.anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=800,
        system="Extract vendor, date, total and line items as JSON.",
        messages=[{"role": "user", "content": invoice_text}],
    )
    return json.loads(msg.content[0].text)
