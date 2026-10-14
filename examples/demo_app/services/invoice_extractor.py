"""Invoice field extraction (demo). Deliberate waste: premium model for a simple
structured-extraction task, one request per document in a loop, no output cap,
unbounded retries."""
import json

from openai import OpenAI
from tenacity import retry

client = OpenAI()
EXTRACTION_MODEL = "gpt-4o"

EXTRACT_PROMPT = """Extract the following fields from the invoice text and return JSON:
vendor_name, invoice_number, invoice_date (ISO 8601), due_date, currency, subtotal, tax, total,
line_items (description, quantity, unit_price). Use null when a field is missing."""


@retry
def extract_invoice(text: str) -> dict:
    resp = client.chat.completions.create(
        model=EXTRACTION_MODEL,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def process_batch(documents: list[str]) -> list[dict]:
    results = []
    for doc in documents:
        resp = client.chat.completions.create(
            model=EXTRACTION_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": doc},
            ],
        )
        results.append(json.loads(resp.choices[0].message.content))
    return results
