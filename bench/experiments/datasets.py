"""Synthetic but realistic samples for one call site, with known correct answers.

The system prompt is the call site's own (from the scan) when it has one; inputs are
support tickets (classification), invoices (extraction) or questions (generation). Every
input is unique, so the zoo can recognise it.
"""
from __future__ import annotations

import hashlib
import random

LABELS = ["billing", "bug", "account", "feature_request", "shipping"]
TICKETS = {
    "billing": ["I was charged twice for order {n}.", "My invoice {n} shows the wrong VAT amount.",
                "Please refund the duplicate payment on {n}."],
    "bug": ["The export button crashes the app (build {n}).", "Search returns nothing since update {n}.",
            "Page {n} shows a blank screen on Safari."],
    "account": ["I can't log in after resetting my password ({n}).", "Please change the email on account {n}.",
                "How do I delete my account {n}?"],
    "feature_request": ["Could you add dark mode? (ref {n})", "Please support CSV import for project {n}.",
                        "It would help to schedule reports weekly ({n})."],
    "shipping": ["Parcel {n} hasn't arrived after two weeks.", "Can I change the delivery address for {n}?",
                 "Tracking for {n} has not updated in days."],
}
VENDORS = ["Acme Ltd", "Globex GmbH", "Initech LLC", "Umbrella SA", "Stark Industries", "Wayne Enterprises",
           "Soylent Corp", "Hooli Inc", "Vandelay Imports", "Wonka GmbH"]
CITIES = [("France", "Paris"), ("Japan", "Tokyo"), ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Canada", "Ottawa"),
          ("India", "New Delhi"), ("Egypt", "Cairo"), ("Norway", "Oslo"), ("Chile", "Santiago"), ("Vietnam", "Hanoi")]
SYSTEM = {
    "classification": "Classify the customer message into one of: " + ", ".join(LABELS) + ". Reply with the label only.",
    "extraction": "Extract vendor, invoice_number, date, total and currency from the invoice. Reply with JSON only.",
    "generation": "Answer the question in one or two sentences.",
}


def key_of(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()[:16]


def task_for(hints: list[str], seed: str) -> str:
    h = set(hints or [])
    if h & {"extraction", "structured_output", "forced_tool", "json"}:
        return "extraction"
    if h & {"classification", "routing", "sentiment", "moderation"}:
        return "classification"
    return ["classification", "extraction", "generation"][int(hashlib.sha1(seed.encode()).hexdigest()[:4], 16) % 3]


def build(exp: str, task: str, n: int, system: str | None, seed: str) -> list[dict]:
    """[{user, gold, labels?}] with unique inputs."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        tag = f"{seed[:6]}-{i:04d}"
        if task == "classification":
            label = LABELS[rng.randrange(len(LABELS))]
            user = rng.choice(TICKETS[label]).format(n=tag)
            rows.append({"user": user, "gold": label, "labels": LABELS})
        elif task == "extraction":
            gold = {"vendor": rng.choice(VENDORS), "invoice_number": f"INV-{tag}",
                    "date": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                    "total": round(rng.uniform(10, 9000), 2), "currency": rng.choice(["USD", "EUR", "GBP", "INR"])}
            user = (f"INVOICE {gold['invoice_number']}\nFrom: {gold['vendor']}\nDate: {gold['date']}\n"
                    f"Items: consulting services\nTotal due: {gold['total']} {gold['currency']}")
            rows.append({"user": user, "gold": gold})
        else:
            country, city = CITIES[i % len(CITIES)]
            user = f"Question {tag}: what is the capital of {country}, and what is it known for?"
            rows.append({"user": user, "gold": f"The capital of {country} is {city}."})
    return rows
