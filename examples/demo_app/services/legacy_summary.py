"""Nightly call-note summaries (demo). Deliberate waste: legacy model; latency-tolerant
batch job that does not use the Batch API."""
from openai import OpenAI

client = OpenAI()


def summarize_notes(notes: list[str]) -> list[str]:
    return [
        client.chat.completions.create(
            model="gpt-4",
            max_tokens=300,
            messages=[
                {"role": "system", "content": "Summarize the call notes in three bullet points."},
                {"role": "user", "content": n},
            ],
        ).choices[0].message.content
        for n in notes
    ]
