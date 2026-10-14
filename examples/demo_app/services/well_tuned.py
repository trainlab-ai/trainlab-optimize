"""Negative control (demo): an already-efficient call. Should produce no findings."""
from openai import OpenAI
from tenacity import retry, stop_after_attempt

client = OpenAI()


@retry(stop=stop_after_attempt(3))
def short_reply(message: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-5-mini",
        max_completion_tokens=200,
        messages=[{"role": "user", "content": message}],
    )
    return resp.choices[0].message.content
