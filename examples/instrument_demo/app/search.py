from . import clients


def embed(chunks: list[str]) -> list[list[float]]:
    r = clients.openai_client.embeddings.create(model="text-embedding-3-small", input=chunks)
    return [d.embedding for d in r.data]


def summarise(docs: list[str]) -> str:
    stream = clients.openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Summarise:\n" + "\n".join(docs)}],
        stream=True,
        stream_options={"include_usage": True},
    )
    return "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
