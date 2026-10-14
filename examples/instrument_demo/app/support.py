from . import clients

SYSTEM = "You are a helpful support agent for Acme. Answer briefly and cite the help-centre article."


def answer(question: str) -> str:
    r = clients.openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        temperature=0.2,
    )
    return r.choices[0].message.content


def classify(ticket: str) -> str:
    r = clients.openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "Label the ticket: billing, bug, account or other. One word."},
                  {"role": "user", "content": ticket}],
        max_tokens=3,
    )
    return r.choices[0].message.content
