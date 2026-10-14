"""Feature code: every SDK shape a real app uses."""
import trainlab_optimize as trainlab
from google.genai import types

from .llm import aan, an, aoa, gm, h, oa


@trainlab.tag(feature="support")
def support_answer(q):
    r = oa.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": q}],
                                   extra_headers=h("support"))
    return r.choices[0].message.content


def classify(ticket):
    with trainlab.tag(feature="classify"):
        return oa.chat.completions.create(model="gpt-4o-mini", max_tokens=3,
                                          messages=[{"role": "user", "content": ticket}],
                                          extra_headers=h("classify")).choices[0].message.content


def summarise_stream(doc, include_usage=True):
    with trainlab.tag(feature="summarise"):
        kw = {"stream_options": {"include_usage": True}} if include_usage else {}
        s = oa.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": doc}],
                                       stream=True, extra_headers=h("summarise"), **kw)
        return "".join(c.choices[0].delta.content or "" for c in s if c.choices)


def first_token_only(doc):
    """App reads one chunk and abandons the stream."""
    with trainlab.tag(feature="peek"):
        s = oa.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": doc}],
                                       stream=True, stream_options={"include_usage": True}, extra_headers=h("peek"))
        for c in s:
            return c.choices[0].delta.content


def agent_step(goal):
    with trainlab.tag(feature="agent"):
        return oa.responses.create(model="gpt-5-mini", input=goal, extra_headers=h("agent")).output_text


def embed(chunks):
    with trainlab.tag(feature="rag-index"):
        return oa.embeddings.create(model="text-embedding-3-small", input=chunks, extra_headers=h("rag-index"))


def extract_invoice(text):
    with trainlab.tag(feature="invoice"):
        m = an.messages.create(model="claude-sonnet-4-5", max_tokens=500,
                               messages=[{"role": "user", "content": text}], extra_headers=h("invoice"))
        return m.content[0].text


def draft_email(ctx):
    with trainlab.tag(feature="email"):
        with an.messages.stream(model="claude-haiku-4-5", max_tokens=300,
                                messages=[{"role": "user", "content": ctx}], extra_headers=h("email")) as s:
            return s.get_final_text()


def translate(text):
    with trainlab.tag(feature="translate"):
        cfg = types.GenerateContentConfig(http_options=types.HttpOptions(headers=h("translate")))
        return gm.models.generate_content(model="gemini-2.5-flash", contents=text, config=cfg).text


def translate_stream(text):
    with trainlab.tag(feature="translate"):
        cfg = types.GenerateContentConfig(http_options=types.HttpOptions(headers=h("translate")))
        return "".join(c.text or "" for c in gm.models.generate_content_stream(model="gemini-2.5-flash",
                                                                              contents=text, config=cfg))


async def async_chat(i):
    with trainlab.tag(feature=f"tenant-{i % 7}"):
        if i % 3 == 0:
            m = await aan.messages.create(model="claude-haiku-4-5", max_tokens=50,
                                          messages=[{"role": "user", "content": f"q{i}"}],
                                          extra_headers=h(f"tenant-{i % 7}"))
            return m.content[0].text
        if i % 3 == 1:
            s = await aoa.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": f"q{i}"}],
                                                  stream=True, stream_options={"include_usage": True},
                                                  extra_headers=h(f"tenant-{i % 7}"))
            return "".join([c.choices[0].delta.content async for c in s if c.choices])
        cfg = types.GenerateContentConfig(http_options=types.HttpOptions(headers=h(f"tenant-{i % 7}")))
        r = await gm.aio.models.generate_content(model="gemini-2.5-flash", contents=f"q{i}", config=cfg)
        return r.text
