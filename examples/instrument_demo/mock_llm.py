"""Offline stand-ins for the OpenAI and Anthropic APIs: realistic usage numbers, no network, no keys."""
import json
import random

import httpx

rng = random.Random(7)


def _tokens(body) -> int:
    return max(20, len(json.dumps(body.get("messages") or body.get("input") or "")) // 4)


def openai_handler(request):
    body = json.loads(request.content)
    if request.url.path.endswith("/embeddings"):
        n = len(body["input"]) if isinstance(body["input"], list) else 1
        return httpx.Response(200, json={"object": "list", "model": body["model"],
                                         "data": [{"object": "embedding", "index": i, "embedding": [0.0] * 8}
                                                  for i in range(n)],
                                         "usage": {"prompt_tokens": _tokens(body) * 3, "total_tokens": 0}})
    pin, pout = _tokens(body) + rng.randint(300, 900), min(body.get("max_tokens") or 10**6, rng.randint(80, 400))
    cached = (pin // 2) if rng.random() < 0.3 else 0
    if body.get("stream"):
        chunks = [{"id": "c", "object": "chat.completion.chunk", "created": 1, "model": body["model"],
                   "choices": [{"index": 0, "delta": {"content": "word "}, "finish_reason": None}]}] * 5
        chunks.append({"id": "c", "object": "chat.completion.chunk", "created": 1, "model": body["model"],
                       "choices": [], "usage": {"prompt_tokens": pin, "completion_tokens": pout,
                                                "total_tokens": pin + pout}})
        data = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, content=data.encode(), headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json={
        "id": "x", "object": "chat.completion", "created": 1, "model": body["model"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "billing"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": pin, "completion_tokens": pout, "total_tokens": pin + pout,
                  "prompt_tokens_details": {"cached_tokens": cached}}})


def anthropic_handler(request):
    body = json.loads(request.content)
    pin = _tokens(body) + rng.randint(1500, 3000)
    return {"id": "m", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "text", "text": '{"vendor": "Acme", "total": 12.5}'}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": pin, "output_tokens": rng.randint(150, 600),
                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}


def clients():
    import anthropic
    import openai
    oa = openai.OpenAI(api_key="demo", max_retries=0,
                       http_client=httpx.Client(transport=httpx.MockTransport(openai_handler)))
    try:                                 # newer anthropic SDKs use httpx2
        import httpx2 as hx
        anthropic.Anthropic(api_key="demo", http_client=hx.Client())
    except (ImportError, TypeError):
        hx = httpx
    an = anthropic.Anthropic(api_key="demo", max_retries=0, http_client=hx.Client(
        transport=hx.MockTransport(lambda req: hx.Response(200, json=anthropic_handler(req)))))
    return oa, an
