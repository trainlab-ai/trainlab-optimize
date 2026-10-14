import litellm
import trainlab_optimize as trainlab

from .llm import BASE, h


def via_litellm(q, i):
    with trainlab.tag(feature="router"):
        if i % 2:
            return litellm.completion(model="openai/gpt-4o-mini", api_base=f"{BASE}/v1", api_key="sk-x",
                                      messages=[{"role": "user", "content": q}], extra_headers=h("router"))
        return litellm.completion(model="anthropic/claude-haiku-4-5", api_base=BASE, api_key="sk-ant-x",
                                  max_tokens=50, messages=[{"role": "user", "content": q}], extra_headers=h("router"))
