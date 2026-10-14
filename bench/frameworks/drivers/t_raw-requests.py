from tl import BASE
import requests

requests.post(f"{BASE}/v1/chat/completions", headers={"Authorization": "Bearer sk"}, json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]})  # @call
requests.post(f"{BASE}/v1/messages", headers={"x-api-key": "k", "anthropic-version": "2023-06-01"}, json={"model": "claude-haiku-4-5", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}]})  # @call
s = requests.Session()
r = s.post(f"{BASE}/v1/chat/completions", stream=True, json={"model": "gpt-4o-mini", "stream": True, "stream_options": {"include_usage": True}, "messages": [{"role": "user", "content": "hi"}]})  # @call
for _ in r.iter_lines():
    pass
requests.get(f"{BASE}/v1/models")
