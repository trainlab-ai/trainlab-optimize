from tl import BASE
from huggingface_hub import InferenceClient

c = InferenceClient(base_url=f"{BASE}/v1", api_key="hf_fake")
c.chat_completion(messages=[{"role": "user", "content": "hi"}], model="meta-llama/Llama-3.1-8B-Instruct")  # @call
for _ in c.chat_completion(messages=[{"role": "user", "content": "hi"}], stream=True, max_tokens=20):  # @call
    pass
c.chat.completions.create(model="tgi", messages=[{"role": "user", "content": "hi"}])  # @call
