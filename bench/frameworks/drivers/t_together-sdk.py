from tl import BASE
import together

c = together.Together(base_url=f"{BASE}/v1")
c.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", messages=[{"role": "user", "content": "hi"}])  # @call
for _ in c.chat.completions.create(model="Qwen/Qwen2.5-72B-Instruct-Turbo", stream=True, messages=[{"role": "user", "content": "hi"}]):  # @call
    pass
c.embeddings.create(model="BAAI/bge-base-en-v1.5", input=["a"])  # @call
