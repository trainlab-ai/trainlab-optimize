import asyncio

from tl import BASE
import ollama

c = ollama.Client(host=BASE)
c.chat(model="llama3.2", messages=[{"role": "user", "content": "hi"}])  # @call
for _ in c.chat(model="qwen2.5", messages=[{"role": "user", "content": "hi"}], stream=True):  # @call
    pass
c.generate(model="llama3.2", prompt="hi")  # @call
c.embed(model="nomic-embed-text", input=["a"])  # @call


async def main():
    await ollama.AsyncClient(host=BASE).chat(model="llama3.2", messages=[{"role": "user", "content": "hi"}])  # @call
asyncio.run(main())
