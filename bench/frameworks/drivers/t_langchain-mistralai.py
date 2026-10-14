from tl import BASE
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(model="mistral-large-latest", endpoint=f"{BASE}/v1", api_key="m")
llm.invoke("hi")  # @call
for _ in llm.stream("hi"):  # @call
    pass
