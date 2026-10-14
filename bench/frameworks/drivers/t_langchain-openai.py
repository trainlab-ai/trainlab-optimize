import asyncio

from tl import BASE
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-4o-mini", base_url=f"{BASE}/v1")
llm.invoke("hi")  # @call
for _ in llm.stream("hi"):  # @call
    pass
llm.batch(["a", "b"])  # @call
OpenAIEmbeddings(model="text-embedding-3-small", base_url=f"{BASE}/v1", check_embedding_ctx_length=False).embed_documents(["a", "b"])  # @call
asyncio.run(llm.ainvoke("hi"))  # @call
