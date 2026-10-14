from tl import BASE
from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(model="llama3.2", base_url=BASE)
llm.invoke("hi")  # @call
for _ in llm.stream("hi"):  # @call
    pass
OllamaEmbeddings(model="nomic-embed-text", base_url=BASE).embed_documents(["a"])  # @call
