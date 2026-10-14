from tl import BASE
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.3-70b-versatile", base_url=BASE)
llm.invoke("hi")  # @call
for _ in llm.stream("hi"):  # @call
    pass
