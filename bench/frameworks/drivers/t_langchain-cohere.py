from tl import BASE
from langchain_cohere import ChatCohere

llm = ChatCohere(model="command-r-plus", base_url=BASE, cohere_api_key="c")
llm.invoke("hi")  # @call
