from tl import BASE
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-5", base_url=BASE, max_tokens=50)
llm.invoke("hi")  # @call
for _ in llm.stream("hi"):  # @call
    pass
