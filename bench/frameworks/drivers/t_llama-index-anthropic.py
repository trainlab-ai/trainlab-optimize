from tl import BASE
from llama_index.llms.anthropic import Anthropic

llm = Anthropic(model="claude-sonnet-4-5", base_url=BASE, max_tokens=50)
llm.complete("hi")  # @call
for _ in llm.stream_complete("hi"):  # @call
    pass
