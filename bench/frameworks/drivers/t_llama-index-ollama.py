from tl import BASE
from llama_index.llms.ollama import Ollama

llm = Ollama(model="llama3.2", base_url=BASE, request_timeout=30)
llm.complete("hi")  # @call
