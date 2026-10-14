from tl import BASE
from llama_index.llms.mistralai import MistralAI

llm = MistralAI(model="mistral-large-latest", api_key="m", endpoint=BASE)
llm.complete("hi")  # @call
