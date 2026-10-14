from tl import BASE
from langchain_core.prompt_values import StringPromptValue
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

w = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", base_url=f"{BASE}/v1"))
w.generate_text(StringPromptValue(text="hi"))  # @call
