from tl import BASE
from google.genai import types
from llama_index.llms.google_genai import GoogleGenAI

llm = GoogleGenAI(model="gemini-2.5-flash", api_key="g", http_options=types.HttpOptions(base_url=BASE))
llm.complete("hi")  # @call
