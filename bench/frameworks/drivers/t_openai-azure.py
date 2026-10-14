from tl import BASE
import openai

c = openai.AzureOpenAI(azure_endpoint=BASE, api_version="2024-10-21", api_key="a")
c.chat.completions.create(model="gpt-4o-prod", messages=[{"role": "user", "content": "hi"}])  # @call
c.embeddings.create(model="embed-prod", input=["a"])  # @call
