from tl import BASE
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o-mini", api_base=f"{BASE}/v1")
llm.complete("hi")  # @call
for _ in llm.stream_complete("hi"):  # @call
    pass
from llama_index.core.llms import ChatMessage
llm.chat([ChatMessage(role="user", content="hi")])  # @call
OpenAIEmbedding(model="text-embedding-3-small", api_base=f"{BASE}/v1").get_text_embedding("a")  # @call
