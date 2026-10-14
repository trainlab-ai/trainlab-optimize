from tl import BASE
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

g = OpenAIChatGenerator(model="gpt-4o", api_base_url=f"{BASE}/v1", api_key=Secret.from_token("k"))
g.run([ChatMessage.from_user("hi")])  # @call
