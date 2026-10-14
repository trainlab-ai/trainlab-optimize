import asyncio

from tl import BASE
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

client = OpenAIChatCompletionClient(model="gpt-4o", base_url=f"{BASE}/v1", api_key="k")
agent = AssistantAgent("a", model_client=client)
asyncio.run(agent.run(task="hi"))  # @call
