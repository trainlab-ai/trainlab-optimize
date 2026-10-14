from tl import BASE
from griptape.drivers.prompt.openai import OpenAiChatPromptDriver
from griptape.structures import Agent

Agent(prompt_driver=OpenAiChatPromptDriver(model="gpt-4o", base_url=f"{BASE}/v1", api_key="k"), stream=False).run("hi")  # @call
