from tl import BASE
from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.openai import OpenAIChat

Agent(model=OpenAIChat(id="gpt-4o", base_url=f"{BASE}/v1")).run("hi")  # @call
Agent(model=Claude(id="claude-sonnet-4-5")).run("hi")  # @call
