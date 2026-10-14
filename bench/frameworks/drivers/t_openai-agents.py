from tl import BASE
from agents import Agent, Runner, set_default_openai_client, set_tracing_disabled
from openai import AsyncOpenAI

set_tracing_disabled(True)
set_default_openai_client(AsyncOpenAI(base_url=f"{BASE}/v1", api_key="k"))
Runner.run_sync(Agent(name="a", instructions="be brief", model="gpt-4.1-mini"), "hi")  # @call
