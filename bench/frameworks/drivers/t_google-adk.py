import asyncio

from tl import BASE
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

runner = InMemoryRunner(agent=Agent(name="a", model="gemini-2.5-flash", instruction="be brief"))


async def main():
    s = await runner.session_service.create_session(app_name=runner.app_name, user_id="u")
    async for _ in runner.run_async(user_id="u", session_id=s.id, new_message=types.Content(role="user", parts=[types.Part(text="hi")])):  # @call
        pass
asyncio.run(main())
