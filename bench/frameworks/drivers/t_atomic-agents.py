# tool_call: {"chat_message": "ok"}
from tl import BASE
import instructor
import openai
from atomic_agents import AgentConfig, AtomicAgent, BasicChatInputSchema, BasicChatOutputSchema

client = instructor.from_openai(openai.OpenAI(base_url=f"{BASE}/v1"))
agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config=AgentConfig(client=client, model="gpt-4o-mini"))
agent.run(BasicChatInputSchema(chat_message="hi"))  # @call
