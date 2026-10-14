from tl import BASE
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(ChatOpenAI(model="gpt-4o", base_url=f"{BASE}/v1"), tools=[])
agent.invoke({"messages": [("user", "hi")]})  # @call
