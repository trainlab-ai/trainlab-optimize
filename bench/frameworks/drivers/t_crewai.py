# reply: Thought: I now can give a great answer\nFinal Answer: ok
# timeout: 300
from tl import BASE
from crewai import LLM, Agent, Crew, Task

llm = LLM(model="openai/gpt-4o-mini", base_url=f"{BASE}/v1", api_key="k")
a = Agent(role="r", goal="g", backstory="b", llm=llm, verbose=False)
crew = Crew(agents=[a], tasks=[Task(description="say ok", expected_output="ok", agent=a)])
crew.kickoff()  # @call
