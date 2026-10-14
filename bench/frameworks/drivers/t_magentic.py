from tl import BASE
from magentic import OpenaiChatModel, prompt


@prompt("Say hi to {name}", model=OpenaiChatModel("gpt-4o-mini", base_url=f"{BASE}/v1"))
def greet(name: str) -> str: ...


greet("bob")  # @call
