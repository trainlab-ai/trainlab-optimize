from tl import BASE
from mirascope import llm


@llm.call("openai/gpt-4o-mini")
def recommend(genre: str) -> str:
    return f"Recommend a {genre} book"


recommend("fantasy")  # @call
