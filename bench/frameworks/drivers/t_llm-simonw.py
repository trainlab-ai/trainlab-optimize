from tl import BASE
import llm

llm.get_model("gpt-4o-mini").prompt("hi", key="sk").text()  # @call
