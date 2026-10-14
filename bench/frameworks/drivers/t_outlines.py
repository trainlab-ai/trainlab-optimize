from tl import BASE
import openai
import outlines

model = outlines.from_openai(openai.OpenAI(base_url=f"{BASE}/v1"), "gpt-4o-mini")
model("hi")  # @call
