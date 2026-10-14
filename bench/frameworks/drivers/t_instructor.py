# reply: {"name": "ok"}
from tl import BASE
import anthropic
import instructor
import openai
from pydantic import BaseModel


class User(BaseModel):
    name: str


oc = instructor.from_openai(openai.OpenAI(base_url=f"{BASE}/v1"), mode=instructor.Mode.JSON)
oc.chat.completions.create(model="gpt-4o-mini", response_model=User, messages=[{"role": "user", "content": "hi"}])  # @call
ac = instructor.from_anthropic(anthropic.Anthropic(base_url=BASE), mode=instructor.Mode.ANTHROPIC_JSON)
ac.messages.create(model="claude-haiku-4-5", max_tokens=50, response_model=User, messages=[{"role": "user", "content": "hi"}])  # @call
