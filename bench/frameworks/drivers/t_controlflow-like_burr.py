from tl import BASE
import openai
from burr.core import ApplicationBuilder, State, action

client = openai.OpenAI(base_url=f"{BASE}/v1")


@action(reads=[], writes=["answer"])
def ask(state: State) -> State:
    r = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])  # @call
    return state.update(answer=r.choices[0].message.content)


app = ApplicationBuilder().with_actions(ask).with_transitions().with_entrypoint("ask").with_state().build()
app.step()
