from tl import BASE
from openai import OpenAI
from vanna.openai import OpenAI_Chat

Impl = type("Impl", (OpenAI_Chat,), {m: (lambda *a, **k: None) for m in getattr(OpenAI_Chat, "__abstractmethods__", ())})
vn = Impl(client=OpenAI(base_url=f"{BASE}/v1"), config={"model": "gpt-4o-mini"})
vn.submit_prompt([vn.user_message("hi")])  # @call
