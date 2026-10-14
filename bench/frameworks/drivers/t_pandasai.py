# reply: ```python\nresult = {"type": "string", "value": "ok"}\n```
from tl import BASE
import pandas as pd
import pandasai as pai
from pandasai_openai import OpenAI

pai.config.set({"llm": OpenAI(api_token="k", api_base=f"{BASE}/v1")})
df = pai.DataFrame(pd.DataFrame({"a": [1, 2]}))
df.chat("say ok")  # @call
