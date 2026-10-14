from tl import BASE
from deepeval.models import GPTModel

GPTModel(model="gpt-4o-mini", base_url=f"{BASE}/v1", api_key="k").generate("hi")  # @call
