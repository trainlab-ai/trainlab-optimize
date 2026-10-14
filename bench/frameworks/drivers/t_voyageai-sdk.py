from tl import BASE
import voyageai

c = voyageai.Client(api_key="v")
voyageai.api_base = f"{BASE}/v1"
try:
    c._client_config = getattr(c, "_client_config", None)
except Exception:
    pass
import os
os.environ["VOYAGE_API_BASE"] = f"{BASE}/v1"
c = voyageai.Client(api_key="v", base_url=f"{BASE}/v1") if "base_url" in voyageai.Client.__init__.__code__.co_varnames else voyageai.Client(api_key="v")
c.embed(["a", "b"], model="voyage-3")  # @call
