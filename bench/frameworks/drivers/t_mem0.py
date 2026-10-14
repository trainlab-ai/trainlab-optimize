# reply: {"facts": []}
# timeout: 300
import os

from tl import BASE
from mem0 import Memory

cfg = {"llm": {"provider": "openai", "config": {"model": "gpt-4o-mini", "openai_base_url": f"{BASE}/v1", "api_key": "k"}},
       "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small", "openai_base_url": f"{BASE}/v1", "api_key": "k", "embedding_dims": 8}},
       "vector_store": {"provider": "qdrant", "config": {"path": os.path.join(os.environ["HOME"], "qdrant"), "embedding_model_dims": 8, "on_disk": False}}}
m = Memory.from_config(cfg)
m.add("I like tea", user_id="u")  # @call
