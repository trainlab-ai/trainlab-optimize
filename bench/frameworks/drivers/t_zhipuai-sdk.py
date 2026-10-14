from tl import BASE
from zhipuai import ZhipuAI

c = ZhipuAI(api_key="z.z", base_url=f"{BASE}/api/paas/v4")
c.chat.completions.create(model="glm-4-plus", messages=[{"role": "user", "content": "hi"}])  # @call
for _ in c.chat.completions.create(model="glm-4-flash", stream=True, messages=[{"role": "user", "content": "hi"}]):  # @call
    pass
