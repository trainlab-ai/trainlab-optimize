from tl import BASE
import dashscope

dashscope.base_http_api_url = f"{BASE}/api/v1"
dashscope.Generation.call(model="qwen-max", messages=[{"role": "user", "content": "hi"}], result_format="message")  # @call
dashscope.Generation.call(model="qwen-turbo", prompt="hi")  # @call
for _ in dashscope.Generation.call(model="qwen-plus", messages=[{"role": "user", "content": "hi"}], result_format="message", stream=True):  # @call
    pass
dashscope.TextEmbedding.call(model="text-embedding-v3", input="a")  # @call
