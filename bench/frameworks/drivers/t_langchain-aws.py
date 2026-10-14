from tl import BASE
from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(model="anthropic.claude-3-5-haiku-20241022-v1:0", region_name="us-east-1", endpoint_url=BASE)
llm.invoke("hi")  # @call
for _ in llm.stream("hi"):  # @call
    pass
