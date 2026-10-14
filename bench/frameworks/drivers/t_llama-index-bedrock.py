from tl import BASE
from llama_index.llms.bedrock_converse import BedrockConverse

llm = BedrockConverse(model="anthropic.claude-3-5-haiku-20241022-v1:0", region_name="us-east-1")
llm.complete("hi")  # @call
