import json

from tl import BASE
import boto3

br = boto3.client("bedrock-runtime", endpoint_url=BASE)
br.converse(modelId="anthropic.claude-3-5-haiku-20241022-v1:0", messages=[{"role": "user", "content": [{"text": "hi"}]}])  # @call
r = br.converse_stream(modelId="amazon.nova-lite-v1:0", messages=[{"role": "user", "content": [{"text": "hi"}]}])  # @call
for _ in r["stream"]:
    pass
br.invoke_model(modelId="anthropic.claude-3-5-sonnet-20241022-v2:0", body=json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]}))  # @call
r = br.invoke_model_with_response_stream(modelId="anthropic.claude-3-5-sonnet-20241022-v2:0", body=json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]}))  # @call
for _ in r["body"]:
    pass
