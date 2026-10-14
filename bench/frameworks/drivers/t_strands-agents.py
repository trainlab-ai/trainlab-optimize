from tl import BASE
from strands import Agent
from strands.models import BedrockModel

agent = Agent(model=BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0", region_name="us-east-1"), callback_handler=None)
agent("hi")  # @call
