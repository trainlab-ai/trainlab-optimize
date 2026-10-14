from tl import BASE
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType

model = ModelFactory.create(model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL, model_type="gpt-4o-mini", url=f"{BASE}/v1", api_key="k")
ChatAgent("be brief", model=model).step("hi")  # @call
