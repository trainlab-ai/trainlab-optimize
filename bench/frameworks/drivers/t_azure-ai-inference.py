from tl import BASE
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential

c = ChatCompletionsClient(endpoint=BASE, credential=AzureKeyCredential("a"))
c.complete(model="Phi-4", messages=[UserMessage(content="hi")])  # @call
for _ in c.complete(model="Phi-4", messages=[UserMessage(content="hi")], stream=True):  # @call
    pass
