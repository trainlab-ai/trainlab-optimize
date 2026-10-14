from tl import BASE
import langroid as lr
import langroid.language_models as lm

cfg = lm.OpenAIGPTConfig(chat_model="gpt-4o-mini", api_base=f"{BASE}/v1", api_key="k", stream=False)
agent = lr.ChatAgent(lr.ChatAgentConfig(llm=cfg))
agent.llm_response("hi")  # @call
