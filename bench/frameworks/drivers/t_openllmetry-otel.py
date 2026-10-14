"""App that already uses OpenLLMetry (wraps the same OpenAI method trainlab hooks)."""
from tl import BASE
import openai
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.trace import TracerProvider

OpenAIInstrumentor().instrument(tracer_provider=TracerProvider())
openai.OpenAI(base_url=f"{BASE}/v1").chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])  # @call
