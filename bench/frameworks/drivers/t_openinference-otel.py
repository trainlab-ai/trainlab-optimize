"""App that already uses OpenInference tracing; trainlab instrument() alongside it."""
from tl import BASE
import openai
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.trace import TracerProvider

OpenAIInstrumentor().instrument(tracer_provider=TracerProvider())
openai.OpenAI(base_url=f"{BASE}/v1").chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])  # @call
