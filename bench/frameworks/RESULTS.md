# instrument() across 72 SDKs, frameworks and HTTP clients

Each project's real installed code runs a short driver (`drivers/t_<name>.py`, written the way
its docs show) against a local fake server that speaks the provider's wire format and logs every
token it serves. Every driver runs twice, with trainlab off and on.

| | |
|---|---|
| Projects | 72 |
| Pass (count, tokens and line all exact) | 72 |
| Breaks the app or changes its calls | 0 |
| LLM calls served / recorded | 145 / 145 |
| Recorded calls pointing at the app's own line | 145 of 145 |
| Other statuses | none |

## Provider SDKs

| Project | API | Calls (server / recorded) | Input tokens | Output tokens | App line | Recorded via | Result |
|---|---|---|---|---|---|---|---|
| openai-sdk | openai | 7 / 7 | exact | exact | 7/7 | openai chat, openai.completions.create, openai.embeddings.create, openai.responses.create | ✅ |
| anthropic-sdk | anthropic | 4 / 4 | exact | exact | 4/4 | anthropic messages, anthropic.messages.stream | ✅ |
| google-genai-sdk | gemini | 4 / 4 | exact | exact | 4/4 | google.genai.models.generate_content, google.genai.models.generate_content_stream | ✅ |
| boto3-bedrock | bedrock | 4 / 4 | exact | exact | 4/4 | botocore.bedrock-runtime | ✅ |
| litellm | openai+anthropic | 5 / 5 | exact | exact | 5/5 | litellm.acompletion, litellm.completion, litellm.embedding | ✅ |
| mistralai-sdk | mistral | 3 / 3 | exact | exact | 3/3 | http:httpx2 | ✅ |
| cohere-sdk | cohere | 4 / 4 | exact | exact | 4/4 | http:httpx | ✅ |
| groq-sdk | groq | 3 / 3 | exact | exact | 3/3 | http:httpx | ✅ |
| together-sdk | together | 3 / 3 | exact | exact | 3/3 | http:httpx | ✅ |
| ollama-sdk | ollama | 5 / 5 | exact | exact | 5/5 | http:httpx | ✅ |
| huggingface-hub | hf | 3 / 3 | exact | exact | 3/3 | http:httpx2 | ✅ |
| dashscope-sdk | dashscope | 4 / 4 | exact | exact | 4/4 | http:requests | ✅ |
| zhipuai-sdk | zhipu | 2 / 2 | exact | exact | 2/2 | http:httpx | ✅ |
| azure-ai-inference | azure-inference | 2 / 2 | exact | exact | 2/2 | http:requests | ✅ |
| google-generativeai | gemini-legacy | 2 / 2 | exact | exact | 2/2 | http:requests | ✅ |
| openai-azure | azure | 2 / 2 | exact | exact | 2/2 | openai chat, openai.embeddings.create | ✅ |
| fireworks-sdk | fireworks | 2 / 2 | exact | exact | 2/2 | http:httpx, openai chat | ✅ |
| voyageai-sdk | voyage | 1 / 1 | exact | exact | 1/1 | http:requests | ✅ |
| helicone-style-proxy | openrouter | 2 / 2 | exact | exact | 2/2 | openai chat | ✅ |
| google-genai-vertex | vertex | 1 / 1 | exact | exact | 1/1 | google.genai.models.generate_content | ✅ |

## Raw HTTP

| Project | API | Calls (server / recorded) | Input tokens | Output tokens | App line | Recorded via | Result |
|---|---|---|---|---|---|---|---|
| raw-requests | openai+anthropic | 3 / 3 | exact | exact | 3/3 | http:requests | ✅ |
| raw-httpx | openai | 3 / 3 | exact | exact | 3/3 | http:httpx | ✅ |
| raw-aiohttp | openai | 1 / 1 | exact | exact | 1/1 | http:aiohttp | ✅ |

## Frameworks and libraries

| Project | API | Calls (server / recorded) | Input tokens | Output tokens | App line | Recorded via | Result |
|---|---|---|---|---|---|---|---|
| langchain-openai | openai | 6 / 6 | exact | exact | 6/6 | openai chat, openai.embeddings.create | ✅ |
| langchain-anthropic | anthropic | 2 / 2 | exact | exact | 2/2 | anthropic messages | ✅ |
| langchain-google-genai | gemini | 2 / 2 | exact | exact | 2/2 | google.genai.models.generate_content, google.genai.models.generate_content_stream | ✅ |
| langchain-aws | bedrock | 2 / 2 | exact | exact | 2/2 | botocore.bedrock-runtime | ✅ |
| langchain-mistralai | mistral | 2 / 2 | exact | exact | 2/2 | http:httpx | ✅ |
| langchain-cohere | cohere | 1 / 1 | exact | exact | 1/1 | http:httpx | ✅ |
| langchain-groq | groq | 2 / 2 | exact | exact | 2/2 | http:httpx | ✅ |
| langchain-ollama | ollama | 3 / 3 | exact | exact | 3/3 | http:httpx | ✅ |
| langchain-community | openai | 1 / 1 | exact | exact | 1/1 | http:requests | ✅ |
| langgraph | openai | 1 / 1 | exact | exact | 1/1 | openai chat | ✅ |
| llama-index-openai | openai | 4 / 4 | exact | exact | 4/4 | openai chat, openai.embeddings.create | ✅ |
| llama-index-anthropic | anthropic | 2 / 2 | exact | exact | 2/2 | anthropic messages, anthropic.beta.messages.create | ✅ |
| llama-index-ollama | ollama | 1 / 1 | exact | exact | 1/1 | http:httpx | ✅ |
| llama-index-gemini | gemini | 1 / 1 | exact | exact | 1/1 | google.genai.models.generate_content | ✅ |
| llama-index-bedrock | bedrock | 1 / 1 | exact | exact | 1/1 | botocore.bedrock-runtime | ✅ |
| llama-index-mistral | mistral | 1 / 1 | exact | exact | 1/1 | http:httpx2 | ✅ |
| dspy | openai | 2 / 2 | exact | exact | 2/2 | litellm.completion | ✅ |
| instructor | openai+anthropic | 2 / 2 | exact | exact | 2/2 | anthropic messages, openai chat | ✅ |
| pydantic-ai | openai+anthropic+gemini | 2 / 2 | exact | exact | 2/2 | anthropic.beta.messages.create, openai chat | ✅ |
| openai-agents | openai | 1 / 1 | exact | exact | 1/1 | openai.responses.create | ✅ |
| smolagents | openai | 2 / 2 | exact | exact | 2/2 | litellm.completion, openai chat | ✅ |
| haystack | openai | 1 / 1 | exact | exact | 1/1 | openai chat | ✅ |
| haystack-anthropic | anthropic | 1 / 1 | exact | exact | 1/1 | anthropic messages | ✅ |
| crewai | openai | 1 / 1 | exact | exact | 1/1 | openai chat | ✅ |
| autogen | openai | 1 / 1 | exact | exact | 1/1 | openai chat | ✅ |
| semantic-kernel | openai | 1 / 1 | exact | exact | 1/1 | openai chat | ✅ |
| agno | openai+anthropic | 2 / 2 | exact | exact | 2/2 | anthropic messages, openai chat | ✅ |
| strands-agents | bedrock | 1 / 1 | exact | exact | 1/1 | botocore.bedrock-runtime | ✅ |
| google-adk | gemini | 1 / 1 | exact | exact | 1/1 | google.genai.models.generate_content | ✅ |
| mirascope | openai | 1 / 1 | exact | exact | 1/1 | openai.responses.create | ✅ |
| magentic | openai | 1 / 1 | exact | exact | 1/1 | openai chat | ✅ |
| llm-simonw | openai | 1 / 1 | exact | exact | 1/1 | openai chat | ✅ |
| guidance | openai | 1 / 1 | exact | exact | 1/1 | openai chat | ✅ |
| outlines | openai | 1 / 1 | exact | exact | 1/1 | openai chat | ✅ |
| mem0 | openai | 2 / 2 | exact | exact | 2/2 | openai chat, openai.embeddings.create | ✅ |
| camel-ai | openai | 1 / 1 | exact | exact | 1/1 | openai chat | ✅ |
| griptape | openai | 1 / 1 | exact | exact | 1/1 | openai chat | ✅ |
| langroid | openai | 1 / 1 | exact | exact | 1/1 | openai chat | ✅ |
| ragas | openai | 1 / 1 | exact | exact | 1/1 | openai chat | ✅ |
| deepeval | openai | 1 / 1 | exact | exact | 1/1 | openai chat | ✅ |
| vanna | openai | 1 / 1 | exact | exact | 1/1 | openai chat | ✅ |
| txtai | openai | 1 / 1 | exact | exact | 1/1 | litellm.completion | ✅ |
| atomic-agents | openai | 1 / 1 | exact | exact | 1/1 | openai chat | ✅ |
| controlflow-like:burr | openai | 1 / 1 | exact | exact | 1/1 | openai chat | ✅ |
| langchain-litellm | openai | 1 / 1 | exact | exact | 1/1 | litellm.completion | ✅ |
| pandasai | openai | 5 / 5 | exact | exact | 5/5 | openai chat | ✅ |

## Alongside existing tracing

| Project | API | Calls (server / recorded) | Input tokens | Output tokens | App line | Recorded via | Result |
|---|---|---|---|---|---|---|---|
| openinference-otel | openai | 1 / 1 | exact | exact | 1/1 | openai chat | ✅ |
| openllmetry-otel | openai | 1 / 1 | exact | exact | 1/1 | openai chat | ✅ |
| langfuse-openai | openai | 1 / 1 | exact | exact | 1/1 | openai chat | ✅ |

## Not included

- **Marvin** (Prefect's 3.x alpha): its pinned dependencies don't import together, with or without
  trainlab, so there is nothing to measure.
- **Legacy `vertexai` SDK**: forces HTTPS to its endpoint, so it can't reach a local test server.
  Vertex is tested through the current Google Gen AI SDK (`vertexai=True`).
- **Legacy `google-generativeai`** is tested with `transport="rest"`. Its default transport is gRPC,
  which trainlab does not see; that SDK is deprecated in favour of `google-genai`.

Reproduce: `python bench/frameworks/install.py ENVS && python bench/frameworks/run.py ENVS`.
