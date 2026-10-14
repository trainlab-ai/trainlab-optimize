"""Provider coverage: routers, Chinese and open-model hosts, native SDKs, self-hosted runtimes."""
from trainlab_optimize.pricing import PriceCatalog
from trainlab_optimize.scanner.generic import scan_heuristic
from trainlab_optimize.scanner.python_ast import PythonFileScanner


def py(src: str, path: str = "app/x.py"):
    return [(s.provider, s.model, s.kind) for s in PythonFileScanner(path, src).scan()]


def js(src: str, path: str = "src/x.ts"):
    return [(s.provider, s.model, s.kind) for s in scan_heuristic(path, src, "javascript")]


def test_openai_sdk_pointed_at_other_providers():
    cases = {
        "https://openrouter.ai/api/v1": "openrouter",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1": "dashscope",
        "https://api.deepseek.com": "deepseek",
        "https://api.moonshot.ai/v1": "moonshot",
        "https://open.bigmodel.cn/api/paas/v4": "zhipu",
        "https://ark.cn-beijing.volces.com/api/v3": "volcengine",
        "https://api.x.ai/v1": "xai",
        "https://api.together.xyz/v1": "together",
        "https://integrate.api.nvidia.com/v1": "nvidia",
        "http://localhost:11434/v1": "ollama",
    }
    for url, provider in cases.items():
        src = f'''
from openai import OpenAI
client = OpenAI(base_url="{url}", api_key="x")
client.chat.completions.create(model="m-1", messages=[])
'''
        assert py(src)[0][0] == provider, url


def test_base_url_from_env_default_and_key_name():
    src = '''
import os
from openai import OpenAI
BASE = os.getenv("LLM_BASE", "https://openrouter.ai/api/v1")
client = OpenAI(base_url=BASE, api_key=os.environ["OPENROUTER_API_KEY"])
client.chat.completions.create(model="anthropic/claude-sonnet-4.5", messages=[])
'''
    assert py(src) == [("openrouter", "anthropic/claude-sonnet-4.5", "chat")]
    src2 = '''
import os
from openai import OpenAI
client = OpenAI(base_url=os.environ["BASE_URL"], api_key=os.environ["DASHSCOPE_API_KEY"])
client.chat.completions.create(model="qwen-plus", messages=[])
'''
    assert py(src2)[0][0] == "dashscope"


def test_injected_client_uses_file_hint():
    src = '''
from openai import OpenAI
DEEPSEEK_URL = "https://api.deepseek.com"
class Svc:
    def run(self):
        return self.client.chat.completions.create(model="deepseek-chat", messages=[])
'''
    assert py(src)[0][0] == "deepseek"


def test_native_python_sdks():
    src = '''
import dashscope, ollama, replicate, cohere, voyageai
from dashscope import Generation
from mistralai import Mistral
from groq import Groq
from zhipuai import ZhipuAI
from huggingface_hub import InferenceClient
import google.generativeai as genai

Generation.call(model="qwen-max", messages=[])
dashscope.TextEmbedding.call(model="text-embedding-v3", input="x")
Mistral(api_key="k").chat.complete(model="mistral-small-latest", messages=[])
m = Mistral(api_key="k")
m.chat.complete(model="mistral-large-latest", messages=[])
co = cohere.ClientV2()
co.chat(model="command-r-plus", messages=[])
co.embed(model="embed-english-v3.0", texts=["a"])
ollama.chat(model="llama3.1:8b", messages=[])
replicate.run("meta/meta-llama-3-70b-instruct", input={})
g = Groq()
g.chat.completions.create(model="llama-3.3-70b-versatile", messages=[])
z = ZhipuAI()
z.chat.completions.create(model="glm-4.6", messages=[])
hf = InferenceClient()
hf.chat_completion(model="meta-llama/Llama-3.1-8B-Instruct", messages=[])
vo = voyageai.Client()
vo.embed(["a"], model="voyage-3")
gm = genai.GenerativeModel("gemini-1.5-flash")
'''
    got = py(src)
    assert ("dashscope", "qwen-max", "chat") in got
    assert ("dashscope", "text-embedding-v3", "embedding") in got
    assert ("mistral", "mistral-large-latest", "chat") in got
    assert ("cohere", "command-r-plus", "chat") in got
    assert ("cohere", "embed-english-v3.0", "embedding") in got
    assert ("ollama", "llama3.1:8b", "chat") in got
    assert ("replicate", "meta/meta-llama-3-70b-instruct", "chat") in got
    assert ("groq", "llama-3.3-70b-versatile", "chat") in got
    assert ("zhipu", "glm-4.6", "chat") in got
    assert ("huggingface", "meta-llama/Llama-3.1-8B-Instruct", "chat") in got
    assert ("voyage", "voyage-3", "embedding") in got
    assert ("gemini", "gemini-1.5-flash", "model_config") in got


def test_native_methods_need_the_sdk():
    # .chat()/.embed() on arbitrary objects are not LLM calls
    src = '''
from slack_sdk import WebClient
c = WebClient()
c.chat(model="x", messages=[])
'''
    assert py(src) == []


def test_langchain_and_llamaindex_classes():
    src = '''
from langchain_community.chat_models import ChatTongyi
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from llama_index.llms.openrouter import OpenRouter
ChatTongyi(model="qwen-max")
ChatGroq(model="llama-3.1-8b-instant")
ChatOllama(model="llama3.1")
OpenRouter(model="openai/gpt-4o-mini")
'''
    assert [p for p, _, _ in py(src)] == ["dashscope", "groq", "ollama", "openrouter"]


def test_aisuite_and_router_model_prefixes():
    src = '''
import aisuite as ai
client = ai.Client()
client.chat.completions.create(model="groq:llama-3.3-70b-versatile", messages=[])
'''
    assert py(src)[0][0] == "groq"


def test_raw_http_dashscope_ollama_cohere():
    src = '''
import requests
requests.post("https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation", json={})
requests.post("http://localhost:11434/api/chat", json={})
requests.post("https://api.cohere.com/v2/chat", json={})
requests.post("https://api.example.com/api/chat", json={})
'''
    assert [p for p, _, _ in py(src)] == ["dashscope", "ollama", "cohere"]


def test_js_providers():
    src = '''
import OpenAI from "openai";
const client = new OpenAI({ baseURL: "https://openrouter.ai/api/v1", apiKey: process.env.OPENROUTER_API_KEY });
await client.chat.completions.create({ model: "deepseek/deepseek-chat", messages });
'''
    assert js(src) == [("openrouter", "deepseek/deepseek-chat", "chat")]
    src = '''
import { Mistral } from "@mistralai/mistralai";
const m = new Mistral({ apiKey });
await m.chat.complete({ model: "mistral-small-latest", messages });
'''
    assert js(src) == [("mistral", "mistral-small-latest", "chat")]
    src = '''
import { CohereClientV2 } from "cohere-ai";
await cohere.chat({ model: "command-a-03-2025", messages });
'''
    assert js(src) == [("cohere", "command-a-03-2025", "chat")]
    src = '''
import ollama from "ollama";
await ollama.chat({ model: "qwen2.5:7b", messages });
'''
    assert js(src) == [("ollama", "qwen2.5:7b", "chat")]
    src = '''
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
const r = await generateText({ model: openrouter("anthropic/claude-sonnet-4.5"), prompt });
'''
    assert js(src) == [("openrouter", "anthropic/claude-sonnet-4.5", "chat")]
    src = '''
import { ChatGroq } from "@langchain/groq";
const llm = new ChatGroq({ model: "llama-3.3-70b-versatile" });
'''
    assert js(src) == [("groq", "llama-3.3-70b-versatile", "model_config")]


def test_prices_for_market_models():
    cat = PriceCatalog.load()
    for model, provider in [("qwen-plus", "dashscope"), ("qwen-max", None), ("glm-4.6", None),
                            ("kimi-k2.5", None), ("MiniMax-M2", None), ("deepseek-chat", None),
                            ("mistral-large-latest", None), ("command-r-plus", None), ("sonar-pro", None),
                            ("anthropic/claude-sonnet-4.5", "openrouter"), ("llama-3.3-70b-versatile", "groq"),
                            ("voyage-3", None), ("claude-opus-4-1", None)]:
        assert cat.lookup(model, provider), model
    # maker's own price wins over a host's
    assert cat.lookup("qwen-plus").provider == "dashscope"
    assert cat.lookup("command-r-plus").provider.startswith("cohere")
    # a host's price for an open model is flagged as approximate
    assert cat.lookup("llama-3.3-70b-versatile", "groq").approximate


def test_openrouter_url_is_not_a_server_route():
    src = "const OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions';\nrouter.post('/v1/chat/completions', h);"
    assert js(src) == [("openrouter", None, "chat")]


def test_azure_foundry_client_factories():
    src = '''
from azure.ai.projects import AIProjectClient
project = AIProjectClient(endpoint="x", credential=None)
openai = project.get_openai_client()
openai.responses.create(conversation="c", input="hi", extra_body={})
'''
    assert py(src) == [("azure", None, "chat")]
    src = '''
from agent_framework import Agent
async def run(prompt):
    client = await _openai_client()
    return await client.responses.create(extra_body={}, input=prompt, store=False)
'''
    assert [p for p, _, _ in py(src)] == ["openai"]
