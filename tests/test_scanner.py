from pathlib import Path

from trainlab_optimize.scanner import scan
from trainlab_optimize.scanner.python_ast import PythonFileScanner

DEMO = Path(__file__).resolve().parents[1] / "examples/demo_app"


def sites_of(src: str, path: str = "app.py"):
    return PythonFileScanner(path, src).scan()


def test_demo_app_golden():
    r = scan(DEMO)
    by = {(s.path, s.line): s for s in r.call_sites}
    assert len(r.call_sites) == 7
    inv = by[("services/invoice_extractor.py", 19)]
    assert inv.model == "gpt-4o" and inv.model_source == "constant"
    assert inv.retry_unbounded and "extraction" in inv.task_hints
    assert by[("services/invoice_extractor.py", 34)].in_loop
    rag = by[("services/support_rag.py", 33)]
    assert rag.provider == "anthropic" and rag.system_prompt_tokens > 1500
    assert rag.prompt_dynamic_prefix and not rag.uses_cache_control
    clf = by[("services/ticket_classifier.py", 19)]
    assert clf.provider == "litellm" and clf.model == "gpt-4.1" and clf.model_source == "env_default"
    assert by[("services/legacy_summary.py", 10)].in_loop  # list comprehension
    ts = by[("frontend/chat.ts", 7)]
    assert ts.confidence == "heuristic" and ts.model == "gpt-4o"
    assert {m["model"] for m in r.model_refs} == {"claude-sonnet-4-5", "text-embedding-3-large"}


def test_injected_client_without_import():
    s = sites_of("def create(x):\n    return x.chat.completions.create(model='gpt-4o', messages=[])\n")
    assert [(x.kind, x.provider, x.model, x.confidence) for x in s] == [("chat", "openai", "gpt-4o", "heuristic")]
    s = sites_of("def create(x, m):\n    return x.chat.completions.create(model=m, messages=[])\n")
    assert [(x.kind, x.model) for x in s] == [("chat", None)]
    assert sites_of("def create(x):\n    return x.chat.completions.create(foo=1)\n") == []


def test_param_default_and_self_attr():
    src = '''
from openai import OpenAI
class Svc:
    model = "gpt-4.1-mini"
    def __init__(self):
        self.client = OpenAI()
    def run(self, q, model="gpt-4o"):
        a = self.client.chat.completions.create(model=model, messages=[])
        b = self.client.chat.completions.create(model=self.model, messages=[])
        return a, b
'''
    s = sites_of(src)
    assert [x.model for x in s] == ["gpt-4o", "gpt-4.1-mini"]
    assert s[0].model_source == "param_default"


def test_openai_compatible_base_url():
    src = '''
from openai import OpenAI
client = OpenAI(base_url="https://api.groq.com/openai/v1")
client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[])
'''
    s = sites_of(src)
    assert s[0].provider == "groq"


def test_azure_and_bedrock_clients():
    src = '''
import boto3
from openai import AzureOpenAI
az = AzureOpenAI()
br = boto3.client("bedrock-runtime")
az.chat.completions.create(model="prod-gpt4o", messages=[])
br.converse(modelId="anthropic.claude-haiku-4-5-20251001-v1:0", messages=[])
'''
    s = sites_of(src)
    assert [x.provider for x in s] == ["azure", "bedrock"]


def test_cache_control_detected():
    src = '''
import anthropic
c = anthropic.Anthropic()
c.messages.create(model="claude-sonnet-4-5", max_tokens=100,
    system=[{"type": "text", "text": "x" * 10, "cache_control": {"type": "ephemeral"}}],
    messages=[])
'''
    assert sites_of(src)[0].uses_cache_control


def test_bounded_retry_not_flagged():
    src = '''
from openai import OpenAI
from tenacity import retry, stop_after_attempt
c = OpenAI()
@retry(stop=stop_after_attempt(3))
def f():
    return c.responses.create(model="gpt-5-mini", input="hi")
'''
    s = sites_of(src)
    assert s[0].has_retry and not s[0].retry_unbounded


def test_langchain_config_site():
    src = '''
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)
'''
    s = sites_of(src)
    assert s[0].kind == "model_config" and s[0].model == "gpt-4o"


def test_tests_excluded_by_default(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_x.py").write_text(
        "from openai import OpenAI\nOpenAI().chat.completions.create(model='gpt-4o', messages=[])\n")
    assert scan(tmp_path).call_sites == []
    assert len(scan(tmp_path, include_tests=True).call_sites) == 1


def test_syntax_error_file_skipped(tmp_path):
    (tmp_path / "bad.py").write_text("import openai\ndef (:\n")
    r = scan(tmp_path)
    assert r.call_sites == [] and r.parse_errors == ["bad.py"]


def test_twilio_messages_create_ignored():
    src = '''
from openai import OpenAI
from twilio.rest import Client
sms = Client()
sms.messages.create(body="hi", to="+1", from_="+2")
'''
    assert sites_of(src) == []


def test_wrapper_config_without_llm_import():
    src = '''
from mylib import LiteLLMModel, Settings
m = LiteLLMModel(name="gpt-4o")
s = Settings(llm="claude-sonnet-4-5", temperature=0)
t = Thing(name="not-a-model")
'''
    s = sites_of(src)
    assert [(x.kind, x.model, x.confidence) for x in s] == [
        ("model_config", "gpt-4o", "heuristic"), ("model_config", "claude-sonnet-4-5", "heuristic")]


def test_exclude(tmp_path):
    (tmp_path / "examples").mkdir()
    (tmp_path / "app.py").write_text("from openai import OpenAI\nOpenAI().responses.create(model='gpt-4o', input='x')\n")
    (tmp_path / "examples/demo.py").write_text("from openai import OpenAI\nOpenAI().responses.create(model='gpt-4o', input='x')\n")
    assert len(scan(tmp_path).call_sites) == 2
    assert [s.path for s in scan(tmp_path, exclude=["examples"]).call_sites] == ["app.py"]


def test_litellm_via_project_wrapper():
    src = "from myapp.llm import litellm\ndef f(**kw):\n    return litellm.completion(**kw)\n"
    s = sites_of(src)
    assert [(x.kind, x.provider) for x in s] == [("chat", "litellm")]


def test_client_from_factory_without_import():
    src = '''
from myapp.helpers import get_openai_client
def embed(self, docs):
    client = get_openai_client(self.key)
    return client.embeddings.create(input=docs, model=self.model_name)
'''
    s = sites_of(src)
    assert [(x.kind, x.provider, x.confidence) for x in s] == [("embedding", "openai", "heuristic")]


def test_aioboto3_bedrock_async_with():
    src = '''
import aioboto3
async def run(session, req):
    async with session.client(service_name="bedrock-runtime") as client:
        return await client.converse(**req)
'''
    s = sites_of(src)
    assert [(x.kind, x.provider) for x in s] == [("chat", "bedrock")]


def test_js_bedrock_command():
    from trainlab_optimize.scanner.generic import scan_js
    js = 'const out = await client.send(new ConverseCommand({ modelId: "anthropic.claude-haiku-4-5-20251001-v1:0", messages }));'
    s = scan_js("a.ts", js)
    assert s[0].provider == "bedrock" and s[0].model.startswith("anthropic.claude-haiku")


def test_loop_kinds_and_scope():
    src = '''
from openai import OpenAI
c = OpenAI()
def batch(docs):
    return [c.responses.create(model="gpt-4o", input=d) for d in docs]
def agent():
    while True:
        r = c.responses.create(model="gpt-4o", input="x")
'''
    s = PythonFileScanner("examples/a.py", src).scan()
    assert [x.loop_kind for x in s] == ["comprehension", "while"]
    assert all(x.scope == "example" for x in s)


def test_short_model_names_are_refs():
    s = PythonFileScanner("cfg.py", 'REASONING = "o3"\nFAST = "gpt-5-mini"\n')
    s.scan()
    assert sorted(v for _, v in s.model_refs) == ["gpt-5-mini", "o3"]


def test_deeply_nested_expression_does_not_crash(tmp_path):
    expr = " + ".join(["'a'"] * 3000)
    (tmp_path / "deep.py").write_text(f"import openai\nx = {expr}\n")
    r = scan(tmp_path)
    assert r.call_sites == []


def test_splat_payload_in_langchain_core_file():
    src = '''
from langchain_core.language_models import BaseChatModel
class ChatX(BaseChatModel):
    def _generate(self, payload):
        return self.client.chat.completions.create(**payload)
'''
    s = sites_of(src)
    assert [(x.kind, x.sdk_call) for x in s] == [("chat", "self.client.chat.completions.create")]


def test_litellm_alias_and_partial():
    src = '''
from functools import partial
from litellm import completion as litellm_completion
def setup(cfg):
    fn = partial(litellm_completion, model="claude-sonnet-4-5", temperature=0)
    return litellm_completion(model=cfg.model, messages=[])
'''
    s = sites_of(src)
    assert [(x.kind, x.provider, x.model) for x in s] == [
        ("chat", "litellm", "claude-sonnet-4-5"), ("chat", "litellm", None)]


def test_raw_http_endpoints_and_server_routes():
    src = '''
import aiohttp
from fastapi import APIRouter
router = APIRouter()

@router.post("/chat/completions")
async def proxy(payload, url):
    request_url = f"{url}/chat/completions"
    async with aiohttp.ClientSession() as s:
        return await s.post(request_url, json=payload)

ANTHROPIC = "https://api.anthropic.com/v1/messages"
OWN = "/api/chat/completions"
DOC = """Call the /v1/chat/completions endpoint like this."""
'''
    s = sites_of(src)
    assert [(x.line, x.kind, x.provider, x.sdk_call) for x in s] == [
        (8, "chat", "openai_compatible", "HTTP /chat/completions"),
        (12, "chat", "anthropic", "HTTP /v1/messages")]
    assert s[0].function == "proxy"


def test_go_java_csharp_sdks():
    from trainlab_optimize.scanner.generic import scan_heuristic
    go = '''import "github.com/openai/openai-go"
resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
    Model: "gpt-4o-mini",
})'''
    s = scan_heuristic("svc/llm.go", go, "go")
    assert [(x.kind, x.provider, x.model, x.language) for x in s] == [("chat", "openai", "gpt-4o-mini", "go")]
    java = '''import com.openai.client.OpenAIClient;
ChatCompletionCreateParams p = ChatCompletionCreateParams.builder().model(ChatModel.GPT_4O).build();
ChatCompletion c = client.chat().completions().create(p);'''
    s = scan_heuristic("src/main/java/A.java", java, "jvm")
    assert [(x.kind, x.provider, x.model) for x in s] == [("chat", "openai", "gpt-4o")]
    cs = '''using OpenAI.Chat;
ChatClient client = new(model: "gpt-4o", apiKey: key);
ChatCompletion completion = await client.CompleteChatAsync(messages);'''
    s = scan_heuristic("Svc/Chat.cs", cs, "csharp")
    assert [(x.kind, x.provider, x.model) for x in s] == [("chat", "openai", "gpt-4o")]


def test_go_http_endpoint_and_route_exclusion():
    from trainlab_optimize.scanner.generic import scan_heuristic
    go = '''httpReq, err := http.NewRequest(http.MethodPost, m.baseURL+"/chat/completions", body)
r.POST("/v1/chat/completions", handler)
// POST /v1/messages
const anthropicURL = "https://api.anthropic.com/v1/messages"'''
    s = scan_heuristic("internal/llm.go", go, "go")
    assert [(x.line, x.provider, x.sdk_call) for x in s] == [
        (1, "openai_compatible", "HTTP /chat/completions"), (4, "anthropic", "HTTP /v1/messages")]


def test_ts_fetch_to_own_backend_ignored():
    from trainlab_optimize.scanner.generic import scan_heuristic
    ts = '''const r = await fetch(`${WEBUI_BASE_URL}/api/chat/completions`, {method: "POST"})
const o = await fetch("https://api.openai.com/v1/chat/completions", {method: "POST"})'''
    s = scan_heuristic("src/lib/apis.ts", ts, "javascript")
    assert [(x.line, x.provider) for x in s] == [(2, "openai")]


def test_csharp_azure_inference_and_server_path_check():
    from trainlab_optimize.scanner.generic import scan_heuristic
    cs = '''using Azure.AI.Inference;
var reply = await this.chatCompletionsClient.CompleteAsync(settings, cancellationToken: ct);
if (context.Request.Path == "/v1/chat/completions") { await next(context); }'''
    s = scan_heuristic("src/Agent.cs", cs, "csharp")
    assert [(x.line, x.provider, x.sdk_call) for x in s] == [(2, "azure", "CompleteAsync")]


def test_helper_with_model_kwarg_is_not_a_config():
    src = '''
from mylib import RestrictModel
def fmt(m):
    return to_openai_responses_message_dict(m, model="o3-mini")
allowed = RestrictModel(model="gpt-4")
llm = get_llm(model="gpt-4o")
'''
    assert [(x.line, x.model) for x in sites_of(src)] == [(5, "gpt-4"), (6, "gpt-4o")]


def test_cross_file_prompt_and_model(tmp_path):
    pkg = tmp_path / "app"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    big = "Follow these support rules carefully. " * 200
    (pkg / "prompts.py").write_text(f'BASE = "{big}"\nSYSTEM_PROMPT = BASE + "Answer briefly."\nMODEL = "claude-sonnet-4-5"\n')
    (pkg / "service.py").write_text('''
import anthropic
from .prompts import SYSTEM_PROMPT, MODEL
from app import prompts as P
c = anthropic.Anthropic()
def answer(q):
    return c.messages.create(model=MODEL, max_tokens=300, system=SYSTEM_PROMPT,
                             messages=[{"role": "user", "content": q}])
def answer2(q):
    return c.messages.create(model=P.MODEL, max_tokens=300, system=P.SYSTEM_PROMPT, messages=[])
''')
    r = scan(tmp_path)
    assert [(s.model, s.model_source) for s in r.call_sites] == [("claude-sonnet-4-5", "constant")] * 2
    assert all(s.system_prompt_tokens > 1500 for s in r.call_sites)


def test_notebook_cells(tmp_path):
    import json
    nb = {"metadata": {"kernelspec": {"language": "python"}}, "cells": [
        {"cell_type": "markdown", "source": ["# Demo"]},
        {"cell_type": "code", "source": ["!pip install openai\n", "%load_ext autoreload\n",
                                         "from openai import OpenAI\n", "client = OpenAI()\n"]},
        {"cell_type": "code", "source": "resp = client.chat.completions.create(\n    model='gpt-4o', messages=[])"},
    ]}
    (tmp_path / "demo.ipynb").write_text(json.dumps(nb))
    r = scan(tmp_path)
    s = r.call_sites[0]
    assert (s.cell, s.line, s.model, s.provider, s.scope) == (2, 1, "gpt-4o", "openai", "example")
    assert s.location == "demo.ipynb:cell 2:1"


def test_bom_nullbyte_minified_and_broken_symlink(tmp_path):
    (tmp_path / "bom.py").write_bytes(b"\xef\xbb\xbffrom openai import OpenAI\nOpenAI().responses.create(model='gpt-4o', input='x')\n")
    (tmp_path / "app.min.js").write_text("openai.chat.completions.create({model:'gpt-4o'})")
    (tmp_path / "dangling.py").symlink_to(tmp_path / "missing.py")
    r = scan(tmp_path)
    assert [s.path for s in r.call_sites] == ["bom.py"]


def test_ruby_php_rust_kotlin():
    from trainlab_optimize.scanner.generic import scan_heuristic
    rb = 'response = client.chat(parameters: { model: "gpt-4o-mini", messages: [{ role: "user", content: q }] })'
    assert [(x.provider, x.model) for x in scan_heuristic("app/services/ai.rb", rb, "ruby")] == [("openai", "gpt-4o-mini")]
    php = "$result = $client->chat()->create([\n    'model' => 'gpt-4o',\n    'messages' => $messages,\n]);"
    assert [(x.provider, x.model) for x in scan_heuristic("app/Ai.php", php, "php")] == [("openai", "gpt-4o")]
    rs = 'use async_openai::Client;\nlet req = CreateChatCompletionRequestArgs::default().model("gpt-4o-mini").build()?;\nlet resp = client.chat().create(req).await?;'
    assert [(x.provider, x.model) for x in scan_heuristic("src/main.rs", rs, "rust")] == [("openai", "gpt-4o-mini")]
    rig = 'use rig::providers::openai;\nlet agent = client.agent(openai::GPT_4O).preamble("You are helpful").build();'
    assert [(x.kind, x.model) for x in scan_heuristic("src/agent.rs", rig, "rust")] == [("model_config", "gpt-4o")]
    kt = 'import com.aallam.openai.client.OpenAI\nval req = ChatCompletionRequest(model = ModelId("gpt-4o-mini"), messages = msgs)\nval c = openAI.chatCompletion(req)'
    assert [(x.provider, x.model) for x in scan_heuristic("app/src/main/kotlin/Chat.kt", kt, "jvm")] == [("openai", "gpt-4o-mini")]


def test_notebook_top_level_await_and_broken_cell(tmp_path):
    import json
    nb = {"metadata": {"kernelspec": {"language": "python"}}, "cells": [
        {"cell_type": "code", "source": "import anthropic\nclient = anthropic.AsyncAnthropic()"},
        {"cell_type": "code", "source": "%%bash\necho hi"},
        {"cell_type": "code", "source": "def broken(:\n    pass"},
        {"cell_type": "code", "source": "r = await client.messages.create(model='claude-haiku-4-5', max_tokens=50, messages=[])"},
    ]}
    (tmp_path / "nb.ipynb").write_text(json.dumps(nb))
    s = scan(tmp_path).call_sites
    assert [(x.cell, x.model) for x in s] == [(4, "claude-haiku-4-5")]


def test_langchain4j_any_builder_and_semantic_kernel():
    from trainlab_optimize.scanner.generic import scan_heuristic
    java = 'import dev.langchain4j.model.bedrock.BedrockChatModel;\nChatModel m = BedrockChatModel.builder()\n    .modelId("anthropic.claude-haiku-4-5-20251001-v1:0")\n    .build();'
    assert [(x.kind, x.model) for x in scan_heuristic("src/main/java/A.java", java, "jvm")] == [
        ("model_config", "anthropic.claude-haiku-4-5-20251001-v1:0")]
    cs = 'using Microsoft.SemanticKernel;\nbuilder.AddOpenAIChatCompletion(modelId: "gpt-4o-mini", apiKey: key);\nvar r = await chat.GetChatMessageContentAsync(history);'
    assert [(x.kind, x.model) for x in scan_heuristic("src/P.cs", cs, "csharp")] == [
        ("model_config", "gpt-4o-mini"), ("chat", "gpt-4o-mini")]
    py = 'from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion\nsvc = OpenAIChatCompletion(ai_model_id="gpt-4o")\n'
    assert [(x.kind, x.model) for x in sites_of(py)] == [("model_config", "gpt-4o")]


def test_test_like_folders(tmp_path):
    for d in ("ecosystem-tests", "integration_tests", "tests-e2e", "latest", "contest"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "a.py").write_text("from openai import OpenAI\nOpenAI().responses.create(model='gpt-4o', input='x')\n")
    assert sorted(s.path.split("/")[0] for s in scan(tmp_path).call_sites) == ["contest", "latest"]


def test_latin1_crlf_and_unreadable(tmp_path):
    (tmp_path / "latin.py").write_bytes("# caf\xe9\r\nfrom openai import OpenAI\r\nOpenAI().responses.create(model='gpt-4o', input='x')\r\n".encode("latin-1"))
    bad = tmp_path / "secret.py"
    bad.write_text("from openai import OpenAI\nOpenAI().responses.create(model='gpt-4o', input='x')\n")
    bad.chmod(0)
    r = scan(tmp_path)
    paths = [s.path for s in r.call_sites]
    assert "latin.py" in paths and r.call_sites[0].line == 3
    bad.chmod(0o644)


def test_client_post_to_provider_host_is_a_call():
    from trainlab_optimize.scanner.generic import scan_heuristic
    rs = '''let res = client.post("https://api.anthropic.com/v1/messages").json(&body).send().await?;
router.post("/v1/chat/completions", handler);'''
    s = scan_heuristic("src/llm/claude.rs", rs, "rust")
    assert [(x.line, x.provider) for x in s] == [(1, "anthropic")]


def test_router_vendor_namespaces_match_code_names():
    from trainlab_optimize.analyzer.spend import base_model
    for billed, code in [("qwen/qwen-plus", "qwen-plus"), ("meta-llama/llama-3.3-70b-instruct", "llama-3.3-70b-instruct"),
                         ("moonshotai/kimi-k2", "kimi-k2"), ("deepseek/deepseek-chat:free", "deepseek-chat")]:
        assert base_model(billed) == base_model(code)


def test_jvm_round1_gaps():
    from trainlab_optimize.scanner.generic import OTHER_TEST, scan_heuristic
    kt = '''import aws.sdk.kotlin.services.bedrockruntime.BedrockRuntimeClient
val resp = client.converse(ConverseRequest { modelId = "anthropic.claude-haiku-4-5-20251001-v1:0" })'''
    assert [(x.provider, x.model) for x in scan_heuristic("App.kt", kt, "jvm")] == [
        ("bedrock", "anthropic.claude-haiku-4-5-20251001-v1:0")]
    java = '''import com.anthropic.client.AnthropicClient;
try (var s = client.messages().createStreaming(params)) { }'''
    assert [x.provider for x in scan_heuristic("Ex.java", java, "jvm")] == ["anthropic"]
    spring = '''import org.springframework.ai.openai.OpenAiChatModel;
var chatModel = OpenAiChatModel.builder().defaultOptions(opts).build();'''
    assert [(x.kind, x.provider) for x in scan_heuristic("Cfg.java", spring, "jvm")] == [("model_config", "openai")]
    assert OTHER_TEST.search("lib/src/jvmTest/kotlin/ClientTestBase.kt")


def test_dotnet_interactive_notebook(tmp_path):
    import json as _json
    nb = {"metadata": {"kernelspec": {"language": "C#", "name": ".net-csharp"}},
          "cells": [{"cell_type": "markdown", "source": ["# Q&A"]},
                    {"cell_type": "code", "metadata": {"dotnet_interactive": {"language": "csharp"}},
                     "source": ["#r \"nuget: Azure.AI.OpenAI\"\n", "using Azure.AI.OpenAI;\n"]},
                    {"cell_type": "code", "metadata": {},
                     "source": ["var options = new ChatCompletionsOptions { DeploymentName = \"gpt-4o\" };\n",
                                "var response = await client.GetChatCompletionsAsync(options);\n"]}]}
    (tmp_path / "qa.ipynb").write_text(_json.dumps(nb))
    s = scan(tmp_path).call_sites
    assert [(x.provider, x.cell, x.line, x.scope) for x in s] == [("azure", 2, 2, "example")]


def test_absolute_url_with_api_segment_is_an_endpoint():
    from trainlab_optimize.scanner.python_ast import http_endpoint
    assert http_endpoint("https://internlm-chat.intern-ai.org.cn/puyu/api/v1/chat/completions")[1] == "internlm"
    assert http_endpoint("/api/chat/completions") is None


def test_name_assigned_many_times_resolves_fast():
    import time
    lines = ["import litellm"]
    for i in range(300):
        lines += [f"model = m{i % 7}", f"m{i % 7} = model"]
    lines.append("litellm.completion(model=model, messages=[])")
    t0 = time.time()
    assert len(sites_of("\n".join(lines))) == 1 and time.time() - t0 < 2


def test_bench_round2_gaps():
    from trainlab_optimize.scanner.generic import scan_heuristic
    go = 'func run(b string) {\n\turl := strings.TrimRight(b, "/") + "/chat/completions"\n}'
    assert [(x.line, x.sdk_call) for x in scan_heuristic("cmd/bench/bench.go", go, "go")] == [(2, "HTTP /chat/completions")]
    rb = 'resp = client.converse(\n  model_id: "anthropic.claude-haiku-4-5-20251001-v1:0",\n  messages: msgs)'
    assert [x.provider for x in scan_heuristic("tools/snippets.rb", rb, "ruby")] == ["bedrock"]
    assert [x.sdk_call for x in scan_heuristic("s.rb", 'r = client.responses.create(model: "gpt-5", input: h)', "ruby")] \
        == ["responses.create"]
    # OpenAI-shaped client called positionally (g4f style), without an openai import
    assert [x.kind for x in sites_of("from g4f.client import Client\nc = Client()\nc.chat.completions.create(msgs, 'gpt-4o')\n")] \
        == ["chat"]
    from trainlab_optimize.scanner.python_ast import TEST_PATH_RE
    assert TEST_PATH_RE.search("etc/unittest/client.py")


def test_bench_round3_gaps(tmp_path):
    import json as _json
    from trainlab_optimize.scanner.generic import scan_heuristic
    twilio = 'const twilio = require("twilio")(sid, token);\ntwilio.messages.create({ from, to, body: "hi" });'
    assert scan_heuristic("examples/example.js", twilio, "javascript") == []
    passed_client = 'export async function ask(client) {\n  return client.messages.create({ model: "claude-haiku-4-5", max_tokens: 50, messages });\n}'
    assert [x.provider for x in scan_heuristic("src/ask.ts", passed_client, "javascript")] == ["anthropic"]
    nb = {"metadata": {"kernelspec": {"language": "python", "name": "python3"}},
          "cells": [{"cell_type": "code", "metadata": {"dotnet_interactive": {"language": "csharp"}},
                     "source": ["#r \"nuget: Azure.AI.OpenAI\"\n",
                                "var response = await client.GetChatCompletionsAsync(options);\n"]},
                    {"cell_type": "code", "metadata": {}, "source": ["from openai import OpenAI\n",
                     "OpenAI().chat.completions.create(model='gpt-4o', messages=[])\n"]}]}
    (tmp_path / "mixed.ipynb").write_text(_json.dumps(nb))
    got = sorted((x.provider, x.cell) for x in scan(tmp_path).call_sites)
    assert got == [("azure", 1), ("openai", 2)]
    assert [x.kind for x in sites_of("async def f(client):\n    await client.chat.completions.create(messages='hi', provider=p)\n")] == ["chat"]


def test_fstring_on_variable_host_is_a_client_call():
    src = 'import httpx\ndef f(endpoint):\n    url = f"{endpoint}/api/v1/chat/completions"\n    return httpx.post(url)\n'
    assert [(x.kind, x.sdk_call) for x in sites_of(src)] == [("chat", "HTTP /v1/chat/completions")]
    # a FastAPI route literal stays excluded
    assert sites_of('from fastapi import FastAPI\napp = FastAPI()\n@app.post("/api/v1/chat/completions")\ndef h(): pass\n') == []
