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
    s = sites_of("def create(x):\n    return x.chat.completions.create(model='gpt-4o')\n")
    assert [(x.kind, x.provider, x.model, x.confidence) for x in s] == [("chat", "openai", "gpt-4o", "heuristic")]
    assert sites_of("def create(x):\n    return x.chat.completions.create(model=m)\n") == []


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
    assert s[0].provider == "openai_compatible"


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
