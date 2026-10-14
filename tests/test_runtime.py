"""Profile 2 runtime recorder: real SDKs, mocked HTTP. Every test checks what was recorded and
that the app saw exactly what the SDK returned."""
import asyncio
import json
import warnings

import pytest

import trainlab_optimize as trainlab

httpx = pytest.importorskip("httpx")

HERE = "tests/test_runtime.py"


@pytest.fixture()
def events(tmp_path):
    got = []
    trainlab.instrument(path=False, project_root=_root(), sinks=[_ListSink(got)])
    yield got
    trainlab.uninstrument()


class _ListSink:
    def __init__(self, out):
        self.out = out

    def write(self, ev):
        self.out.append(ev.to_json())

    def flush(self):
        pass


def _root():
    import os
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _lineno(marker):
    for i, line in enumerate(open(__file__, encoding="utf-8"), 1):
        if f"# @{marker}" in line:
            return i
    raise AssertionError(marker)


# ------------------------------------------------------------------ OpenAI
def _openai_handler(request):
    body = json.loads(request.content)
    if request.url.path.endswith("/embeddings"):
        return httpx.Response(200, json={"object": "list", "model": body["model"], "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1]}],
            "usage": {"prompt_tokens": 7, "total_tokens": 7}})
    if request.url.path.endswith("/responses"):
        return httpx.Response(200, json={
            "id": "resp_1", "object": "response", "created_at": 1, "status": "completed", "model": body["model"],
            "output": [{"type": "message", "id": "m1", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "hi", "annotations": []}]}],
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "usage": {"input_tokens": 30, "output_tokens": 9, "total_tokens": 39,
                      "input_tokens_details": {"cached_tokens": 10},
                      "output_tokens_details": {"reasoning_tokens": 4}}})
    if body.get("stream"):
        chunks = [{"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini-2024-07-18",
                   "choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]}
                  for t in ("Hello", " there", " friend")]
        if (body.get("stream_options") or {}).get("include_usage"):
            chunks.append({"id": "c", "object": "chat.completion.chunk", "created": 1,
                           "model": "gpt-4o-mini-2024-07-18", "choices": [],
                           "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}})
        data = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, content=data.encode(), headers={"content-type": "text/event-stream"})
    if body.get("model") == "boom":
        return httpx.Response(400, json={"error": {"message": "bad model", "type": "invalid_request_error"}})
    return httpx.Response(200, json={
        "id": "x", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini-2024-07-18",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
                  "prompt_tokens_details": {"cached_tokens": 100}}})


def _openai(base_url=None, cls=None):
    import openai
    cls = cls or openai.OpenAI
    return cls(api_key="sk-test", base_url=base_url, max_retries=0,
               http_client=httpx.Client(transport=httpx.MockTransport(_openai_handler)))


def test_openai_chat_records_tokens_and_line(events):
    r = _openai().chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}])  # @oa1
    assert r.choices[0].message.content == "hi"              # app result untouched
    assert len(events) == 1
    e = events[0]
    assert (e["provider"], e["model"], e["op"]) == ("openai", "gpt-4o-mini-2024-07-18", "chat")
    assert (e["input_tokens"], e["output_tokens"], e["cached_input_tokens"]) == (120, 30, 100)
    assert (e["path"], e["line"], e["func"]) == (HERE, _lineno("oa1"), "test_openai_chat_records_tokens_and_line")
    assert e["sdk"] == "openai.chat.completions.create" and e["ok"] is True and "ms" in e


def test_openai_compatible_providers_by_base_url(events):
    for url, want in (("https://openrouter.ai/api/v1", "openrouter"),
                      ("https://dashscope.aliyuncs.com/compatible-mode/v1", "dashscope"),
                      ("https://api.deepseek.com/v1", "deepseek"),
                      ("https://api.groq.com/openai/v1", "groq"),
                      ("http://localhost:11434/v1", "ollama")):
        _openai(url).chat.completions.create(model="m", messages=[])
        assert events[-1]["provider"] == want, url


def test_azure_openai(events):
    import openai
    c = openai.AzureOpenAI(api_key="k", api_version="2024-06-01", azure_endpoint="https://x.openai.azure.com",
                           max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(_openai_handler)))
    c.chat.completions.create(model="my-deployment", messages=[])
    assert events[-1]["provider"] == "azure"


def test_openai_stream_with_usage(events):
    s = _openai().chat.completions.create(model="gpt-4o-mini", messages=[], stream=True,  # @oas
                                          stream_options={"include_usage": True})
    assert events == []                                       # recorded only when the stream is consumed
    text = "".join(c.choices[0].delta.content for c in s if c.choices)
    assert text == "Hello there friend"
    assert len(events) == 1
    e = events[0]
    assert e["stream"] is True and (e["input_tokens"], e["output_tokens"]) == (11, 3)
    assert "output_tokens_estimated" not in e and e["line"] == _lineno("oas")


def test_openai_stream_without_usage_estimates_output(events):
    with _openai().chat.completions.create(model="gpt-4o-mini", messages=[], stream=True) as s:
        for _ in s:
            pass
    e = events[0]
    assert e["output_tokens"] == 5 and e["output_tokens_estimated"] is True   # 18 chars / 4
    assert "input_tokens" not in e                                           # messages=[]: nothing to count


def test_stream_without_usage_estimates_input_from_request(events):
    """Found in the sandbox: OpenAI streams report no usage by default, so input was missing."""
    for _ in _openai().chat.completions.create(model="m", stream=True,
                                               messages=[{"role": "user", "content": "x" * 400}]):
        pass
    e = events[0]
    assert e["input_tokens"] == 104 and e["input_tokens_estimated"] is True     # 400/4 + 4 per message
    assert "x" * 20 not in json.dumps(e)


def test_openai_stream_abandoned_is_recorded_once(events):
    s = _openai().chat.completions.create(model="gpt-4o-mini", messages=[], stream=True)
    next(iter(s))
    s.close()
    del s
    assert len(events) == 1 and events[0]["stream_incomplete"] is True


def test_abandoned_stream_keeps_tags_of_the_call(events):
    """Found in the sandbox: a stream dropped inside tag() was recorded at garbage collection,
    after the tag block had exited, and lost its tags."""
    import gc

    def peek():
        with trainlab.tag(feature="peek"):
            s = _openai().chat.completions.create(model="m", messages=[], stream=True)
            for c in s:
                return c
    peek()
    gc.collect()
    assert events[0]["tags"] == {"feature": "peek"} and events[0]["stream_incomplete"] is True


def test_litellm_style_cache_write_tokens():
    """Found in the sandbox: LiteLLM reports Anthropic cache writes as cache_creation_input_tokens."""
    from trainlab_optimize.runtime.core import CallEvent
    from trainlab_optimize.runtime.patches import _openai_chat_usage
    ev = CallEvent(provider="anthropic", model="m", op="chat", sdk="litellm")
    _openai_chat_usage({"prompt_tokens": 900, "completion_tokens": 10, "cache_creation_input_tokens": 300,
                        "prompt_tokens_details": {"cached_tokens": 200}}, ev)
    assert (ev.input_tokens, ev.cached_input_tokens, ev.cache_write_tokens) == (900, 200, 300)


def test_jsonl_is_write_through(tmp_path):
    """Found in the sandbox: gunicorn workers exit without atexit, losing buffered calls."""
    trainlab.instrument(path=tmp_path)
    try:
        _openai().chat.completions.create(model="m", messages=[])
        assert len(next(tmp_path.glob("events-*.jsonl")).read_text().splitlines()) == 1   # no flush() needed
    finally:
        trainlab.uninstrument()


def test_openai_responses_and_embeddings(events):
    c = _openai()
    r = c.responses.create(model="gpt-5-mini", input="x")
    assert r.output_text == "hi"
    c.embeddings.create(model="text-embedding-3-small", input="x")
    a, b = events
    assert (a["input_tokens"], a["output_tokens"], a["cached_input_tokens"], a["reasoning_tokens"]) == (30, 9, 10, 4)
    assert a["sdk"] == "openai.responses.create"
    assert (b["op"], b["input_tokens"], b["model"]) == ("embedding", 7, "text-embedding-3-small")


def test_openai_errors_pass_through_and_are_recorded(events):
    import openai
    with pytest.raises(openai.BadRequestError):
        _openai().chat.completions.create(model="boom", messages=[])
    e = events[0]
    assert e["ok"] is False and e["error"] == "BadRequestError" and "bad model" not in json.dumps(e)


def test_openai_async_and_async_stream(events):
    import openai

    async def run():
        c = openai.AsyncOpenAI(api_key="k", max_retries=0,
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(_openai_handler)))
        await c.chat.completions.create(model="gpt-4o-mini", messages=[])  # @oaa
        s = await c.chat.completions.create(model="gpt-4o-mini", messages=[], stream=True,
                                            stream_options={"include_usage": True})
        out = [ch async for ch in s]
        return out

    chunks = asyncio.run(run())
    assert len(chunks) == 4
    a, b = events
    assert a["input_tokens"] == 120 and a["line"] == _lineno("oaa") and a["func"] == "run"
    assert b["stream"] is True and b["output_tokens"] == 3


def test_no_prompt_text_is_stored(events):
    _openai().chat.completions.create(model="gpt-4o-mini",
                                      messages=[{"role": "user", "content": "SECRET-PROMPT-123"}])
    assert "SECRET" not in json.dumps(events) and "sk-test" not in json.dumps(events)


# ------------------------------------------------------------------ tags, sampling, safety
def test_tags_context_and_decorator(events):
    @trainlab.tag(feature="summarise")
    def summarise():
        return _openai().chat.completions.create(model="m", messages=[])

    with trainlab.tag(team="billing"):
        summarise()
    _openai().chat.completions.create(model="m", messages=[])
    assert events[0]["tags"] == {"team": "billing", "feature": "summarise"}
    assert "tags" not in events[1]


def test_async_tags_are_isolated_per_task(events):
    import openai

    async def one(name):
        c = openai.AsyncOpenAI(api_key="k", max_retries=0,
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(_openai_handler)))
        with trainlab.tag(feature=name):
            await asyncio.sleep(0)
            await c.chat.completions.create(model="m", messages=[])

    async def run():
        await asyncio.gather(*(one(f"f{i}") for i in range(5)))

    asyncio.run(run())
    assert sorted(e["tags"]["feature"] for e in events) == [f"f{i}" for i in range(5)]


def test_sampling_sets_weight():
    got = []
    trainlab.instrument(path=False, sinks=[_ListSink(got)], sample_rate=0.25)
    try:
        for _ in range(400):
            _openai().chat.completions.create(model="m", messages=[])
    finally:
        trainlab.uninstrument()
    assert 50 < len(got) < 150 and all(e["weight"] == 4.0 for e in got)


def test_failing_sink_never_breaks_app():
    class Bad:
        def write(self, ev):
            raise RuntimeError("disk full")

        def flush(self):
            raise RuntimeError("disk full")

    trainlab.instrument(path=False, sinks=[Bad()])
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for _ in range(3):
                r = _openai().chat.completions.create(model="m", messages=[])
                assert r.choices[0].message.content == "hi"
        assert sum("disabled" in str(x.message) for x in w) == 1
    finally:
        trainlab.uninstrument()


def test_broken_usage_never_breaks_app(events, monkeypatch):
    from trainlab_optimize.runtime import patches
    monkeypatch.setattr(patches, "_openai_chat_usage", lambda u, ev: 1 / 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _openai().chat.completions.create(model="m", messages=[])
    assert r.choices[0].message.content == "hi"


def test_uninstrument_restores_and_kill_switch(monkeypatch):
    import openai.resources.chat.completions as cc
    orig = cc.Completions.create
    trainlab.instrument(path=False)
    assert cc.Completions.create is not orig
    trainlab.instrument(path=False)                            # idempotent: not wrapped twice
    assert cc.Completions.create.__trainlab_original__ is orig
    trainlab.uninstrument()
    assert cc.Completions.create is orig
    monkeypatch.setenv("TRAINLAB_INSTRUMENT", "0")
    assert trainlab.instrument() is None and cc.Completions.create is orig


def test_jsonl_sink_writes_file(tmp_path):
    trainlab.instrument(path=tmp_path / ".trainlab" / "traces", service="api", tags={"env": "test"})
    try:
        _openai().chat.completions.create(model="m", messages=[])
        trainlab.flush()
    finally:
        trainlab.uninstrument()
    files = list((tmp_path / ".trainlab" / "traces").glob("events-*.jsonl"))
    assert len(files) == 1
    e = json.loads(files[0].read_text().splitlines()[0])
    assert e["service"] == "api" and e["tags"] == {"env": "test"} and e["v"] == 1
    assert (tmp_path / ".trainlab" / ".gitignore").read_text() == "*\n"


# ------------------------------------------------------------------ Anthropic
def _anthropic_handler(request):
    body = json.loads(request.content)
    usage = {"input_tokens": 50, "output_tokens": 20, "cache_read_input_tokens": 1000,
             "cache_creation_input_tokens": 200}
    if body.get("stream"):
        evs = [("message_start", {"type": "message_start", "message": {
                    "id": "m", "type": "message", "role": "assistant", "model": body["model"], "content": [],
                    "stop_reason": None, "stop_sequence": None, "usage": {**usage, "output_tokens": 1}}}),
               ("content_block_start", {"type": "content_block_start", "index": 0,
                                        "content_block": {"type": "text", "text": ""}}),
               ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                        "delta": {"type": "text_delta", "text": "Hello"}}),
               ("content_block_stop", {"type": "content_block_stop", "index": 0}),
               ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn",
                                                                      "stop_sequence": None},
                                  "usage": {"output_tokens": 20}}),
               ("message_stop", {"type": "message_stop"})]
        data = "".join(f"event: {n}\ndata: {json.dumps(d)}\n\n" for n, d in evs)
        return httpx.Response(200, content=data.encode(), headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json={"id": "m", "type": "message", "role": "assistant", "model": body["model"],
                                     "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
                                     "stop_sequence": None, "usage": usage})


def _anthropic(cls=None):
    import anthropic
    cls = cls or anthropic.Anthropic
    try:                                   # newer anthropic SDKs are built on httpx2
        cls(api_key="k", http_client=httpx.Client())
        hx = httpx
    except TypeError:
        import httpx2 as hx
    http = hx.AsyncClient if cls.__name__.startswith("Async") else hx.Client

    def handler(request):
        r = _anthropic_handler(request)
        return hx.Response(r.status_code, content=r.content, headers=dict(r.headers))
    return cls(api_key="k", max_retries=0, http_client=http(transport=hx.MockTransport(handler)))


def test_anthropic_create_counts_cache_in_input(events):
    r = _anthropic().messages.create(model="claude-sonnet-4-5", max_tokens=10, messages=[])  # @an1
    assert r.content[0].text == "hi"
    e = events[0]
    assert (e["provider"], e["model"], e["line"]) == ("anthropic", "claude-sonnet-4-5", _lineno("an1"))
    assert (e["input_tokens"], e["cached_input_tokens"], e["cache_write_tokens"], e["output_tokens"]) == \
        (1250, 1000, 200, 20)


def test_anthropic_stream_create_and_stream_manager(events):
    c = _anthropic()
    for _ in c.messages.create(model="claude-haiku-4-5", max_tokens=10, messages=[], stream=True):
        pass
    with c.messages.stream(model="claude-haiku-4-5", max_tokens=10, messages=[]) as s:  # @ans
        assert s.get_final_text() == "Hello"
    a, b = events
    for e in (a, b):
        assert e["stream"] is True and (e["input_tokens"], e["output_tokens"]) == (1250, 20)
    assert b["sdk"] == "anthropic.messages.stream" and b["line"] == _lineno("ans")


def test_anthropic_async(events):
    import anthropic

    async def run():
        c = _anthropic(anthropic.AsyncAnthropic)
        await c.messages.create(model="claude-haiku-4-5", max_tokens=10, messages=[])
        async with c.messages.stream(model="claude-haiku-4-5", max_tokens=10, messages=[]) as s:
            await s.get_final_text()

    asyncio.run(run())
    assert [e["output_tokens"] for e in events] == [20, 20] and events[1]["stream"] is True


# ------------------------------------------------------------------ LiteLLM
def test_litellm_records_once_even_when_it_calls_openai_sdk(events):
    litellm = pytest.importorskip("litellm")
    r = litellm.completion(model="openai/gpt-4o-mini", messages=[{"role": "user", "content": "x"}],  # @ll
                           client=_openai())
    assert r.choices[0].message.content == "hi"
    assert len(events) == 1, events                          # not double counted via the OpenAI hook
    e = events[0]
    assert e["sdk"] == "litellm.completion" and e["provider"] == "openai" and e["input_tokens"] == 120
    assert e["line"] == _lineno("ll")


def test_litellm_mock_response_and_prefix(events):
    litellm = pytest.importorskip("litellm")
    litellm.completion(model="dashscope/qwen-max", messages=[{"role": "user", "content": "x"}], mock_response="ok")
    assert events[0]["provider"] == "dashscope"


# ------------------------------------------------------------------ Bedrock via botocore
def test_bedrock_converse_and_other_aws_calls_untouched(events):
    boto3 = pytest.importorskip("boto3")
    from botocore.stub import Stubber
    br = boto3.client("bedrock-runtime", region_name="us-east-1", aws_access_key_id="a", aws_secret_access_key="b")
    with Stubber(br) as st:
        st.add_response("converse", {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}}, "stopReason": "end_turn",
            "usage": {"inputTokens": 40, "outputTokens": 8, "totalTokens": 48, "cacheReadInputTokens": 10},
            "metrics": {"latencyMs": 5}})
        r = br.converse(modelId="anthropic.claude-3-5-haiku-20241022-v1:0",  # @br
                        messages=[{"role": "user", "content": [{"text": "x"}]}])
    assert r["output"]["message"]["content"][0]["text"] == "hi"
    s3 = boto3.client("s3", region_name="us-east-1", aws_access_key_id="a", aws_secret_access_key="b")
    with Stubber(s3) as st:
        st.add_response("list_buckets", {"Buckets": []})
        s3.list_buckets()
    assert len(events) == 1
    e = events[0]
    assert (e["provider"], e["model"], e["input_tokens"], e["output_tokens"], e["cached_input_tokens"]) == \
        ("bedrock", "anthropic.claude-3-5-haiku-20241022-v1:0", 50, 8, 10)
    assert e["line"] == _lineno("br")


# ------------------------------------------------------------------ Google Gen AI
def _genai_handler(request):
    body = {"candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
            "modelVersion": "gemini-2.5-flash",
            "usageMetadata": {"promptTokenCount": 60, "candidatesTokenCount": 12, "cachedContentTokenCount": 20,
                              "thoughtsTokenCount": 5, "totalTokenCount": 77}}
    if "streamGenerateContent" in request.url.path:
        return httpx.Response(200, content=f"data: {json.dumps(body)}\n\n".encode(),
                              headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json=body)


def test_gemini_generate_and_stream(events):
    genai = pytest.importorskip("google.genai")
    from google.genai import types
    c = genai.Client(api_key="k", http_options=types.HttpOptions(
        httpx_client=httpx.Client(transport=httpx.MockTransport(_genai_handler))))
    r = c.models.generate_content(model="gemini-2.5-flash", contents="x")  # @gm
    assert r.text == "hi"
    for _ in c.models.generate_content_stream(model="gemini-2.5-flash", contents="x"):
        pass
    a, b = events
    assert (a["provider"], a["model"], a["input_tokens"], a["output_tokens"], a["cached_input_tokens"],
            a["reasoning_tokens"]) == ("gemini", "gemini-2.5-flash", 60, 12, 20, 5)
    assert a["line"] == _lineno("gm")
    assert b["stream"] is True and b["output_tokens"] == 12


# ------------------------------------------------------------------ overhead
def test_overhead_under_one_ms_per_call():
    import time

    class FakeResource:
        _client = None

    def fake(self, **kw):
        return {"model": "m", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    from trainlab_optimize.runtime.core import wrap_call
    from trainlab_optimize.runtime import patches
    got = []
    trainlab.instrument(path=False, sinks=[_ListSink(got)])
    try:
        from trainlab_optimize import runtime
        w = wrap_call(runtime._ref, fake, lambda a, k: ("openai", k.get("model"), "chat"),
                      lambda r, ev: patches._openai_chat_usage(r["usage"], ev), sdk_name="x")
        n = 3000
        t0 = time.perf_counter()
        for _ in range(n):
            fake(FakeResource(), model="m")
        base = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(n):
            w(FakeResource(), model="m")
        inst = time.perf_counter() - t0
    finally:
        trainlab.uninstrument()
    per_call_ms = (inst - base) / n * 1000
    assert len(got) == n
    assert per_call_ms < 1.0, per_call_ms


# ------------------------------------------------------------------ lazy patching
def test_instrument_is_fast_and_patches_on_import(tmp_path):
    import subprocess
    import sys
    code = """
import sys, time
t0 = time.perf_counter()
import trainlab_optimize as t
t.instrument(path=False)
took = time.perf_counter() - t0
assert 'litellm' not in sys.modules and 'openai' not in sys.modules, 'instrument() imported SDKs'
import openai.resources.chat.completions as cc
assert hasattr(cc.Completions.create, '__trainlab_original__')
assert 'openai' in t.status()['patched']
print(round(took, 3))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert float(out.stdout.strip()) < 1.0


# ------------------------------------------------------------------ OTel span processor
def test_code_location_span_processor():
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from trainlab_optimize.runtime.otel import CodeLocationSpanProcessor
    exp = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(CodeLocationSpanProcessor(project_root=_root()))
    tp.add_span_processor(SimpleSpanProcessor(exp))
    tracer = tp.get_tracer("x")
    with trainlab.tag(feature="kb"):
        with tracer.start_as_current_span("chat gpt-4o", attributes={"gen_ai.request.model": "gpt-4o"}):  # @otel
            pass
    with tracer.start_as_current_span("db.query"):
        pass
    chat, db = exp.get_finished_spans()
    assert chat.attributes["code.file.path"] == HERE and chat.attributes["code.line.number"] == _lineno("otel")
    assert chat.attributes["trainlab.tag.feature"] == "kb"
    assert "code.file.path" not in db.attributes


def test_otel_sink_round_trips_through_importer(tmp_path):
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from trainlab_optimize.runtime import OTelSink
    exp = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exp))
    sink = OTelSink()
    sink.tracer = tp.get_tracer("t")
    trainlab.instrument(path=False, sinks=[sink], project_root=_root())
    try:
        with trainlab.tag(feature="kb"):
            _openai().chat.completions.create(model="gpt-4o-mini", messages=[])
    finally:
        trainlab.uninstrument()
    (span,) = exp.get_finished_spans()
    doc = {"resourceSpans": [{"resource": {"attributes": []}, "scopeSpans": [{"spans": [{
        "startTimeUnixNano": span.start_time,
        "attributes": [{"key": k, "value": {"intValue" if isinstance(v, int) else "stringValue": v}}
                       for k, v in span.attributes.items()]}]}]}]}
    from trainlab_optimize.importers.traces import parse_otel
    (r,) = parse_otel(doc)
    assert (r.model, r.input_tokens, r.output_tokens, r.code_path, r.tags) == \
        ("gpt-4o-mini-2024-07-18", 120, 30, HERE, {"feature": "kb"})


# ------------------------------------------------------------------ importer + report
def test_events_importer_weights_and_tags(tmp_path):
    from trainlab_optimize.importers.auto import read_any
    d = tmp_path / "traces"
    d.mkdir()
    lines = [
        {"v": 1, "provider": "openai", "model": "gpt-4o", "op": "chat", "sdk": "s", "ts": "2026-10-01T10:00:00Z",
         "input_tokens": 100, "output_tokens": 10, "path": "app/a.py", "line": 5, "tags": {"feature": "x"},
         "weight": 4.0, "ok": True},
        {"v": 1, "provider": "openai", "model": "gpt-4o", "op": "chat", "sdk": "s", "ts": "2026-10-01T11:00:00Z",
         "input_tokens": 50, "output_tokens": 5, "path": "app/a.py", "line": 5, "tags": {"feature": "x"},
         "weight": 4.0, "ok": True},
        {"v": 1, "provider": "openai", "model": "gpt-4o", "op": "chat", "sdk": "s", "ts": "2026-10-01T12:00:00Z",
         "path": "app/a.py", "line": 5, "tags": {"feature": "x"}, "ok": False, "error": "RateLimitError"},
    ]
    (d / "events-20261001-1.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n{broken")
    recs, kind = read_any(d)
    assert kind == "trainlab" and len(recs) == 1
    r = recs[0]
    assert (r.requests, r.input_tokens, r.output_tokens, r.code_path, r.code_line, r.tags) == \
        (8, 600, 60, "app/a.py", 5, {"feature": "x"})
    assert r.source.startswith("trainlab:")


def test_demo_end_to_end(tmp_path):
    """Run the demo app under instrument(), then scan + import + report with the CLI."""
    import shutil
    import subprocess
    import sys
    src = _root() + "/examples/instrument_demo"
    demo = tmp_path / "demo"
    shutil.copytree(src, demo, ignore=shutil.ignore_patterns(".trainlab", "__pycache__"))
    run = subprocess.run([sys.executable, "run_demo.py"], cwd=demo, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    tl = [sys.executable, "-m", "trainlab_optimize.cli"]
    for cmd in (["scan", "app"], ["import", ".trainlab/traces"], ["report", "--format", "json,md"]):
        out = subprocess.run(tl + cmd, cwd=demo, capture_output=True, text=True, timeout=300)
        assert out.returncode == 0, (cmd, out.stdout, out.stderr)
    rep = json.loads(next((demo / ".trainlab" / "report").glob("*.json")).read_text())
    spend = rep["spend"]
    assert set(spend["by_tag"]) >= {"feature=support-bot", "feature=invoice-extract", "feature=search",
                                    "team=cx", "env=demo"}
    assert abs(sum(v for k, v in spend["by_tag"].items() if k.startswith("feature=")) - spend["monthly_total_usd"]) < 0.05
    methods = {a["method"] for a in rep["attribution"]}
    assert methods == {"exact_trace"}, methods
    assert rep["coverage"]["exact_share"] > 0.99
    assert spend["by_project"] and "acme-api" in spend["by_project"]


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_forked_child_does_not_rewrite_parent_buffer(tmp_path):
    import os
    if not hasattr(os, "fork"):
        pytest.skip("no fork")
    trainlab.instrument(path=tmp_path, project_root=_root())
    try:
        _openai().chat.completions.create(model="m", messages=[])     # written by the parent
        pid = os.fork()
        if pid == 0:
            try:
                with trainlab.tag(who="child"):
                    _openai().chat.completions.create(model="m", messages=[])
                trainlab.flush()
            finally:
                os._exit(0)
        os.waitpid(pid, 0)
        trainlab.flush()
    finally:
        trainlab.uninstrument()
    who = [json.loads(ln).get("tags", {}).get("who", "parent")
           for f in tmp_path.glob("events-*.jsonl") for ln in f.read_text().splitlines()]
    assert sorted(who) == ["child", "parent"]          # the parent's call is written once, by the parent


def test_openai_with_raw_and_streaming_response(events):
    c = _openai()
    raw = c.chat.completions.with_raw_response.create(model="m", messages=[])
    assert raw.parse().choices[0].message.content == "hi"
    with c.chat.completions.with_streaming_response.create(model="m", messages=[]) as resp:
        assert json.loads(resp.read())["usage"]["prompt_tokens"] == 120   # body untouched by trainlab
    a, b = events                                   # one event each, never a second from the HTTP layer
    assert a["input_tokens"] == 120 and a["ok"] is True
    assert b["input_tokens"] == 120 and b["output_tokens"] == 30 and b["sdk"] == "openai.chat.completions.create"


def test_gemini_async_stream(events):
    genai = pytest.importorskip("google.genai")
    from google.genai import types

    async def run():
        c = genai.Client(api_key="k", http_options=types.HttpOptions(
            httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(_genai_handler))))
        await c.aio.models.generate_content(model="gemini-2.5-flash", contents="x")
        async for _ in await c.aio.models.generate_content_stream(model="gemini-2.5-flash", contents="x"):
            pass

    asyncio.run(run())
    a, b = events
    assert a["input_tokens"] == 60 and b["stream"] is True and b["output_tokens"] == 12


def test_bedrock_converse_stream(events):
    boto3 = pytest.importorskip("boto3")
    from botocore.client import BaseClient
    br = boto3.client("bedrock-runtime", region_name="us-east-1", aws_access_key_id="a", aws_secret_access_key="b")
    stream_events = [{"messageStart": {"role": "assistant"}},
                     {"contentBlockDelta": {"delta": {"text": "hi"}, "contentBlockIndex": 0}},
                     {"messageStop": {"stopReason": "end_turn"}},
                     {"metadata": {"usage": {"inputTokens": 33, "outputTokens": 4, "totalTokens": 37},
                                   "metrics": {"latencyMs": 3}}}]
    # botocore's Stubber can't produce event streams, so wrap a fake _make_api_call with the same hook.
    orig = BaseClient._make_api_call.__trainlab_original__

    def fake(self, op, params):
        assert op == "ConverseStream"
        return {"stream": iter(stream_events), "ResponseMetadata": {}}

    import trainlab_optimize.runtime.patches as P
    from trainlab_optimize.runtime.core import wrap_call
    from trainlab_optimize import runtime
    spec = [s for s in P._bedrock_specs() if s[2]["sdk_name"] == "botocore.bedrock-runtime"][0][2]
    w = wrap_call(runtime._ref, fake, spec["describe"], spec["extract"], is_stream=spec["is_stream"],
                  wrap_stream=spec["wrap_stream"], sdk_name=spec["sdk_name"])
    r = w(br, "ConverseStream", {"modelId": "amazon.nova-lite-v1:0"})
    texts = [e["contentBlockDelta"]["delta"]["text"] for e in r["stream"] if "contentBlockDelta" in e]
    assert texts == ["hi"] and orig is not None
    (e,) = events
    assert (e["model"], e["input_tokens"], e["output_tokens"], e["stream"]) == ("amazon.nova-lite-v1:0", 33, 4, True)


# ------------------------------------------------------------------ HTTP layer (any client, any SDK)
def _fake_server(handler_fn):
    """A real local HTTP server (the HTTP layer patches the client transports themselves)."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
            code, ctype, payload = handler_fn(self.path, body)
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _formats(path, body):
    if path == "/v1/chat/completions":
        if body.get("stream"):
            chunks = [{"choices": [{"index": 0, "delta": {"content": "hey"}}], "model": body["model"]},
                      {"choices": [], "model": body["model"], "usage": {"prompt_tokens": 9, "completion_tokens": 2}}]
            return 200, "text/event-stream", "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n"
        return 200, "application/json", {"model": body["model"], "choices": [{"message": {"content": "x"}}],
                                         "usage": {"prompt_tokens": 11, "completion_tokens": 3}}
    if path == "/v2/chat":
        return 200, "application/json", {"message": {"content": [{"text": "x"}]}, "finish_reason": "COMPLETE",
                                         "usage": {"billed_units": {"input_tokens": 21, "output_tokens": 4}}}
    if path == "/api/chat":
        lines = [{"model": body["model"], "message": {"content": "x"}, "done": False},
                 {"model": body["model"], "message": {"content": ""}, "done": True, "prompt_eval_count": 31,
                  "eval_count": 5}]
        return 200, "application/x-ndjson", "\n".join(json.dumps(x) for x in lines).encode() + b"\n"
    if path.endswith("/text-generation/generation"):
        return 200, "application/json", {"output": {"text": "x"}, "usage": {"input_tokens": 41, "output_tokens": 6}}
    if path == "/app/api/chat":                         # an app's own endpoint, not an LLM
        return 200, "application/json", {"reply": "hello", "user": 1}
    if path == "/v1/messages":
        return 429, "application/json", {"type": "error", "error": {"type": "rate_limit_error"}}
    return 404, "application/json", {}


@pytest.fixture()
def server():
    srv, base = _fake_server(_formats)
    yield base
    srv.shutdown()


def test_http_layer_records_any_client_and_format(events, server):
    import requests
    httpx.post(f"{server}/v1/chat/completions", json={"model": "mistral-large", "messages": []})  # @h1
    requests.post(f"{server}/v2/chat", json={"model": "command-r", "messages": []})
    for _ in requests.post(f"{server}/api/chat", json={"model": "llama3.2"}, stream=True).iter_lines():
        pass
    httpx.post(f"{server}/api/v1/services/aigc/text-generation/generation", json={"model": "qwen-max"})
    with httpx.stream("POST", f"{server}/v1/chat/completions", json={"model": "m", "stream": True}) as r:
        for _ in r.iter_lines():
            pass
    got = [(e["sdk"], e["model"], e["input_tokens"], e["output_tokens"]) for e in events]
    assert got == [("http:httpx", "mistral-large", 11, 3), ("http:requests", "command-r", 21, 4),
                   ("http:requests", "llama3.2", 31, 5), ("http:httpx", "qwen-max", 41, 6),
                   ("http:httpx", "m", 9, 2)]
    assert events[0]["line"] == _lineno("h1") and events[0]["path"] == HERE
    assert events[2]["stream"] is True and events[4]["stream"] is True


def test_http_layer_ignores_non_llm_and_records_errors(events, server):
    import requests
    requests.post(f"{server}/app/api/chat", json={"msg": "hi"})       # app route, plain JSON: not an LLM call
    httpx.post(f"{server}/v1/messages", json={"model": "claude"})      # 429
    assert len(events) == 1 and events[0]["ok"] is False and events[0]["error"] == "HTTP 429"


def test_http_layer_aiohttp(events, server):
    aiohttp = pytest.importorskip("aiohttp")

    async def run():
        async with aiohttp.ClientSession() as s:
            async with s.post(f"{server}/v1/chat/completions", json={"model": "gpt-4o", "messages": []}) as r:
                await r.json()
            async with s.post(f"{server}/api/chat", json={"model": "llama3.2"}) as r:
                async for _ in r.content:
                    pass

    asyncio.run(run())
    assert [(e["model"], e["input_tokens"]) for e in events] == [("gpt-4o", 11), ("llama3.2", 31)]


def test_sdk_call_is_not_recorded_twice_by_http_layer(events, server):
    import openai
    c = openai.OpenAI(base_url=f"{server}/v1", api_key="k", max_retries=0)
    c.chat.completions.create(model="m", messages=[])
    for _ in c.chat.completions.create(model="m", messages=[], stream=True):
        pass
    assert [e["sdk"] for e in events] == ["openai.chat.completions.create"] * 2
    assert events[1]["input_tokens"] == 9          # usage only on the stream's last chunk


def test_threads_and_tasks_keep_app_line_and_tags(events):
    """Found in the framework run: LangChain batch(), CrewAI, Strands run LLM calls on pool
    threads, Google ADK in its own asyncio tasks - stacks with no app frame at all. The line
    that handed the work over must be used, with its tags."""
    import concurrent.futures as cf
    import functools
    import threading

    import openai
    c = _openai()
    create = functools.partial(c.chat.completions.create, model="m", messages=[])   # library code only
    with trainlab.tag(feature="pool"):
        with cf.ThreadPoolExecutor(2) as ex:
            list(ex.map(lambda f: f(), [create]))  # @pool
            ex.submit(create).result()  # @submit
        t = threading.Thread(target=create)
        t.start()  # @thread
        t.join()
    assert [e["line"] for e in events] == [_lineno("pool"), _lineno("submit"), _lineno("thread")]
    assert all(e["tags"] == {"feature": "pool"} and e["path"] == HERE for e in events)

    ac = openai.AsyncOpenAI(api_key="k", max_retries=0,
                            http_client=httpx.AsyncClient(transport=httpx.MockTransport(_openai_handler)))

    async def main():
        with trainlab.tag(feature="task"):
            await asyncio.gather(ac.chat.completions.create(model="m", messages=[]))  # @task
    asyncio.run(main())
    assert (events[3]["line"], events[3]["tags"]) == (_lineno("task"), {"feature": "task"})


def test_sampled_out_sdk_call_is_not_recorded_by_http_layer(server):
    import openai
    got = []
    trainlab.instrument(path=False, sinks=[_ListSink(got)], sample_rate=0.000001)
    try:
        c = openai.OpenAI(base_url=f"{server}/v1", api_key="k", max_retries=0)
        for _ in range(30):
            c.chat.completions.create(model="m", messages=[])
            for _ in c.chat.completions.create(model="m", messages=[], stream=True):
                pass
    finally:
        trainlab.uninstrument()
    assert got == []
