"""Profile 3: experiments. Gates, samples, clients, runner (budget, resume, early stop, judge),
capture, report integration and the CLI, without network access."""
import csv
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from trainlab_optimize.experiment import gates
from trainlab_optimize.experiment.clients import Result, build_request, parse_response, resolve
from trainlab_optimize.experiment.runner import Config, plan, run
from trainlab_optimize.experiment.samples import load_samples, make_sample
from trainlab_optimize.pricing import PriceCatalog

CAT = PriceCatalog.load()


# ------------------------------------------------------------------ gates
def test_wilson_and_verdicts():
    lo, hi = gates.wilson(100, 100)
    assert 0.97 < lo < 0.975 and hi == 1.0
    assert gates.verdict(100, 100, 0.95) == ("PASS", None)
    assert gates.verdict(60, 100, 0.95)[0] == "FAIL"
    v, need = gates.verdict(98, 100, 0.95)
    assert v == "INCONCLUSIVE" and need and need > 100
    assert gates.verdict(19, 19, 0.95)[0] == "INCONCLUSIVE"        # too few samples to pass


def test_label_json_and_task_detection():
    assert gates.norm_label("Label: Billing.") == gates.norm_label("**billing**") == "billing"
    assert gates.score("classification", "billing", "Billing.")["pass"] is True
    ref = '{"vendor": "Acme", "total": 12.5, "items": [1, 2]}'
    assert gates.score("extraction", ref, '```json\n{"total": 12.50, "vendor": "acme", "items": [1,2]}\n```')["pass"]
    bad = gates.score("extraction", ref, '{"vendor": "Acme", "total": 13, "items": [1, 2]}')
    assert bad["pass"] is False and 0 < bad["field_score"] < 1
    assert gates.score("extraction", ref, "vendor: Acme")["reason"] == "invalid JSON"
    assert gates.score("generation", "a b c", "a b c")["pass"] is None             # needs a judge
    assert gates.score("classification", "x", "I'm sorry, I can't help")["pass"] is False
    assert gates.detect_task(['{"a": 1}'] * 10) == "extraction"
    assert gates.detect_task(["billing", "bug", "account"] * 10) == "classification"
    assert gates.detect_task(["A long free text answer that goes on for a while here."] * 10) == "generation"
    assert gates.judge_choice("B") == "B" and gates.judge_choice("tie.") == "TIE" and gates.judge_choice("?") is None


# ------------------------------------------------------------------ samples
def test_loaders(tmp_path):
    (tmp_path / "cap.jsonl").write_text("\n".join(json.dumps(x) for x in [
        {"site": "app/x.py:4", "model": "gpt-4o", "messages": [{"role": "system", "content": "S"},
                                                               {"role": "user", "content": "q1"}],
         "params": {"max_tokens": 5}, "output": "a1"},
        {"messages": [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]},   # chat eval file
        {"prompt": "q3", "completion": "a3"},
        {"input": "q4", "expected": "a4"},
        {"type": "SPAN", "input": "skip"},                                                          # Langfuse non-generation
        {"messages": [{"role": "user", "content": "q1"}, {"role": "system", "content": "x"}]},
    ]))
    s = load_samples(tmp_path / "cap.jsonl")
    by = {x.messages[-1]["content"]: x for x in s}
    assert by["q1"].reference == "a1" and by["q1"].params == {"max_tokens": 5} and by["q1"].site == "app/x.py:4"
    assert by["q2"].reference == "a2" and by["q2"].messages == [{"role": "user", "content": "q2"}]
    assert by["q3"].reference == "a3" and by["q4"].expected == "a4"
    with open(tmp_path / "d.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Question", "Answer"])
        w.writerow(["mail me at bob@example.com", "yes"])
    c = load_samples(tmp_path / "d.csv", do_redact=True, system="Be brief")
    assert c[0].messages == [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "mail me at <EMAIL>"}]
    assert c[0].expected == "yes"
    otel = {"resourceSpans": [{"resource": {"attributes": []}, "scopeSpans": [{"spans": [{"attributes": [
        {"key": "gen_ai.request.model", "value": {"stringValue": "gpt-4o"}},
        {"key": "gen_ai.prompt.0.role", "value": {"stringValue": "user"}},
        {"key": "gen_ai.prompt.0.content", "value": {"stringValue": "hello"}},
        {"key": "gen_ai.completion.0.content", "value": {"stringValue": "hi"}},
        {"key": "code.file.path", "value": {"stringValue": "a.py"}},
        {"key": "code.line.number", "value": {"intValue": 3}}]}]}]}]}
    (tmp_path / "spans.json").write_text(json.dumps(otel))
    o = load_samples(tmp_path / "spans.json")
    assert o[0].messages[-1]["content"] == "hello" and o[0].reference == "hi" and o[0].site == "a.py:3"


# ------------------------------------------------------------------ clients
def test_routes_and_request_formats(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k2")
    monkeypatch.setenv("GEMINI_API_KEY", "k3")
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    url, hdr, body = build_request(resolve("gpt-5-mini"), msgs, {"max_tokens": 9, "temperature": 0.2})
    assert url.endswith("/chat/completions") and body["max_completion_tokens"] == 9 and "temperature" not in body
    url, hdr, body = build_request(resolve("claude-haiku-4-5"), msgs, {"max_tokens": 9})
    assert url.endswith("/v1/messages") and body["system"] == "S" and body["messages"] == [{"role": "user", "content": "U"}]
    assert hdr["x-api-key"] == "k2"
    url, hdr, body = build_request(resolve("gemini/gemini-2.5-flash"), msgs, {"response_format": {"type": "json_object"}})
    assert ":generateContent" in url and body["systemInstruction"]["parts"][0]["text"] == "S"
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert resolve("llama3-70b-8192", "groq").provider == "groq"
    assert resolve("global.anthropic.claude-sonnet-4-6").api == "bedrock"
    assert resolve("qwen-plus").provider == "dashscope"
    r = parse_response(resolve("claude-haiku-4-5"), {"content": [{"type": "text", "text": "x"}],
                                                     "usage": {"input_tokens": 5, "output_tokens": 2,
                                                               "cache_read_input_tokens": 10}})
    assert (r.text, r.input_tokens, r.cached_input_tokens) == ("x", 15, 10)


# ------------------------------------------------------------------ runner with simulated models
def _samples(n, task="classification", gold=False, ref=True):
    out = []
    for i in range(n):
        label = ["billing", "bug", "account"][i % 3]
        s = make_sample([{"role": "user", "content": f"ticket {i}"}], {"max_tokens": 5},
                        reference=label if ref else None, expected=label if gold else None)
        s.meta["gold"] = label
        out.append(s)
    return out


def _fake(acc: dict, calls: list, lock=threading.Lock()):
    """call_fn: each model answers correctly with its accuracy (deterministic per sample)."""
    def fn(route, messages, params):
        user = messages[-1]["content"]
        with lock:
            calls.append((route.model, user))
        if user.startswith("You are grading"):
            return Result(text="TIE", input_tokens=50, output_tokens=1, latency_ms=1)
        i = int(user.split()[-1])
        gold = ["billing", "bug", "account"][i % 3]
        h = int(hashlib.sha1(f"{route.model}{i}".encode()).hexdigest()[:8], 16) / 16 ** 8
        text = gold if h < acc.get(route.model, 1.0) else "other"
        return Result(text=text, input_tokens=20, output_tokens=2, latency_ms=1)
    return fn


def test_run_validates_good_and_rejects_bad(tmp_path):
    calls = []
    cfg = Config(baseline="gpt-4o", candidates=["gpt-4o-mini", "gpt-4.1-nano"], max_cost=1.0, concurrency=3)
    res = run(cfg, _samples(120), tmp_path, CAT, call_fn=_fake({"gpt-4o-mini": 1.0, "gpt-4.1-nano": 0.6}, calls))
    v = {c["model"]: c for c in res["candidates"]}
    assert v["gpt-4o-mini"]["verdict"] == "PASS" and v["gpt-4.1-nano"]["verdict"] == "FAIL"
    assert v["gpt-4.1-nano"]["stopped_early"] and v["gpt-4.1-nano"]["scored"] < 60     # stopped paying for it
    assert res["recommended"] == "gpt-4o-mini" and res["reference"] == "production outputs"
    assert not any(m == "gpt-4o" for m, _ in calls)                                     # production outputs reused
    assert 0 < v["gpt-4o-mini"]["savings_pct"] < 1
    price = CAT.lookup("gpt-4o-mini")
    assert res["models"]["gpt-4o-mini"]["cost_usd"] == pytest.approx(120 * price.cost(20, 2), rel=1e-6)
    assert not any("text" in x for x in res["per_sample"])                               # no answers in the result


def test_budget_is_a_hard_cap_and_resume_never_pays_twice(tmp_path):
    calls = []
    s = _samples(200, ref=False)
    cfg = Config(baseline="gpt-4o", candidates=["gpt-4o-mini"], max_cost=0.0025, concurrency=4, name="b")
    r1 = run(cfg, s, tmp_path, CAT, call_fn=_fake({}, calls))
    assert r1["stopped"] == "budget" and r1["spent_this_run_usd"] <= 0.0025
    first = len(calls)
    cfg.max_cost = 5.0
    r2 = run(cfg, s, tmp_path, CAT, call_fn=_fake({}, calls))
    assert r2["stopped"] is None and r2["calls"]["cached"] == first
    assert len(calls) == len(set(calls)) == 400                                          # each request paid once
    r3 = run(cfg, s, tmp_path, CAT, call_fn=_fake({}, calls))
    assert r3["calls"]["new_calls"] == 0 and r3["spent_this_run_usd"] == 0


def test_gold_answers_judge_and_unpriced(tmp_path):
    calls = []
    res = run(Config(baseline="gpt-4o", candidates=["gpt-4o-mini"], max_cost=1.0), _samples(60, gold=True, ref=False),
              tmp_path, CAT, call_fn=_fake({"gpt-4o": 0.97, "gpt-4o-mini": 1.0}, calls))
    assert res["reference"] == "expected answers" and res["baseline_accuracy"] < 1
    gen = [make_sample([{"role": "user", "content": f"Explain topic {i}"}], {}, reference="A long answer about it.")
           for i in range(30)]
    nj = run(Config(baseline="gpt-4o", candidates=["gpt-4o-mini"], task="generation", max_cost=1.0, name="nj"),
             gen, tmp_path, CAT, call_fn=lambda r, m, p: Result(text="Another answer.", input_tokens=5, output_tokens=5))
    assert nj["candidates"][0]["verdict"] == "NEEDS-JUDGE"
    wj = run(Config(baseline="gpt-4o", candidates=["gpt-4o-mini"], task="generation", judge="gpt-4.1", max_cost=1.0,
                    name="wj"), gen, tmp_path, CAT,
             call_fn=lambda r, m, p: Result(text="TIE" if m[-1]["content"].startswith("You are grading") else "Ans.",
                                            input_tokens=5, output_tokens=1))
    assert wj["candidates"][0]["verdict"] == "PASS" and wj["candidates"][0]["judge"]["not_worse"] == 30
    with pytest.raises(ValueError, match="no price"):
        run(Config(baseline="gpt-4o", candidates=["my-private-model-x"], max_cost=1.0), gen, tmp_path, CAT,
            call_fn=lambda r, m, p: Result(text="x"))


def test_plan_lists_destinations_and_problems(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = plan(Config(baseline="gpt-4o", candidates=["claude-haiku-4-5"], max_cost=0.000001), _samples(10))
    assert any("ANTHROPIC_API_KEY" in x for x in p["problems"])
    assert any("worst case of a single call" in x for x in p["problems"])
    assert "api.anthropic.com" in p["data_goes_to"]


# ------------------------------------------------------------------ report integration
def test_report_marks_finding_validated(tmp_path):
    from trainlab_optimize.analyzer.experiments import apply
    from trainlab_optimize.schema import Attribution, CallSite, Finding
    site = CallSite(id="s1", path="app/a.py", line=10, kind="chat", provider="openai", sdk_call="x", model="gpt-4o")
    f = Finding(id="", rule="R101", title="t", severity="medium", strategy="model", call_site_ids=["s1"],
                summary="", recommendation="r", savings_low_usd=10, savings_high_usd=20, tier="estimated")
    x = {"id": "x-1", "site": "app/a.py:10", "site_id": "s1", "baseline": "gpt-4o", "recommended": "gpt-4o-mini",
         "threshold": 0.95, "samples": 100, "task": "classification", "created": "2026-10-15",
         "candidates": [{"model": "gpt-4o-mini", "verdict": "PASS", "passed": 100, "scored": 100, "pass_rate": 1.0,
                         "lower_95": 0.97, "savings_pct": 0.9}]}
    out = apply([f], [site], {"s1": Attribution("s1", "gpt-4o", 1000.0, 10.0, "exact_trace")}, [x])
    assert out[0].tier == "validated" and out[0].savings_high_usd == pytest.approx(900)
    x2 = {**x, "recommended": None, "candidates": [{**x["candidates"][0], "verdict": "FAIL"}]}
    f2 = Finding(id="", rule="R101", title="t", severity="medium", strategy="model", call_site_ids=["s1"],
                 summary="", recommendation="r", savings_high_usd=20, tier="estimated")
    out = apply([f2], [site], {}, [x2])
    assert out[0].severity == "info" and out[0].savings_high_usd == 0


# ------------------------------------------------------------------ capture
def test_capture_writes_redacted_samples(tmp_path):
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    import trainlab_optimize as trainlab

    def handler(request):
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 1, "model": "gpt-4o",
                                         "choices": [{"index": 0, "message": {"role": "assistant", "content": "billing"},
                                                      "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 10, "completion_tokens": 1}})
    c = openai.OpenAI(api_key="k", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    trainlab.instrument(path=False, project_root=str(tmp_path.parent),
                        capture={"path": str(tmp_path / "samples"), "max_per_site": 3, "sample_rate": 1.0})
    try:
        for i in range(5):
            c.chat.completions.create(model="gpt-4o", max_tokens=5,
                                      messages=[{"role": "system", "content": "Classify."},
                                                {"role": "user", "content": f"refund order {i} to jo{i}@example.com"}])
    finally:
        trainlab.uninstrument()
    files = list((tmp_path / "samples").glob("*.jsonl"))
    assert len(files) == 1
    lines = [json.loads(x) for x in files[0].read_text().splitlines()]
    assert len(lines) == 3                                                     # max_per_site
    assert lines[0]["output"] == "billing" and lines[0]["params"] == {"max_tokens": 5}
    assert "<EMAIL>" in lines[0]["messages"][1]["content"] and "example.com" not in files[0].read_text()
    s = load_samples(tmp_path / "samples")
    assert len(s) == 3 and s[0].reference == "billing" and s[0].site == lines[0]["site"]


def test_capture_is_off_by_default(tmp_path):
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    import trainlab_optimize as trainlab
    c = openai.OpenAI(api_key="k", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}}],
                                            "model": "m", "id": "x", "object": "chat.completion", "created": 1}))))
    trainlab.instrument(path=False, project_root=str(tmp_path))
    try:
        c.chat.completions.create(model="m", messages=[{"role": "user", "content": "secret"}])
    finally:
        trainlab.uninstrument()
    assert not (tmp_path / ".trainlab" / "samples").exists()


# ------------------------------------------------------------------ CLI round trip against a local server
def test_cli_run_list_show(tmp_path, monkeypatch, capsys):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            user = body["messages"][-1]["content"]
            label = ["billing", "bug", "account"][int(user.split()[-1]) % 3]
            data = json.dumps({"id": "x", "object": "chat.completion", "created": 1, "model": body["model"],
                               "choices": [{"index": 0, "message": {"role": "assistant", "content": label},
                                            "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 12, "completion_tokens": 1}}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{srv.server_address[1]}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(k, raising=False)
    f = tmp_path / "s.jsonl"
    f.write_text("\n".join(json.dumps({"messages": [{"role": "user", "content": f"ticket {i}"}],
                                       "output": ["billing", "bug", "account"][i % 3]}) for i in range(80)))
    from trainlab_optimize.cli import main
    ws = str(tmp_path / "ws")
    try:
        assert main(["--workspace", ws, "experiment", "run", "--samples", str(f), "--baseline", "gpt-4o",
                     "--candidates", "gpt-4o-mini", "--max-cost", "0.5", "--name", "t1"]) == 0
        out = capsys.readouterr().out
        assert "PASS" in out and "Validated" in out
        assert main(["--workspace", ws, "experiment", "list"]) == 0
        assert "t1" in capsys.readouterr().out
        assert main(["--workspace", ws, "experiment", "show", "t1"]) == 0
        assert "gpt-4o-mini" in capsys.readouterr().out
        assert "sk-test" not in "".join(p.read_text() for p in (tmp_path / "ws").rglob("*") if p.is_file())
    finally:
        srv.shutdown()
