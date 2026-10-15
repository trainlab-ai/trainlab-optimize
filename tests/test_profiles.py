"""One command per profile (trace, experiment run with no --site), and the fixes the
support-desk sample app turned up."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

from trainlab_optimize.experiment.gates import json_agreement
from trainlab_optimize.experiment.runner import Config, plan
from trainlab_optimize.experiment.samples import make_sample
from trainlab_optimize.redact import redact
from trainlab_optimize.runtime.capture import Capture
from trainlab_optimize.scanner import scan

ROOT = Path(__file__).resolve().parents[1]
DESK = ROOT / "examples" / "support_desk"


# ------------------------------------------------------------------ scanner: agent loops aren't batchable
def test_loop_that_feeds_answers_back_is_not_a_batch_loop(tmp_path):
    (tmp_path / "a.py").write_text(textwrap.dedent('''
        from openai import OpenAI
        c = OpenAI()
        def agent(q):
            messages = [{"role": "user", "content": q}]
            for _ in range(3):
                r = c.chat.completions.create(model="gpt-4o", messages=messages)
                messages.append({"role": "assistant", "content": r.choices[0].message.content})
        def batchable(items):
            for it in items:
                msgs = [{"role": "user", "content": f"Classify {it}"}]
                c.chat.completions.create(model="gpt-4o", messages=msgs)
        def history(items):
            hist = ""
            for it in items:
                r = c.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": hist + it}])
                hist += r.choices[0].message.content
        def collect(items):
            out = []
            for it in items:
                out.append(c.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": it}]))
    '''))
    kinds = {s.function: s.loop_kind for s in scan(str(tmp_path)).call_sites}
    assert kinds == {"agent": "agent", "batchable": "for", "history": "agent", "collect": "for"}


# ------------------------------------------------------------------ redaction keeps prompt layout
def test_redaction_keeps_whitespace_but_previews_are_one_line():
    text = "Subject: Refund\n\nMail me at a.b@example.com or +44 7123 456789\n\nThanks"
    assert redact(text) == "Subject: Refund\n\nMail me at <EMAIL> or <PHONE>\n\nThanks"
    assert redact(text, 200) == "Subject: Refund Mail me at <EMAIL> or <PHONE> Thanks"


# ------------------------------------------------------------------ extraction: prose fields compared loosely
def test_prose_fields_in_json_are_compared_by_overlap_other_fields_exactly():
    ref = {"order_id": "NW-12345", "summary": "Customer was charged twice for the Pro plan"}
    assert json_agreement(ref, {"order_id": "NW-12345",
                                "summary": "The customer got charged twice for their Pro plan"})[0]
    assert not json_agreement(ref, {"order_id": "NW-12345", "summary": "Sync is broken on Windows laptops"})[0]
    assert not json_agreement(ref, {"order_id": "NW-12346", "summary": ref["summary"]})[0]
    # a long value under a non-prose key still has to match exactly
    assert not json_agreement({"address": "221B Baker Street London NW1 6XE"},
                              {"address": "221B Baker Street London"})[0]


# ------------------------------------------------------------------ capture: tool calls are skipped
@pytest.mark.parametrize("req, tools", [
    ({"messages": [{"role": "user", "content": "hi"}]}, False),
    ({"messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function"}]}, True),
    ({"messages": [{"role": "assistant", "tool_calls": [{"id": "1"}]}, {"role": "tool", "content": "{}"}]}, True),
    ({"input": [{"type": "function_call_output", "output": "x"}]}, True),
    ({"messages": [{"role": "user", "content": [{"type": "tool_result", "content": "x"}]}]}, True),
])
def test_capture_skips_tool_calls(req, tools):
    assert Capture.uses_tools(req) is tools


# ------------------------------------------------------------------ plan blocks runs that can't give a verdict
def _samples(n, out):
    return [make_sample([{"role": "user", "content": f"ticket {i}"}], reference=out(i), site="a.py:1")
            for i in range(n)]


def test_plan_blocks_free_text_without_judge_and_too_few_samples(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    prose = lambda i: f"Thanks for writing in about ticket {i}. We have looked into it and fixed the issue for you today."
    p = plan(Config(baseline="gpt-4o", candidates=["gpt-4o-mini"], max_cost=5), _samples(30, prose))
    assert p["task"] == "generation" and any("--judge" in b for b in p["blocking"])
    p = plan(Config(baseline="gpt-4o", candidates=["gpt-4o-mini"], max_cost=5, judge="gpt-4.1"), _samples(30, prose))
    assert not p["blocking"]
    p = plan(Config(baseline="gpt-4o", candidates=["gpt-4o-mini"], max_cost=5), _samples(5, lambda i: "billing"))
    assert any("at least 20" in b for b in p["blocking"])


# ------------------------------------------------------------------ a passing model that costs more is not recommended
def test_passing_model_that_costs_more_is_not_validated():
    from trainlab_optimize.analyzer import experiments as ax
    from trainlab_optimize.schema import CallSite, Finding
    site = CallSite(id="s1", path="a.py", line=3, kind="chat", provider="openai", sdk_call="x", model="gpt-4o-mini")
    x = {"id": "x-1", "site": "a.py:3", "site_id": "s1", "baseline": "gpt-4o-mini", "threshold": 0.95,
         "recommended": "gpt-5-nano", "created": "2026-10-15",
         "candidates": [{"model": "gpt-5-nano", "verdict": "PASS", "passed": 58, "scored": 58, "pass_rate": 1.0,
                         "lower_95": 0.955, "savings_pct": -4.76}]}
    out = ax.apply([], [site], {}, [x])
    assert not any(f.tier == "validated" and f.savings_high_usd for f in out)
    f = Finding(id="", rule="R101", title="t", severity="medium", strategy="model", call_site_ids=["s1"],
                summary="", recommendation="", evidence={})
    ax.apply([f], [site], {}, [x])
    assert not f.title.startswith("Validated: switch")


# ------------------------------------------------------------------ end to end on the support desk, against the mock
openai = pytest.importorskip("openai")


def _cli(*args, cwd, env=None, timeout=240):
    e = {**os.environ, **(env or {})}
    e.pop("TRAINLAB_INSTRUMENT", None)
    return subprocess.run([sys.executable, "-m", "trainlab_optimize.cli", *args], cwd=cwd, env=e,
                          capture_output=True, text=True, timeout=timeout)


@pytest.fixture(scope="module")
def desk_run():
    """trace --capture on 100 tickets, with the mock's own usage log as ground truth."""
    work = Path(tempfile.mkdtemp())
    log = work / "mock.log"
    r = _cli("--workspace", str(work / "ws"), "trace", "--capture", "--path", str(DESK), "--",
             sys.executable, str(DESK / "run.py"), "--mock", "--tickets", "100",
             cwd=work, env={"MOCK_OPENAI_LOG": str(log), "PYTHONPATH": str(ROOT / "src")})
    return work, log, r


def test_trace_records_every_call_exactly(desk_run):
    work, log, r = desk_run
    assert r.returncode == 0, r.stdout + r.stderr
    truth: dict = {}
    for line in log.read_text().splitlines():
        d = json.loads(line)
        t = truth.setdefault(d["model"], [0, 0, 0])
        t[0] += 1; t[1] += d["in"]; t[2] += d["out"]
    got: dict = {}
    sites = set()
    for f in (work / "ws" / "traces").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            e = json.loads(line)
            g = got.setdefault(e["model"], [0, 0, 0])
            g[0] += 1; g[1] += e.get("input_tokens") or 0; g[2] += e.get("output_tokens") or 0
            sites.add((e["path"], e["line"], e["tags"]["feature"]))
    assert got == truth                         # threads, asyncio, streaming, tools, Responses API, embeddings
    assert {s[0] for s in sites} == {f"support_desk/{m}.py" for m in
                                     ("triage", "extract", "sentiment", "kb", "reply", "agent", "digest")}
    assert "recorded" in r.stdout and (work / "ws" / "report" / "trainlab-report.html").exists()
    assert not (work / "ws" / "samples" / "support_desk_agent.py_35.jsonl").exists()    # tool calls skipped


def test_experiment_run_without_site_tests_every_captured_site(desk_run):
    work, _, _ = desk_run
    srv = subprocess.Popen([sys.executable, str(DESK / "mock_openai.py"), "--port", "0"],
                           stdout=subprocess.PIPE, text=True)
    try:
        url = srv.stdout.readline().split()[4]
        r = _cli("--workspace", str(work / "ws"), "experiment", "run", "--max-cost", "1",
                 cwd=work, env={"OPENAI_BASE_URL": url, "OPENAI_API_KEY": "mock"})
    finally:
        srv.terminate()
        srv.wait()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "skipped: the answers are free text" in r.stdout          # reply drafts, no judge given
    assert "costs more per call: not recommended" in r.stdout        # gpt-5-nano on sentiment (reasoning tokens)
    rep = json.loads((work / "ws" / "report" / "trainlab-report.json").read_text())
    titles = [f["title"] for f in rep["findings"]]
    # routing and extraction both validate: the small models disagree with gpt-4o on `urgency` now
    # and then, a judgment field that is measured and shown but not scored
    assert sum(1 for t in titles if t.startswith("Validated: switch `gpt-4o` to `gpt-4o-mini`")) == 2
    assert "judgment, not scored: urgency" in r.stdout and "facts (must match): order_id" in r.stdout
    assert not any("gpt-5-nano" in t and t.startswith("Validated") for t in titles)
    assert "empty answer" not in r.stdout          # reasoning models get room to answer


def test_trace_without_command_explains_usage(tmp_path):
    r = _cli("trace", cwd=tmp_path)
    assert r.returncode == 2 and "trainlab trace -- python app.py" in r.stderr


# ------------------------------------------------------------------ fixes from the first real OpenAI run
def test_reasoning_models_get_room_above_the_production_output_cap():
    from trainlab_optimize.experiment.runner import REASONING_ALLOWANCE, _out_cap, _with_cap, is_reasoning
    s = make_sample([{"role": "user", "content": "route this"}], params={"max_tokens": 5}, reference="billing")
    assert _out_cap(s, model="gpt-4o-mini") == 5
    assert _out_cap(s, model="gpt-5-mini") == 5 + REASONING_ALLOWANCE
    assert _with_cap(s, _out_cap(s, model="gpt-5-mini"))["max_tokens"] == 5 + REASONING_ALLOWANCE
    assert all(is_reasoning(m) for m in ("o4-mini", "gpt-5-nano", "openai/gpt-5", "gemini-2.5-flash",
                                         "deepseek-reasoner", "claude-3-7-sonnet-thinking"))
    assert not any(is_reasoning(m) for m in ("gpt-4o", "gpt-4.1-mini", "claude-haiku-4-5", "gemini-2.0-flash"))


def test_empty_answers_are_errors_not_quality_failures(tmp_path, monkeypatch):
    from trainlab_optimize.experiment.clients import Result
    from trainlab_optimize.experiment.runner import run
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    samples = _samples(40, lambda i: ["billing", "bug", "account"][i % 3])

    def call_fn(route, messages, params):
        return Result(text="", input_tokens=20, output_tokens=params.get("max_tokens") or 0, latency_ms=1.0)
    res = run(Config(baseline="gpt-4o", candidates=["gpt-4o-mini"], max_cost=5), samples, tmp_path, call_fn=call_fn)
    c = res["candidates"][0]
    assert c["verdict"] == "ERROR" and c["errors"] == c["scored"] and "empty answer" in c["first_error"]


def test_extraction_scores_fields_by_role(tmp_path, monkeypatch):
    """Facts decide, `urgency` is measured but not scored, prose differing in wording goes to the judge."""
    from trainlab_optimize.experiment.clients import Result
    from trainlab_optimize.experiment.gates import field_roles, json_diff_fields
    from trainlab_optimize.experiment.runner import run
    assert field_roles([{"order_id": "NW-1", "urgency": "high", "summary": "a b c", "notes": "x",
                         "address": "221B Baker Street London NW1 6XE United Kingdom"}], ignore=["notes"]) == {
        "order_id": "fact", "urgency": "judgment", "summary": "prose", "notes": "ignore", "address": "fact"}
    assert field_roles([{"urgency": "high"}], strict=["urgency"]) == {"urgency": "fact"}
    assert json_diff_fields({"a": 1, "u": "high"}, {"a": 1, "u": "normal"}) == ["u"]
    assert json_diff_fields({"a": 1}, None) == ["(invalid JSON)"]
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    ref = lambda i: json.dumps({"order_id": f"NW-{i}", "urgency": "high",
                                "summary": f"Customer {i} was charged twice and asks for a refund"})
    samples = _samples(40, ref)

    def call_fn(route, messages, params):
        text = messages[-1]["content"]
        if "field_a" in text:                                   # the judge: wording differs, same meaning
            return Result(text="TIE", input_tokens=50, output_tokens=1, latency_ms=1.0)
        i = int(text.split()[-1])
        out = {"order_id": f"NW-{i}" if i % 10 else "NW-0000", "urgency": "normal" if i % 4 == 0 else "high",
               "summary": f"Duplicate billing on account {i}; refund requested" if i % 2 else
                          f"Customer {i} was charged twice and asks for a refund"}
        return Result(text=json.dumps(out), input_tokens=20, output_tokens=10, latency_ms=1.0)

    cfg = Config(baseline="gpt-4o", candidates=["gpt-4o-mini"], max_cost=5, early_stop=False)
    c = run(cfg, samples, tmp_path / "a", call_fn=call_fn)["candidates"][0]
    assert c["fields"]["urgency"] == {"role": "judgment", "agree": 30, "n": 40, "rate": 0.75}
    assert c["fields"]["order_id"]["agree"] == 36 and c["field_mismatch"] == {"summary": 20, "order_id": 4}
    assert c["passed"] == 16          # no judge: the 20 reworded summaries count as misses, and 4 wrong order ids

    cfg = Config(baseline="gpt-4o", candidates=["gpt-4o-mini"], max_cost=5, early_stop=False, judge="gpt-4.1")
    res = run(cfg, samples, tmp_path / "b", call_fn=call_fn)
    c = res["candidates"][0]
    assert res["prose_scoring"] == "judge" and c["fields"]["summary"]["agree"] == 40
    assert c["passed"] == 36 and c["field_mismatch"] == {"order_id": 4}    # only the wrong order ids fail

    cfg = Config(baseline="gpt-4o", candidates=["gpt-4o-mini"], max_cost=5, early_stop=False, judge="gpt-4.1",
                 strict_fields=["urgency"])
    assert run(cfg, samples, tmp_path / "c", call_fn=call_fn)["candidates"][0]["passed"] == 28   # 8 more: urgency


def test_server_route_matching_is_not_a_call_site(tmp_path):
    (tmp_path / "srv.py").write_text(textwrap.dedent('''
        import requests
        def handle(path):
            if path.endswith("/chat/completions"):
                return 1
            if path in ("/v2/chat", "/v1/messages"):
                return 2
        def client(base):
            return requests.post(f"{base}/v1/chat/completions", json={"model": "gpt-4o"})
    '''))
    sites = scan(str(tmp_path)).call_sites
    assert [s.line for s in sites] == [9]


def test_validated_finding_shows_the_tested_model_not_the_static_guess():
    from trainlab_optimize.analyzer import experiments as ax
    from trainlab_optimize.schema import CallSite, Finding
    site = CallSite(id="s1", path="a.py", line=3, kind="chat", provider="openai", sdk_call="x", model="gpt-4o")
    f = Finding(id="", rule="R101", title="t", severity="medium", strategy="model", call_site_ids=["s1"],
                summary="", recommendation="", evidence={"candidate_model": "gpt-5-nano", "price_ratio": 31.8})
    x = {"id": "x-1", "site": "a.py:3", "site_id": "s1", "baseline": "gpt-4o", "threshold": 0.95,
         "recommended": "gpt-4o-mini", "created": "2026-10-15",
         "candidates": [{"model": "gpt-4o-mini", "verdict": "PASS", "passed": 97, "scored": 97, "pass_rate": 1.0,
                         "lower_95": 0.973, "savings_pct": 0.94}]}
    ax.apply([f], [site], {}, [x])
    assert "candidate_model" not in f.evidence and f.evidence["tested_model"] == "gpt-4o-mini"


def test_run_all_with_judge_does_not_starve_cheap_sites(desk_run):
    """A judged extraction is estimated far above a one-word classification. Sharing the budget by
    estimate alone once left triage too little for gpt-5-mini's worst-case reservations (4,096
    reasoning tokens a call, 4 in flight), and it stopped part-way. Each share now covers that."""
    import shutil
    work, _, _ = desk_run
    ws = work / "ws-judge"
    shutil.copytree(work / "ws", ws, ignore=shutil.ignore_patterns("experiments", "report"))
    srv = subprocess.Popen([sys.executable, str(DESK / "mock_openai.py"), "--port", "0"],
                           stdout=subprocess.PIPE, text=True)
    try:
        url = srv.stdout.readline().split()[4]
        r = _cli("--workspace", str(ws), "experiment", "run", "--max-cost", "0.5", "--judge", "gpt-4.1",
                 cwd=work, env={"OPENAI_BASE_URL": url, "OPENAI_API_KEY": "mock"})
        show = _cli("--workspace", str(ws), "experiment", "list", cwd=work)
    finally:
        srv.terminate()
        srv.wait()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "stopped: budget" not in r.stdout and "ran out of its share" not in r.stdout
    assert r.stdout.count("Validated: `gpt-4o-mini`") == 2            # triage and extraction
    xid = next(ln.split()[0] for ln in show.stdout.splitlines() if "triage.py" in ln)
    ex = _cli("--workspace", str(ws), "experiment", "show", xid, "--examples", cwd=work)
    assert "where it disagreed with gpt-4o (current)" in ex.stdout and "candidate:" in ex.stdout


def test_budget_share_covers_worst_case_reservations():
    """The numbers from the real run that starved triage: estimates triage $0.0118, judged extraction
    $0.2356, judged replies $0.168, a skipped digest, sentiment $0.0003; gpt-5-mini reserves $0.0083
    a call, 4 at a time."""
    from trainlab_optimize.experiment.cli import budget_share
    ests = [0.0118, 0.2356, 0.168, 0.0, 0.0003]
    floors = [0.0083 * 2 * 4, 0.0092 * 2 * 4, 0.0019 * 2 * 4, 0.0, 0.0018 * 2 * 4]
    triage = budget_share(1.0, ests, floors, 0)
    assert triage >= 0.0083 * 2 * 4 > 0.0284                 # 0.0284 is what it got, and it stopped
    assert budget_share(1.0, ests, floors, 1) > 0.5           # the judged site still gets the most
    assert budget_share(0.05, ests, floors, 0) == 0.05         # never more than what's left
    assert budget_share(1.0, [0, 0], [0, 0], 0) == 0.5         # no estimates: even split
