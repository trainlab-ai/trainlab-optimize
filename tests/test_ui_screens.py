"""The UI's data endpoints (Overview, Spend, Live, Experiments) against a recorded support-desk run."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("openai")
ROOT = Path(__file__).resolve().parents[1]
DESK = ROOT / "examples" / "support_desk"


@pytest.fixture(scope="module")
def ui():
    work = Path(tempfile.mkdtemp())
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    env.pop("TRAINLAB_INSTRUMENT", None)
    r = subprocess.run([sys.executable, "-m", "trainlab_optimize.cli", "--workspace", str(work / "ws"), "trace",
                        "--capture", "--path", str(DESK), "--", sys.executable, str(DESK / "run.py"), "--mock",
                        "--tickets", "60"], cwd=work, env=env, capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stdout + r.stderr
    mock = subprocess.Popen([sys.executable, str(DESK / "mock_openai.py"), "--port", "0"], stdout=subprocess.PIPE,
                            text=True)
    base = mock.stdout.readline().split()[4]
    old = {k: os.environ.get(k) for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY")}
    os.environ.update(OPENAI_BASE_URL=base, OPENAI_API_KEY="mock")
    from trainlab_optimize.ui import serve
    from trainlab_optimize.workspace import Workspace
    httpd, app, url = serve(Workspace(work / "ws"), 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield url.split("/?")[0], app, work
    httpd.shutdown()
    mock.terminate()
    mock.wait()
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def call(ui, path, body=None, token=True):
    base, app, _ = ui
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 method="POST" if body is not None else "GET",
                                 headers={"X-Trainlab-Token": app.token} if token else {})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_every_endpoint_needs_the_session_token(ui):
    for path in ("/api/overview", "/api/spend", "/api/live", "/api/experiments"):
        assert call(ui, path, token=False)[0] == 403


def test_overview_names_the_untested_calls(ui):
    _, o = call(ui, "/api/overview")
    assert o["ready"] and o["has_usage"] and o["traced_share"] > 0.99
    assert {u["site"] for u in o["untested"]} == {"support_desk/extract.py:40", "support_desk/triage.py:17"}
    assert o["validated"] == 0 and not o["tested"]


def test_spend_shows_cost_beside_the_line_and_serves_only_scanned_files(ui):
    _, s = call(ui, "/api/spend")
    files = {f["path"]: f for f in s["files"]}
    assert files["support_desk/extract.py"]["sites"][0]["line"] == 40
    assert files["support_desk/extract.py"]["sites"][0]["method"] == "exact_trace"
    code, src = call(ui, "/api/source?path=support_desk/triage.py")
    assert code == 200 and "chat.completions.create" in src["lines"][16]
    for bad in ("../../../etc/passwd", "run.py", "support_desk/data.py"):
        assert call(ui, "/api/source?path=" + bad)[0] == 400


def test_live_counts_every_recorded_call_and_follows_new_ones(ui):
    _, app, work = ui
    _, L = call(ui, "/api/live")
    traces = sorted((work / "ws" / "traces").glob("*.jsonl"))
    recorded = sum(1 for f in traces for _ in open(f))
    assert L["events"] == recorded == L["calls_window"]
    assert {s["site"] for s in L["sites"]} >= {"support_desk/triage.py:17", "support_desk/kb.py:14"}
    ev = {"v": 1, "provider": "openai", "model": "gpt-4o", "op": "chat", "ts": time.strftime("%Y-%m-%dT%H:%M:%S.000Z",
          time.gmtime(time.time() + 2)), "input_tokens": 100, "output_tokens": 5, "ms": 50.0, "ok": True,
          "path": "support_desk/triage.py", "line": 17, "func": "classify", "tags": {"feature": "triage"}}
    with open(traces[0], "a") as fh:
        fh.write(json.dumps(ev) + "\n")
    _, L2 = call(ui, "/api/live")
    assert L2["events"] == recorded + 1 and L2["live"] is True
    assert L2["recent"][0]["site"] == "support_desk/triage.py:17"


def test_experiment_from_the_ui_plans_runs_and_shows_disagreements(ui):
    _, app, _ = ui
    _, x = call(ui, "/api/experiments")
    site = next(s for s in x["sites"] if s["site"] == "support_desk/triage.py:17")
    assert site["samples"] >= 50 and "gpt-4o-mini" in site["candidates"]
    code, p = call(ui, "/api/experiments/plan", {"site": site["site"], "max_cost": 1.0})
    assert code == 200 and p["samples"] >= 50 and not p.get("blocking")
    code, err = call(ui, "/api/experiments/plan", {"site": "nowhere.py:1"})
    assert code == 400
    code, job = call(ui, "/api/experiments/run", {"site": site["site"], "max_cost": 1.0})
    assert code == 200 and job["status"] == "running"
    for _ in range(240):
        j = next(j for j in app.jobs.list() if j["id"] == job["id"])
        if j["status"] != "running":
            break
        time.sleep(0.25)
    assert j["status"] == "done", j
    _, x = call(ui, "/api/experiments")
    assert x["experiments"][0]["id"] == j["result"]
    _, d = call(ui, "/api/experiments/" + j["result"])
    assert d["recommended"] == "gpt-4o-mini" and "per_sample" not in d
    assert any(c.get("examples") for c in d["candidates"])          # gpt-5-mini disagrees on some labels
    assert call(ui, "/api/experiments/..%2F..%2Fetc")[0] == 400
    _, o = call(ui, "/api/overview")
    assert [t["site"] for t in o["tested"]] == ["support_desk/triage.py:17"] and o["validated"] > 0
