"""Sandbox test: real SDKs, real HTTP, fake server ground truth vs trainlab traces.

    python -m venv venv && venv/bin/pip install . openai anthropic litellm google-genai boto3 \
        fastapi uvicorn gunicorn httpx
    venv/bin/python bench/sandbox/run_sandbox.py        # results in bench/sandbox/out/results.json
"""
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
BIN = Path(sys.executable).parent
APP = HERE / "acme"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
ENV = {**os.environ, "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
       "PYTHONPATH": str(APP), "PYTHONWARNINGS": "ignore"}
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy", "ANTHROPIC_BASE_URL"):
    ENV.pop(k, None)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def start_server(name):
    port, truth = free_port(), OUT / f"{name}.truth.jsonl"
    truth.unlink(missing_ok=True)
    p = subprocess.Popen([PY, str(HERE / "fake_llm.py"), str(port), str(truth)], env=ENV, stdout=subprocess.PIPE)
    p.stdout.readline()
    return p, port, truth


def run_py(name, code, port, extra_env=None, timeout=600):
    traces = OUT / name / "traces"
    if traces.exists():
        for f in traces.glob("*"):
            f.unlink()
    env = {**ENV, "LLM_BASE": f"http://127.0.0.1:{port}", "TRACES": str(traces), **(extra_env or {})}
    t0 = time.perf_counter()
    r = subprocess.run([PY, "-c", textwrap.dedent(code)], cwd=APP, env=env, capture_output=True, text=True,
                       timeout=timeout)
    return r, traces, time.perf_counter() - t0


def load(path_or_dir):
    p = Path(path_or_dir)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p] if p.exists() else []
    return [json.loads(ln) for f in files for ln in f.read_text().splitlines() if ln.strip()]


def reconcile(name, truth_rows, events, note=""):
    ok_truth = [t for t in truth_rows if t["status"] == 200]
    tr = Counter(t.get("feature") for t in ok_truth)
    ev_ok = [e for e in events if e.get("ok")]
    ef = Counter((e.get("tags") or {}).get("feature") for e in ev_ok)
    sums = lambda rows, k: sum(r.get(k) or 0 for r in rows)  # noqa: E731
    exact_rows = [t for t in ok_truth if t.get("usage", True)]
    row = {
        "scenario": name,
        "server_calls": len(ok_truth),
        "recorded_ok": len(ev_ok),
        "input_truth": sums(ok_truth, "inp"), "input_rec": sums(ev_ok, "input_tokens"),
        "output_truth": sums(exact_rows, "out"),
        "output_rec": sum(e.get("output_tokens") or 0 for e in ev_ok if not e.get("output_tokens_estimated")),
        "cached_truth": sums(ok_truth, "cached"), "cached_rec": sums(ev_ok, "cached_input_tokens"),
        "cw_truth": sums(ok_truth, "cw"), "cw_rec": sums(ev_ok, "cache_write_tokens"),
        "features_match": tr == ef,
        "located": sum(1 for e in events if e.get("path") and not e["path"].startswith("/")),
        "events": len(events),
        "note": note,
    }
    if tr != ef:
        row["feature_diff"] = {k: (tr.get(k, 0), ef.get(k, 0)) for k in set(tr) | set(ef) if tr.get(k) != ef.get(k)}
    return row


RESULTS = []

# ---------------------------------------------------------------- S1: sync script, every SDK shape
srv, port, truth = start_server("s1")
r, traces, secs = run_py("s1", """
    import os, trainlab_optimize as trainlab
    trainlab.instrument(path=os.environ["TRACES"], service="acme-batch", tags={"env": "sandbox"})
    from acme import features as f
    for i in range(30):
        f.support_answer(f"question {i}")
        f.classify(f"ticket {i}")
    for i in range(10):
        f.summarise_stream("doc " * 50, include_usage=True)
        f.summarise_stream("doc " * 50, include_usage=False)
        f.first_token_only("doc")
        f.agent_step("plan a trip")
        f.embed([f"chunk {j}" for j in range(16)])
        f.extract_invoice("INVOICE " * 40)
        f.draft_email("ctx " * 30)
        f.translate("hola " * 10)
        f.translate_stream("bonjour " * 10)
    import gc; gc.collect()
    trainlab.flush()
    print(trainlab.status())
""", port)
srv.terminate()
assert r.returncode == 0, r.stderr[-3000:]
ev = load(traces)
RESULTS.append(reconcile("S1 sync: OpenAI chat/stream/responses/embeddings, Anthropic create/stream, Gemini",
                         load(truth), ev, f"{secs:.1f}s"))
s1_events = ev

# ---------------------------------------------------------------- S2: asyncio, 300 concurrent tasks, tags per task
srv, port, truth = start_server("s2")
r, traces, secs = run_py("s2", """
    import os, asyncio, trainlab_optimize as trainlab
    trainlab.instrument(path=os.environ["TRACES"])
    from acme import features as f
    async def main():
        await asyncio.gather(*(f.async_chat(i) for i in range(300)))
    asyncio.run(main())
    trainlab.flush()
""", port)
srv.terminate()
assert r.returncode == 0, r.stderr[-3000:]
RESULTS.append(reconcile("S2 asyncio: 300 concurrent calls, 7 tenant tags, 3 SDKs", load(truth), load(traces),
                         f"{secs:.1f}s"))

# ---------------------------------------------------------------- S3: LiteLLM (calls the OpenAI / HTTP layer itself)
srv, port, truth = start_server("s3")
r, traces, secs = run_py("s3", """
    import os, trainlab_optimize as trainlab
    trainlab.instrument(path=os.environ["TRACES"])
    from acme.litellm_feature import via_litellm
    for i in range(40):
        via_litellm(f"q{i}", i)
    trainlab.flush()
""", port)
srv.terminate()
assert r.returncode == 0, r.stderr[-3000:]
ev3 = load(traces)
RESULTS.append(reconcile("S3 LiteLLM → OpenAI + Anthropic (no double count)", load(truth), ev3,
                         f"sdk: {dict(Counter(e['sdk'] for e in ev3))}"))

# ---------------------------------------------------------------- S4: errors and SDK retries
srv, port, truth = start_server("s4")
r, traces, secs = run_py("s4", """
    import os, openai, anthropic, trainlab_optimize as trainlab
    trainlab.instrument(path=os.environ["TRACES"])
    base = os.environ["LLM_BASE"]
    oa = openai.OpenAI(base_url=base + "/v1", api_key="k", max_retries=2)
    an = anthropic.Anthropic(base_url=base, api_key="k", max_retries=0)
    caught = []
    for code in ("500", "429", "400"):
        try:
            oa.with_options(timeout=5).chat.completions.create(model="gpt-4o", messages=[],
                extra_headers={"x-fail": code, "x-stainless-retry-count": "0"})
        except Exception as e:
            caught.append(type(e).__name__)
    try:
        an.messages.create(model="claude-haiku-4-5", max_tokens=5, messages=[], extra_headers={"x-fail": "529"})
    except Exception as e:
        caught.append(type(e).__name__)
    oa.chat.completions.create(model="gpt-4o", messages=[])        # one success afterwards
    trainlab.flush()
    print("app caught:", caught)
""", port, timeout=120)
srv.terminate()
assert r.returncode == 0, r.stderr[-3000:]
t4, e4 = load(truth), load(traces)
RESULTS.append({"scenario": "S4 errors: 500/429 (SDK retries x2), 400, Anthropic 529",
                "server_calls": len(t4), "recorded_ok": sum(1 for e in e4 if e["ok"]),
                "events": len(e4), "errors_recorded": [e.get("error") for e in e4 if not e["ok"]],
                "app_saw": r.stdout.strip(), "note": "server hits incl. retries vs logical calls"})

# ---------------------------------------------------------------- S5: FastAPI under gunicorn, 4 workers, --preload
srv, port, truth = start_server("s5")
traces5 = OUT / "s5" / "traces"
for f in traces5.glob("*") if traces5.exists() else []:
    f.unlink()
wport = free_port()
genv = {**ENV, "LLM_BASE": f"http://127.0.0.1:{port}", "TRAINLAB_PROJECT_ROOT": str(APP)}
gun = subprocess.Popen([str(BIN / "gunicorn"), "acme.web:app", "-w", "4", "--preload",
                        "-k", "uvicorn.workers.UvicornWorker", "-b", f"127.0.0.1:{wport}", "--log-level", "warning"],
                       cwd=APP, env=genv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
for _ in range(100):
    try:
        socket.create_connection(("127.0.0.1", wport), timeout=0.2).close()
        break
    except OSError:
        time.sleep(0.2)
load_code = f"""
import httpx, concurrent.futures as cf
def hit(i):
    feat = ["search", "chat", "billing", "ops"][i % 4]
    kind = "sync" if i % 2 else "async"
    with httpx.Client(timeout=30) as c:
        return c.post(f"http://127.0.0.1:{wport}/{{kind}}/{{feat}}", headers={{"x-feature": feat}}).status_code
with cf.ThreadPoolExecutor(32) as ex:
    codes = list(ex.map(hit, range(800)))
print(sum(c == 200 for c in codes))
"""
lr = subprocess.run([PY, "-c", load_code], env=ENV, capture_output=True, text=True, timeout=600)
time.sleep(0.5)
gun.send_signal(signal.SIGTERM)                       # graceful shutdown: workers flush at exit
gun.wait(timeout=60)
srv.terminate()
# with --preload the traces go to <cwd>/.trainlab/traces of the app
ev5 = load(APP / ".trainlab" / "traces")
RESULTS.append(reconcile("S5 FastAPI + gunicorn 4 workers --preload, 800 requests, 32 concurrent", load(truth),
                         ev5, f"HTTP 200s: {lr.stdout.strip()}; trace files: "
                              f"{len(list((APP / '.trainlab' / 'traces').glob('*.jsonl')))}"))
s5_events = ev5

# ---------------------------------------------------------------- S6: sampling 10%
srv, port, truth = start_server("s6")
r, traces, secs = run_py("s6", """
    import os, trainlab_optimize as trainlab
    trainlab.instrument(path=os.environ["TRACES"], sample_rate=0.1)
    from acme import features as f
    for i in range(2000):
        f.classify(f"t{i}")
    trainlab.flush()
""", port)
srv.terminate()
t6, e6 = load(truth), load(traces)
RESULTS.append({"scenario": "S6 sample_rate=0.1 over 2000 calls", "server_calls": len(t6), "events": len(e6),
                "weighted_calls": sum(e["weight"] for e in e6),
                "weighted_input": round(sum(e["input_tokens"] * e["weight"] for e in e6)),
                "input_truth": sum(t["inp"] for t in t6)})

# ---------------------------------------------------------------- S7: kill switch
srv, port, truth = start_server("s7")
r, traces, secs = run_py("s7", """
    import os, trainlab_optimize as trainlab
    print(trainlab.instrument(path=os.environ["TRACES"]))
    from acme import features as f
    for i in range(20):
        f.classify("x")
""", port, extra_env={"TRAINLAB_INSTRUMENT": "0"})
srv.terminate()
RESULTS.append({"scenario": "S7 TRAINLAB_INSTRUMENT=0", "server_calls": len(load(truth)),
                "events": len(load(traces)), "exit": r.returncode, "instrument_returned": r.stdout.strip()})

# ---------------------------------------------------------------- S8: overhead over real HTTP
srv, port, truth = start_server("s8")
bench = """
    import os, time, statistics, openai
    on = os.environ.get("ON") == "1"
    if on:
        import trainlab_optimize as trainlab; trainlab.instrument(path=os.environ["TRACES"])
    oa = openai.OpenAI(base_url=os.environ["LLM_BASE"] + "/v1", api_key="k", max_retries=0)
    for _ in range(100): oa.chat.completions.create(model="gpt-4o-mini", messages=[])
    xs = []
    for _ in range(7):
        t = time.perf_counter()
        for _ in range(300): oa.chat.completions.create(model="gpt-4o-mini", messages=[])
        xs.append((time.perf_counter() - t) / 300 * 1e3)
    print(statistics.median(xs))
"""
off = float(run_py("s8", bench, port, {"ON": "0"})[0].stdout)
on_ = float(run_py("s8", bench, port, {"ON": "1"})[0].stdout)
startup = run_py("s8", """
    import time; t = time.perf_counter()
    import trainlab_optimize as trainlab; trainlab.instrument(path=False)
    print(time.perf_counter() - t)
""", port)[0].stdout
srv.terminate()
RESULTS.append({"scenario": "S8 overhead (localhost HTTP round trip)", "ms_per_call_off": round(off, 3),
                "ms_per_call_on": round(on_, 3), "overhead_ms": round(on_ - off, 3),
                "instrument_startup_s": round(float(startup), 3)})

json.dump({"results": RESULTS}, open(OUT / "results.json", "w"), indent=1)
json.dump({"s1": s1_events[:5], "s5_paths": {f"{p}:{l}": n for (p, l), n in Counter((e.get("path"), e.get("line")) for e in s5_events).items()}},
          open(OUT / "samples.json", "w"), indent=1)
for row in RESULTS:
    print(json.dumps(row))
