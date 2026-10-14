"""Run every framework/SDK driver under trainlab instrument() against the fake server.

    python bench/frameworks/install.py ENVS          # one venv per project (uv)
    python bench/frameworks/run.py ENVS [name ...]   # results -> bench/frameworks/results.json

Each driver runs twice against a fresh fake server: with TRAINLAB_INSTRUMENT=0 (the app on
its own) and with trainlab on. Checked per project:
- safety: the app behaves the same (exit code, number of LLM calls sent) with trainlab on;
- recorded calls = calls the server answered;
- input/output token sums = what the server reported;
- every call points at the driver's `# @call` line (the app's line, not the framework's).
"""
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENVS = Path(sys.argv[1])
ONLY = sys.argv[2:]
OUT = ENVS.parent / "fw-out"
OUT.mkdir(exist_ok=True)
corpus = json.loads((HERE / "corpus.json").read_text())

BASE_ENV = {k: v for k, v in os.environ.items()
            if k.upper() not in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL",
                                 "PYTHONPATH", "VIRTUAL_ENV")}
BASE_ENV.update({"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost", "PYTHONWARNINGS": "ignore",
                 "PYTHONDONTWRITEBYTECODE": "1"})


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def header(text, key):
    m = re.search(rf"^# {key}: (.*)$", text, re.M)
    return m.group(1).strip() if m else None


def app_env(base, tag, traces, kill):
    return {**BASE_ENV, "TRAINLAB_INSTRUMENT": "0" if kill else "1",
            "FAKE_BASE": base, "TRAINLAB_TRACES": str(traces), "FW_ROOT": str(HERE),
            "PYTHONPATH": str(HERE / "drivers"),
            "OPENAI_API_KEY": "sk-fake", "OPENAI_BASE_URL": f"{base}/v1", "OPENAI_API_BASE": f"{base}/v1",
            "ANTHROPIC_API_KEY": "sk-ant-fake", "ANTHROPIC_BASE_URL": base,
            "GEMINI_API_KEY": "g-fake", "GOOGLE_API_KEY": "g-fake", "GOOGLE_GEMINI_BASE_URL": base,
            "AWS_ACCESS_KEY_ID": "AKIAFAKE", "AWS_SECRET_ACCESS_KEY": "fake", "AWS_DEFAULT_REGION": "us-east-1",
            "AWS_REGION": "us-east-1", "AWS_ENDPOINT_URL_BEDROCK_RUNTIME": base, "AWS_EC2_METADATA_DISABLED": "true",
            "OLLAMA_HOST": base, "MISTRAL_API_KEY": "m", "CO_API_KEY": "c", "COHERE_API_KEY": "c",
            "CO_API_URL": base, "GROQ_API_KEY": "g", "GROQ_BASE_URL": base, "TOGETHER_API_KEY": "t",
            "TOGETHER_BASE_URL": f"{base}/v1", "FIREWORKS_API_KEY": "f", "HF_TOKEN": "hf_fake",
            "DASHSCOPE_API_KEY": "d", "ZHIPUAI_API_KEY": "z.z", "VOYAGE_API_KEY": "v",
            "AZURE_OPENAI_API_KEY": "a", "AZURE_OPENAI_ENDPOINT": base, "OPENAI_API_VERSION": "2024-10-21",
            "LITELLM_LOG": "ERROR", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "CREWAI_DISABLE_TELEMETRY": "true", "ANONYMIZED_TELEMETRY": "False",
            "DO_NOT_TRACK": "1", "DEEPEVAL_TELEMETRY_OPT_OUT": "YES", "MEM0_TELEMETRY": "False",
            "HAYSTACK_TELEMETRY_ENABLED": "False", "SCARF_NO_ANALYTICS": "true", "AGNO_TELEMETRY": "false",
            "TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1", "LANGCHAIN_TRACING_V2": "false",
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1", "DSPY_CACHEDIR": str(OUT / f"{tag}.dspy"),
            "HOME": str(OUT / f"{tag}.home"), "TIKTOKEN_CACHE_DIR": str(ENVS.parent / "tiktoken")}


def execute(name, src, drv, kill):
    """One run of the driver against a fresh server. Returns (rc, stderr, llm truth rows, events)."""
    tag = name.replace(":", "_") + (".off" if kill else "")
    truth, traces = OUT / f"{tag}.truth.jsonl", OUT / tag
    truth.unlink(missing_ok=True)
    traces.mkdir(exist_ok=True)
    for f in traces.glob("*"):
        f.unlink()
    for d in (OUT / f"{tag}.home", OUT / f"{tag}.dspy"):
        d.mkdir(exist_ok=True)
    port = free_port()
    senv = {**BASE_ENV, "FAKE_REPLY": (header(src, "reply") or "ok").replace("\\n", "\n"), "FAKE_SEED": "7"}
    if header(src, "tool_call"):
        senv["FAKE_TOOL_CALL"] = header(src, "tool_call")
    srv = subprocess.Popen([sys.executable, str(HERE / "fake_llm.py"), str(port), str(truth)], env=senv,
                           stdout=subprocess.PIPE)
    srv.stdout.readline()
    py = ENVS / name.replace(":", "_") / "bin" / "python"
    try:
        r = subprocess.run([str(py), str(drv)], cwd=HERE, env=app_env(f"http://127.0.0.1:{port}", tag, traces, kill),
                           capture_output=True, text=True, timeout=int(header(src, "timeout") or 240))
        rc, err = r.returncode, (r.stderr or "")[-2500:]
    except subprocess.TimeoutExpired:
        rc, err = -9, "timeout"
    time.sleep(0.2)
    srv.terminate()
    srv.wait()
    t = [json.loads(x) for x in truth.read_text().splitlines()] if truth.exists() else []
    ev = [json.loads(x) for f in sorted(traces.glob("*.jsonl")) for x in f.read_text().splitlines() if x.strip()]
    return rc, err, [x for x in t if x["api"] != "other"], ev


def run_one(name):
    drv = HERE / "drivers" / f"t_{name.replace(':', '_')}.py"
    if not drv.exists():
        return {"project": name, "status": "NO-DRIVER"}
    src = drv.read_text()
    call_lines = {i for i, ln in enumerate(src.splitlines(), 1) if "# @call" in ln}
    t0 = time.perf_counter()
    rc_off, err_off, llm_off, _ = execute(name, src, drv, kill=True)
    rc, err, llm, all_ev = execute(name, src, drv, kill=False)
    secs = round(time.perf_counter() - t0, 1)
    ok_ev = [e for e in all_ev if e.get("ok")]
    failed = [e for e in all_ev if not e.get("ok")]
    rel = f"drivers/{drv.name}"
    at_line = [e for e in ok_ev if e.get("path") == rel and e.get("line") in call_lines]
    exact_in = [e for e in ok_ev if not e.get("input_tokens_estimated")]
    exact_out = [e for e in ok_ev if not e.get("output_tokens_estimated")]
    row = {
        "project": name, "kind": corpus[name]["kind"], "api": corpus[name]["api"],
        "server_calls": len(llm), "server_calls_without_trainlab": len(llm_off), "recorded": len(ok_ev),
        "failed_recorded": [e.get("error") for e in failed],
        "input_truth": sum(x.get("inp") or 0 for x in llm if x.get("usage")),
        "input_rec": sum(e.get("input_tokens") or 0 for e in exact_in),
        "output_truth": sum(x.get("out") or 0 for x in llm if x.get("usage")),
        "output_rec": sum(e.get("output_tokens") or 0 for e in exact_out),
        "estimated": sum(1 for e in ok_ev if e.get("input_tokens_estimated") or e.get("output_tokens_estimated")),
        "located": len(at_line),
        "elsewhere": sorted({f"{e.get('path')}:{e.get('line')}" for e in ok_ev if e not in at_line})[:3],
        "providers": sorted({e["provider"] for e in ok_ev}), "sdks": sorted({e["sdk"] for e in ok_ev}),
        "models": sorted({str(e.get("model")) for e in ok_ev}),
        "exit": rc, "exit_without_trainlab": rc_off, "secs": secs,
    }
    usage_less = any(not x.get("usage") for x in llm)
    if rc_off == 0 and rc != 0:
        status = "BREAKS-APP"
    elif len(llm) != len(llm_off):
        status = "CHANGES-CALLS"
    elif rc_off != 0 and not llm_off:
        status = "DRIVER-ERROR"           # the app itself fails without trainlab: fix the driver
    elif not llm:
        status = "NO-CALLS"
    elif not ok_ev:
        status = "MISSED"
    elif len(ok_ev) != len(llm):
        status = "COUNT-MISMATCH"
    elif row["located"] != len(ok_ev):
        status = "WRONG-LINE"
    elif row["input_rec"] != row["input_truth"] or row["output_rec"] != row["output_truth"]:
        status = "ESTIMATED" if row["estimated"] and usage_less else "TOKEN-MISMATCH"
    else:
        status = "PASS"
    row["status"] = status
    if status != "PASS":
        row["stderr"] = err if rc else err_off
    return row


if __name__ == "__main__":
    import concurrent.futures as cf
    names = [n for n in corpus if not ONLY or n in ONLY]
    with cf.ThreadPoolExecutor(int(os.environ.get("FW_JOBS", "6"))) as ex:
        rows = list(ex.map(run_one, names))
    res_file = HERE / "results.json"
    old = json.loads(res_file.read_text()) if res_file.exists() and ONLY else {"results": []}
    keep = {r["project"]: r for r in old["results"]}
    keep.update({r["project"]: r for r in rows})
    res_file.write_text(json.dumps({"results": [keep[n] for n in corpus if n in keep]}, indent=1))
    for r in rows:
        print(f"{r['status']:15} {r['project']:24} server={r.get('server_calls')} rec={r.get('recorded')} "
              f"loc={r.get('located')} in={r.get('input_rec')}/{r.get('input_truth')} "
              f"out={r.get('output_rec')}/{r.get('output_truth')} {r.get('sdks', '')}")
