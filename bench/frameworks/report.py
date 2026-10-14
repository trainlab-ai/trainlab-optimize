"""bench/frameworks/results.json -> bench/frameworks/RESULTS.md"""
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())["results"]
KIND = {"sdk": "Provider SDKs", "raw-http": "Raw HTTP", "framework": "Frameworks and libraries",
        "otel": "Alongside existing tracing", "wrapper": "Alongside existing tracing"}

c = collections.Counter(r["status"] for r in R)
calls = sum(r.get("server_calls") or 0 for r in R)
rec = sum(r.get("recorded") or 0 for r in R)
loc = sum(r.get("located") or 0 for r in R)
out = ["# instrument() across 72 SDKs, frameworks and HTTP clients", "",
       "Each project's real installed code runs a short driver (`drivers/t_<name>.py`, written the way",
       "its docs show) against a local fake server that speaks the provider's wire format and logs every",
       "token it serves. Every driver runs twice, with trainlab off and on.", "",
       "| | |", "|---|---|",
       f"| Projects | {len(R)} |",
       f"| Pass (count, tokens and line all exact) | {c['PASS']} |",
       f"| Breaks the app or changes its calls | {c['BREAKS-APP'] + c['CHANGES-CALLS']} |",
       f"| LLM calls served / recorded | {calls} / {rec} |",
       f"| Recorded calls pointing at the app's own line | {loc} of {rec} |",
       f"| Other statuses | {', '.join(f'{k} {v}' for k, v in c.items() if k != 'PASS') or 'none'} |", ""]
for kind in ("sdk", "raw-http", "framework", "otel"):
    rows = [r for r in R if KIND[r["kind"]] == KIND[kind]]
    if not rows:
        continue
    out += [f"## {KIND[kind]}", "",
            "| Project | API | Calls (server / recorded) | Input tokens | Output tokens | App line | Recorded via | Result |",
            "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tok_in = "exact" if r["input_rec"] == r["input_truth"] else f"{r['input_rec']} / {r['input_truth']}"
        tok_out = "exact" if r["output_rec"] == r["output_truth"] else f"{r['output_rec']} / {r['output_truth']}"
        sdks = ", ".join(sorted({s.replace("openai.chat.completions.create", "openai chat")
                                 .replace("anthropic.messages.create", "anthropic messages") for s in r["sdks"]}))
        out.append(f"| {r['project']} | {r['api']} | {r['server_calls']} / {r['recorded']} | {tok_in} | {tok_out} | "
                   f"{r['located']}/{r['recorded']} | {sdks} | {'✅' if r['status'] == 'PASS' else r['status']} |")
    out.append("")
out += ["## Not included", "",
        "- **Marvin** (Prefect's 3.x alpha): its pinned dependencies don't import together, with or without",
        "  trainlab, so there is nothing to measure.",
        "- **Legacy `vertexai` SDK**: forces HTTPS to its endpoint, so it can't reach a local test server.",
        "  Vertex is tested through the current Google Gen AI SDK (`vertexai=True`).",
        "- **Legacy `google-generativeai`** is tested with `transport=\"rest\"`. Its default transport is gRPC,",
        "  which trainlab does not see; that SDK is deprecated in favour of `google-genai`.", "",
        "Reproduce: `python bench/frameworks/install.py ENVS && python bench/frameworks/run.py ENVS`.", ""]
(HERE / "RESULTS.md").write_text("\n".join(out))
print("\n".join(out[:14]))
