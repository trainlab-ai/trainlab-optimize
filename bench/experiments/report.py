"""bench/experiments/results/results.json -> bench/experiments/RESULTS.md"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run import population_rate  # noqa: E402

d = json.loads((HERE / "results" / "results.json").read_text())
rows = d["rows"]
C = collections.Counter
NAMES = {"X01": "Run completes", "X02": "Cost = zoo token log x catalog price", "X03": "Report shows the result",
         "X04": "No PASS for a model below the bar", "X05": "No FAIL for a model clearly above the bar",
         "X06": "Every sample's pass/fail = truth", "X07": "Data only to configured endpoints, no key on disk",
         "X08": "Injected 429/500 all recovered", "X09": "Free text without a judge never validates",
         "X10": "Saving % = measured cost per call", "X11": "Resume never pays twice",
         "X12": "Budget never exceeded",
         "X13": "CLI refuses runs that can't give a verdict, before spending"}
cnt = C()
for r in rows:
    for k, c in r.get("checks", {}).items():
        cnt[(k, c["status"])] += 1
cands = [(r, c) for r in rows for c in r.get("candidates_result", [])]


def band(r, c):
    rate = population_rate(r["task"], r["mode"], r["design"][c["model"]], 0.995 if r["task"] != "generation" else 0.98)
    thr = r["result"]["threshold"]
    return "≥ bar + 3 pts" if rate >= thr + 0.03 else "≤ bar − 3 pts" if rate <= thr - 0.03 else "within 3 pts", rate, thr


out = ["# `trainlab experiment` on 100 repositories", "",
       "Every repository of the frozen 100-repo suite (`bench/repos.lock.json`) gets one or two experiments: a",
       "call site `trainlab experiment plan` proposes (auto), the repo's top call site with its billed model named",
       "by hand (explicit), or, for repos with no LLM code, a bare CSV dataset run through the CLI with no scan",
       "(dataset). Models answer from a simulated zoo that speaks the OpenAI, Azure, Anthropic, Gemini and Bedrock",
       "formats; each candidate has a designed accuracy, so the right verdict is known. The zoo injects 429/500",
       "errors and logs every request. No real API key is involved.", "",
       f"**{len(rows)} experiments on {len({r['repo'] for r in rows})} repositories, {len(cands)} candidate models, "
       f"{d['seconds']} s.**", "",
       "| Check | Pass | Fail | Not applicable |", "|---|---|---|---|"]
for k in sorted(NAMES):
    out.append(f"| {k} {NAMES[k]} | {cnt[(k, 'pass')]} | {cnt[(k, 'fail')]} | {cnt[(k, 'n/a')]} |")
out += ["", "## Verdicts against the truth", "",
        "The bar is the threshold the run used (95% by default, 90% for free text, lowered to the current model's",
        "own accuracy minus 2 points when expected answers show it is below that).", "",
        "| Candidate's true pass rate | PASS | INCONCLUSIVE | FAIL | NEEDS-JUDGE |", "|---|---|---|---|---|"]
grid = C()
for r, c in cands:
    grid[(band(r, c)[0], c["verdict"])] += 1
for b in ("≥ bar + 3 pts", "within 3 pts", "≤ bar − 3 pts"):
    out.append(f"| {b} | {grid[(b, 'PASS')]} | {grid[(b, 'INCONCLUSIVE')]} | {grid[(b, 'FAIL')]} | {grid[(b, 'NEEDS-JUDGE')]} |")
below = sum(1 for r, c in cands if c["verdict"] == "PASS" and band(r, c)[1] < band(r, c)[2])
early = [c for _, c in cands if c.get("stopped_early")]
out += ["", f"PASS given to a model whose true rate is below the bar, by any margin: **{below}**. "
        f"Candidates stopped early once clearly failing: {len(early)} (median {sorted(c['scored'] for c in early)[len(early) // 2] if early else 0} samples).", "",
        "## Coverage", "",
        f"- Experiment kinds: {', '.join(f'{k} {v}' for k, v in C(r['kind'] for r in rows).items())}",
        f"- Tasks: {', '.join(f'{k} {v}' for k, v in C(r['task'] for r in rows).items())}",
        f"- Reference: {', '.join(f'{k} {v}' for k, v in C({'reference': 'production outputs', 'fresh': 'baseline called', 'gold': 'expected answers'}[r['mode']] for r in rows).items())}",
        f"- Free-text runs with a judge: {sum(1 for r in rows if r['task'] == 'generation' and r['judge'])}, without: "
        f"{sum(1 for r in rows if r['task'] == 'generation' and not r['judge'])}",
        f"- Through the CLI: {sum(1 for r in rows if r.get('cli'))}; budget-cut runs: "
        f"{sum(1 for r in rows if r.get('stress') == 'budget')}; resumed runs: {sum(1 for r in rows if r.get('stress') == 'resume')}",
        "", "## Per experiment", "",
        "| Repository | Kind | Task | Reference | Current model | Candidates (verdict, pass/scored) | Checks |",
        "|---|---|---|---|---|---|---|"]
for r in rows:
    vs = "; ".join(f"`{c['model']}` {c['verdict']} {c['passed']}/{c['scored']}" for c in r.get("candidates_result", []))
    bad = [k for k, c in r.get("checks", {}).items() if c["status"] == "fail"]
    out.append(f"| {r['repo']} | {r['kind']} | {r['task']} | {r['mode']} | `{r['baseline']}` | {vs} | "
               f"{'all pass' if not bad else 'FAIL ' + ', '.join(bad)} |")
out += ["", "Reproduce: `python bench/experiments/prepare.py WORK && python bench/experiments/run.py WORK && "
        "python bench/experiments/report.py`.", ""]
(HERE / "RESULTS.md").write_text("\n".join(out))
print("\n".join(out[:32]))
