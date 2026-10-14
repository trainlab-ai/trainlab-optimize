import json
from datetime import date
from pathlib import Path

import pytest

from trainlab_optimize.analyzer.engine import analyze
from trainlab_optimize.analyzer.spend import base_model
from trainlab_optimize.importers import anthropic_usage, openai_usage
from trainlab_optimize.importers.csv_generic import read_csv
from trainlab_optimize.pricing import PriceCatalog
from trainlab_optimize.redact import redact
from trainlab_optimize.report import write_reports
from trainlab_optimize.scanner import scan

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples/demo_app"
USAGE = ROOT / "examples/demo_usage"
TODAY = date(2026, 10, 14)


def api_records():
    recs = openai_usage.parse_pages(json.loads((USAGE / "openai_usage_completions.json").read_text()))
    recs += openai_usage.parse_pages(json.loads((USAGE / "openai_usage_embeddings.json").read_text()),
                                     kind="embeddings")
    recs += anthropic_usage.parse_pages(json.loads((USAGE / "anthropic_usage.json").read_text()))
    return recs


@pytest.fixture(scope="module")
def analysis():
    return analyze(scan(DEMO), api_records(), today=TODAY)


@pytest.mark.parametrize("raw,base", [
    ("gpt-4o-2024-08-06", "gpt-4o"), ("openai/gpt-4o", "gpt-4o"), ("gpt-4-0613", "gpt-4"),
    ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5"),
    ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "claude-haiku-4-5"),
    ("gpt-4-1106-preview", "gpt-4"),
])
def test_base_model(raw, base):
    assert base_model(raw) == base


def test_catalog_lookups():
    c = PriceCatalog.load()
    assert c.lookup("gpt-4o-2024-08-06").input > 0
    assert c.lookup("claude-haiku-4-5") is not None
    assert c.lookup("not-a-model") is None


def test_api_and_csv_import_agree():
    cat = PriceCatalog.load()
    from trainlab_optimize.analyzer.spend import summarize
    a = summarize(api_records(), cat).monthly_total
    b = summarize(read_csv(USAGE / "billing_export.csv"), cat).monthly_total
    assert a == pytest.approx(b, rel=1e-9) and a > 20000


def test_attribution(analysis):
    cov = analysis.coverage
    assert list(cov["unattributed_by_model"]) == ["text-embedding-3-large"]
    methods = {a.method for a in analysis.attributions}
    assert methods == {"exact_model", "split_model"}


def test_expected_findings(analysis):
    rules = {(f.rule, f.title.split("`")[1] if "`" in f.title else f.title) for f in analysis.findings}
    assert ("R101", "gpt-4o") in rules
    assert ("R101", "gpt-4.1") in rules
    assert ("R102", "gpt-4") in rules
    assert any(r == "R103" for r, _ in rules)
    assert any(r == "R106" for r, _ in rules)
    slm = [f for f in analysis.findings if f.trainlab_candidate]
    assert len(slm) == 2


def test_negative_control_has_no_findings(analysis):
    clean = [s.id for s in analysis.scan.call_sites if s.path == "services/well_tuned.py"]
    assert clean and not any(set(clean) & set(f.call_site_ids) for f in analysis.findings)


def test_savings_not_double_counted(analysis):
    assert analysis.savings_high < analysis.spend.monthly_total
    assert analysis.savings_low <= analysis.savings_high
    assert analysis.savings_high == pytest.approx(12650, rel=0.01)


def test_code_only_report(tmp_path):
    a = analyze(scan(DEMO), [], today=TODAY)
    assert a.savings_high is None
    assert all(f.tier == "static" for f in a.findings)
    paths = write_reports(a, tmp_path)
    html = (tmp_path / "trainlab-report.html").read_text()
    assert "No usage imported yet" in html and len(paths) == 2


def test_report_has_no_external_requests(analysis, tmp_path):
    write_reports(analysis, tmp_path)
    html = (tmp_path / "trainlab-report.html").read_text()
    assert "http://" not in html and "https://" not in html.replace(
        "https://raw.githubusercontent.com", "")  # catalog source is plain text, not a request
    assert "<script" not in html and "<link" not in html


def test_redaction():
    out = redact("mail bob@acme.com key sk-proj-ABCDEFGHIJKLMNOPQRST call +1 415 555 0100")
    assert "bob@acme.com" not in out and "sk-proj" not in out and "555" not in out


def test_kwargs_passthrough_not_flagged_for_output_cap():
    from trainlab_optimize.scanner.python_ast import PythonFileScanner
    from trainlab_optimize.scanner import ScanResult
    src = "from openai import OpenAI\nc = OpenAI()\ndef f(m, **kw):\n    return c.chat.completions.create(model=m, messages=[], **kw)\n"
    r = ScanResult(root="x"); r.call_sites = PythonFileScanner("a.py", src).scan()
    assert not [f for f in analyze(r, [], today=TODAY).findings if f.rule == "R105"]


def _analyze_src(src, path="a.py"):
    from trainlab_optimize.scanner import ScanResult
    from trainlab_optimize.scanner.python_ast import PythonFileScanner
    r = ScanResult(root="x"); r.call_sites = PythonFileScanner(path, src).scan()
    return analyze(r, [], today=TODAY)


def test_while_loop_not_batch_candidate():
    src = "from openai import OpenAI\nc = OpenAI()\ndef agent_loop():\n    while True:\n        c.chat.completions.create(model='gpt-5-mini', messages=[], max_tokens=5)\n"
    assert not [f for f in _analyze_src(src).findings if f.rule == "R104"]


def test_gemini_config_not_flagged_for_output_cap():
    src = "from google import genai\nc = genai.Client()\nc.models.generate_content(model='gemini/gemini-3.5-flash-lite', contents='x', config=cfg)\n"
    assert not [f for f in _analyze_src(src).findings if f.rule == "R105"]


def test_deprecation_requires_all_family_entries():
    # 'gemini-2.5-flash' is deprecated on Vertex in the catalog but not on the Gemini API
    src = "from google import genai\nc = genai.Client()\nc.models.generate_content(model='gemini-2.5-flash', contents='x')\n"
    assert not [f for f in _analyze_src(src).findings if f.rule == "R102"]


def test_example_scope_capped_low():
    src = "from openai import OpenAI\nc = OpenAI()\nc.chat.completions.create(model='gpt-4', messages=[])\n"
    sev = {f.rule: f.severity for f in _analyze_src(src, "examples/x.py").findings}
    assert sev["R102"] == "low"


@pytest.mark.parametrize("a,b", [
    ("Claude3.5Sonnet", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
    ("claude-3-5-sonnet-20241022", "claude-3.5-sonnet"),
    ("claude-sonnet-4-5-20250929", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
    ("Claude4.5Sonnet", "claude-sonnet-4-5"),
    ("azure/gpt-4o-0806", "gpt-4o"),
])
def test_billing_and_code_names_meet(a, b):
    assert base_model(a) == base_model(b)


def test_r107_r108_r109(tmp_path):
    from trainlab_optimize.schema import UsageRecord
    big = "You are a meticulous analyst. " * 60
    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text(f'''from openai import OpenAI
c = OpenAI()
PROMPT = "{big}"
def run(q):
    return c.chat.completions.create(model="gpt-5-mini", max_tokens=50,
        messages=[{{"role": "system", "content": PROMPT}}, {{"role": "user", "content": q}}])
def index(docs):
    return [c.embeddings.create(model="text-embedding-3-small", input=d) for d in docs]
''')
    usage = [UsageRecord(date=f"2026-09-{d:02d}", provider="openai", model="gpt-5-mini", input_tokens=20_000_000,
                         output_tokens=100_000, requests=1000, source="api") for d in range(1, 31)]
    a = analyze(scan(tmp_path), usage, today=TODAY)
    rules = {f.rule for f in a.findings}
    assert {"R107", "R108", "R109"} <= rules
    r108 = next(f for f in a.findings if f.rule == "R108")
    assert r108.evidence["avg_input_tokens"] == 20000 and r108.savings_high_usd > 0


def test_multi_repo_and_markdown(tmp_path):
    from trainlab_optimize.report.markdown import render_markdown
    a_dir, b_dir = tmp_path / "svc-a", tmp_path / "svc-b"
    for d, model in ((a_dir, "gpt-4"), (b_dir, "gpt-4o")):
        d.mkdir()
        (d / "app.py").write_text(f"from openai import OpenAI\nOpenAI().chat.completions.create(model='{model}', messages=[], max_tokens=10)\n")
    merged = scan(a_dir).merge(scan(b_dir))
    assert merged.repos == ["svc-a", "svc-b"]
    assert sorted(s.location for s in merged.call_sites) == ["svc-a/app.py:2", "svc-b/app.py:2"]
    md = render_markdown(analyze(merged, [], today=TODAY))
    assert "svc-a, svc-b" in md and "`svc-a/app.py:2`" in md
    paths = write_reports(analyze(merged, [], today=TODAY), tmp_path / "out", formats=("md",))
    assert paths[0].name == "trainlab-report.md"


def test_unknown_format_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_reports(analyze(scan(DEMO), [], today=TODAY), tmp_path, formats=("docx",))


def test_shared_spend_across_repos_is_flagged(tmp_path):
    from trainlab_optimize.schema import UsageRecord
    for name in ("api", "worker"):
        d = tmp_path / name
        d.mkdir()
        (d / "app.py").write_text('from openai import OpenAI\nc = OpenAI()\n'
                                  'c.chat.completions.create(model="gpt-4o", messages=[], max_tokens=50)\n')
    sc = scan(tmp_path / "api").merge(scan(tmp_path / "worker"))
    usage = [UsageRecord(date=f"2026-09-{d:02d}", provider="openai", model="gpt-4o", input_tokens=10_000_000,
                         output_tokens=1_000_000, api_key="key_prod", source="openai_usage_api") for d in range(1, 31)]
    a = analyze(sc, usage, today=TODAY)
    shared = [f for f in a.findings if f.rule == "R904"]
    assert len(shared) == 1 and shared[0].evidence["repos"] == ["api", "worker"]
    assert shared[0].evidence["api_keys"] == ["key_prod"]
    assert a.to_dict()["spend"]["by_api_key"]["key_prod"]["models"] == ["gpt-4o"]
