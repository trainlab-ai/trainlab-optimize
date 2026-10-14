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
    ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "anthropic.claude-haiku-4-5"),
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
    assert analysis.savings_high == pytest.approx(12412, rel=0.01)


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
