import json
from pathlib import Path

from trainlab_optimize.importers.cloud_csv import detect, read_cloud_csv
from trainlab_optimize.importers.csv_generic import read_csv
from trainlab_optimize.importers.traces import read_langfuse, read_otel

FX = Path(__file__).parent / "fixtures"


def test_aws_cur():
    recs = read_csv(FX / "aws_cur.csv")          # auto-detected through the generic CSV entry point
    assert [(r.provider, r.model, r.input_tokens, r.output_tokens, r.cost_usd) for r in recs] == [
        ("bedrock", "Claude3.5Sonnet", 1_200_000, 0, 3.60), ("bedrock", "Claude3.5Sonnet", 0, 150_000, 2.25)]


def test_azure_cost():
    recs = read_csv(FX / "azure_cost.csv")
    assert [(r.date, r.model, r.input_tokens, r.output_tokens, r.cost_usd, r.project) for r in recs] == [
        ("2026-09-01", "azure/gpt-4o-0806", 5_000_000, 0, 12.5, "support-bot"),
        ("2026-09-01", "azure/gpt-4o-0806", 0, 800_000, 8.0, "support-bot")]


def test_gcp_billing():
    recs = read_csv(FX / "gcp_billing.csv")
    assert [(r.model, r.input_tokens, r.output_tokens, r.cost_usd) for r in recs] == [
        ("gemini-2.5-flash", 0, 0, 12.0), ("gemini-2.5-flash", 0, 0, 10.0)]
    assert detect(["service.description", "sku.description", "cost"]) == "gcp"


def test_langfuse_no_prompt_text():
    recs = read_langfuse(FX / "langfuse_obs.json")
    assert sorted((r.model, r.input_tokens, r.output_tokens, round(r.cost_usd, 5), r.code_path, r.code_line)
                  for r in recs) == [
        ("claude-sonnet-4-5", 2500, 400, 0.0135, None, None),
        ("gpt-4o-2024-08-06", 1500, 300, 0.00675, "/app/services/invoice_extractor.py", 19)]
    assert "SECRET" not in json.dumps([r.to_dict() for r in recs])


def test_otel_spans():
    recs = read_otel(FX / "otel_spans.json")
    assert [(r.provider, r.model, r.input_tokens, r.cached_input_tokens, r.project, r.code_line) for r in recs] == [
        ("openai", "gpt-4o", 1500, 1024, "support-api", 34)]
    assert "SECRET" not in json.dumps([r.to_dict() for r in recs])


def test_trace_location_gives_exact_attribution():
    from datetime import date
    from trainlab_optimize.analyzer.engine import analyze
    from trainlab_optimize.scanner import scan
    root = Path(__file__).resolve().parents[1] / "examples/demo_app"
    a = analyze(scan(root), read_otel(FX / "otel_spans.json"), today=date(2026, 10, 14))
    exact = [x for x in a.attributions if x.method == "exact_trace"]
    sites = {s.id: s for s in a.scan.call_sites}
    assert [(sites[x.call_site_id].path, sites[x.call_site_id].line) for x in exact] == [
        ("services/invoice_extractor.py", 34)]


def test_billing_plus_traces_not_double_counted():
    from trainlab_optimize.analyzer.spend import summarize
    from trainlab_optimize.pricing import PriceCatalog
    from trainlab_optimize.schema import UsageRecord
    cat = PriceCatalog.load()
    billing = [UsageRecord(date=f"2026-09-{d:02d}", provider="openai", model="gpt-4o", cost_usd=100.0,
                           requests=1000, source="openai_usage_api") for d in range(1, 31)]
    traces = [UsageRecord(date="2026-09-01", provider="openai", model="gpt-4o", input_tokens=1000,
                          output_tokens=100, requests=10, source="otel:x.json",
                          code_path="/app/a.py", code_line=10),
              UsageRecord(date="2026-09-01", provider="openai", model="gpt-4o", input_tokens=1000,
                          output_tokens=100, requests=30, source="otel:x.json")]
    s = summarize(billing + traces, cat)
    assert round(s.monthly_total) == 3000                     # billing only, traces not added
    (key, (cost, req)), = s.by_location.items()
    assert round(cost) == 1500                                # half the traced cost -> half the bill
    assert s.days == 30


def test_auto_detect_every_supported_file():
    from trainlab_optimize.importers.auto import detect_kind, read_any
    demo = Path(__file__).parent.parent / "examples" / "demo_usage"
    expected = {
        FX / "aws_cur.csv": "aws", FX / "azure_cost.csv": "azure", FX / "gcp_billing.csv": "gcp",
        FX / "langfuse_obs.json": "langfuse", FX / "otel_spans.json": "otel",
        demo / "billing_export.csv": "csv", demo / "anthropic_usage.json": "anthropic-file",
        demo / "openai_usage_completions.json": "openai-file", demo / "openai_usage_embeddings.json": "openai-file",
    }
    for path, kind in expected.items():
        assert detect_kind(path) == kind, path
        records, got = read_any(path)
        assert got == kind and records, path
    emb, _ = read_any(demo / "openai_usage_embeddings.json")
    assert all(r.output_tokens == 0 for r in emb) and any("embedding" in r.model for r in emb)


def test_cli_import_by_path(tmp_path, capsys):
    from trainlab_optimize.cli import main
    ws = str(tmp_path / "ws")
    demo = Path(__file__).parent.parent / "examples" / "demo_usage"
    assert main(["--workspace", ws, "import", str(demo / "anthropic_usage.json")]) == 0
    assert "Anthropic usage API file" in capsys.readouterr().out
    # a JSON file handed to `import csv` is routed to detection instead of failing
    assert main(["--workspace", ws, "import", "csv", str(demo / "openai_usage_completions.json")]) == 0
    assert main(["--workspace", ws, "import", str(tmp_path / "missing.csv")]) == 2
    # re-importing the same file replaces its rows instead of doubling them
    def count():
        data = json.loads((tmp_path / "ws" / "usage.json").read_text())
        return len(data if isinstance(data, list) else data["records"])
    n = count()
    main(["--workspace", ws, "import", str(demo / "anthropic_usage.json")])
    assert count() == n


def test_overlapping_billing_sources_counted_once():
    from trainlab_optimize.analyzer.spend import summarize
    from trainlab_optimize.importers.auto import read_any
    from trainlab_optimize.pricing import PriceCatalog
    demo = Path(__file__).parent.parent / "examples" / "demo_usage"
    cat = PriceCatalog.load()
    csv_only, _ = read_any(demo / "billing_export.csv")
    api = []
    for name in ("openai_usage_completions.json", "openai_usage_embeddings.json", "anthropic_usage.json"):
        api += read_any(demo / name)[0]
    one = summarize(csv_only, cat).monthly_total
    both = summarize(csv_only + api, cat)
    assert abs(both.monthly_total - one) / one < 0.01      # same traffic, not doubled
    assert both.overlaps and "billing_export.csv" in both.overlaps[0]
    # a source covering *different* models is still added
    extra = [r for r in api if "claude" in r.model]
    for r in extra:
        r.model = "claude-opus-4-1"
    assert summarize(csv_only + extra, cat).monthly_total > one


def test_openrouter_activity_file_and_parse():
    from trainlab_optimize.importers.auto import read_any
    recs, kind = read_any(FX / "openrouter_activity.json")
    assert kind == "openrouter-file"
    assert [(r.provider, r.model, r.input_tokens, r.cached_input_tokens, r.cost_usd) for r in recs] == [
        ("openrouter", "anthropic/claude-sonnet-4.5", 3_000_000, 1_000_000, 12.5),
        ("openrouter", "qwen/qwen3-235b-a22b", 2_000_000, 0, 1.0)]       # BYOK inference counts as spend


def test_openrouter_importer_splits_by_key(monkeypatch):
    from trainlab_optimize.importers import openrouter
    activity = json.loads((FX / "openrouter_activity.json").read_text())
    calls = []

    def fake(url, params, headers):
        calls.append((url, dict(params)))
        if url.endswith("/keys"):
            return {"data": [{"hash": "h1", "name": "support-bot"}, {"hash": "h2", "name": "analytics"}]
                    if dict(params)["offset"] == "0" else []}
        return {"data": activity["data"][:1]} if dict(params)["api_key_hash"] == "h1" else {"data": activity["data"][1:]}
    monkeypatch.setattr(openrouter, "http_get_json", fake)
    recs = openrouter.OpenRouterImporter("mgmt").fetch()
    assert [(r.api_key, r.model) for r in recs] == [("support-bot", "anthropic/claude-sonnet-4.5"),
                                                   ("analytics", "qwen/qwen3-235b-a22b")]


def test_openrouter_csv_export_aggregates_rows_per_key():
    recs = read_csv(FX / "openrouter_export.csv")
    got = sorted((r.provider, r.model, r.api_key, r.input_tokens, r.requests, round(r.cost_usd, 4)) for r in recs)
    assert got == [("openrouter", "anthropic/claude-sonnet-4.5", "analytics", 2000, 1, 0.05),
                   ("openrouter", "openai/gpt-4o-mini", "support-bot", 2200, 2, 0.0225)]


def test_unfamiliar_csv_needs_map_then_imports():
    import pytest
    from trainlab_optimize.importers.csv_generic import parse_map
    with pytest.raises(ValueError, match="Billing Item"):        # error lists the file's columns
        read_csv(FX / "alibaba_bill.csv")
    recs = read_csv(FX / "alibaba_bill.csv", column_map=parse_map(
        'model=Billing Item,cost_usd=Pretax Amount (USD),date=Billing Date,api_key=Instance ID'))
    assert sum(r.cost_usd for r in recs) == 3.92 and {r.api_key for r in recs} == {"ak-123"}
    assert sorted(r.model for r in recs) == ["qwen-max", "qwen-plus"]


def test_same_model_on_different_channels_is_not_an_overlap():
    from trainlab_optimize.analyzer.spend import summarize
    from trainlab_optimize.pricing import PriceCatalog
    from trainlab_optimize.schema import UsageRecord
    direct = UsageRecord(date="2026-09-01", provider="anthropic", model="claude-sonnet-4-5", input_tokens=1_000_000,
                         source="anthropic_usage_api")
    routed = UsageRecord(date="2026-09-01", provider="openrouter", model="anthropic/claude-sonnet-4.5",
                         input_tokens=1_000_000, cost_usd=3.0, source="openrouter_api")
    azure = UsageRecord(date="2026-09-01", provider="azure", model="azure/gpt-4o", input_tokens=1_000_000,
                        cost_usd=2.5, source="azure:cost.csv")
    openai = UsageRecord(date="2026-09-01", provider="openai", model="gpt-4o", input_tokens=1_000_000,
                         source="openai_usage_api")
    s = summarize([direct, routed, azure, openai], PriceCatalog.load())
    assert not s.overlaps
    assert round(s.monthly_total / 30, 2) == 11.0          # 3 + 3 + 2.5 + 2.5, all counted


def test_spend_by_api_key():
    from trainlab_optimize.analyzer.spend import summarize
    from trainlab_optimize.pricing import PriceCatalog
    recs = read_csv(FX / "openrouter_export.csv")
    s = summarize(recs, PriceCatalog.load())
    assert set(s.by_api_key) == {"support-bot", "analytics"}
    assert s.by_api_key["analytics"] > s.by_api_key["support-bot"]
