"""trainlab command-line interface."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .analyzer.engine import analyze
from .importers import ImportError_
from .pricing import PriceCatalog
from .report import write_reports
from .scanner import scan
from .workspace import Workspace


def _ws(args) -> Workspace:
    return Workspace(args.workspace)


def cmd_scan(args) -> int:
    paths = args.path if isinstance(args.path, list) else [args.path]
    result = None
    if getattr(args, "append", False):
        result = _ws(args).load_scan()
    for path in paths:
        if not Path(path).is_dir():
            print(f"Not a directory: {path}", file=sys.stderr)
            return 2
        r = scan(path, include_tests=args.include_tests, exclude=args.exclude or [])
        result = r if result is None else result.merge(r)
    p = _ws(args).save_scan(result)
    by_kind: dict[str, int] = {}
    for s in result.call_sites:
        by_kind[s.kind] = by_kind.get(s.kind, 0) + 1
    unresolved = sum(1 for s in result.call_sites if s.kind in ("chat", "embedding") and not s.model)
    repos = f" across {len(result.repos)} repositories" if result.repos else ""
    print(f"Scanned {result.files_scanned} files{repos} in {result.duration_s:.1f}s: "
          f"{len(result.call_sites)} LLM call sites {by_kind or ''}")
    if unresolved:
        print(f"  {unresolved} call site(s) choose the model at runtime")
    if result.model_refs:
        print(f"  {len(result.model_refs)} model name(s) referenced in config")
    if result.parse_errors:
        print(f"  {len(result.parse_errors)} Python file(s) could not be parsed (skipped)")
    print(f"Saved to {p}")
    return 0


API_KEY_ENV = {"openai": "OPENAI_ADMIN_KEY", "anthropic": "ANTHROPIC_ADMIN_KEY",
               "openrouter": "OPENROUTER_MANAGEMENT_KEY"}
API_KEY_KIND = {"openai": "an organization admin key", "anthropic": "an organization admin key",
                "openrouter": "a management key (Settings > Management keys)"}


def _import_api(args, provider: str) -> int:
    from .importers import anthropic_usage, openai_usage, openrouter
    if args.file:
        name = Path(args.file).name
        try:
            data = json.loads(Path(args.file).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            print(f"Import failed: {exc}", file=sys.stderr)
            return 1
        pages = data if isinstance(data, list) else [data]
        if provider == "openai":
            records = openai_usage.parse_pages(pages, kind="embeddings" if args.embeddings else "completions")
        elif provider == "anthropic":
            records = anthropic_usage.parse_pages(pages)
        else:
            records = openrouter.parse_activity(data)
        for r in records:
            r.source = f"{provider}_usage_file:{name}"
    else:
        env = API_KEY_ENV[provider]
        key = args.key or os.environ.get(env)
        if not key:
            print(f"No key. Set {env} or pass --key. The usage API needs {API_KEY_KIND[provider]} "
                  "(read-only usage access), not the key your app uses.", file=sys.stderr)
            return 2
        importer = {"openai": lambda: openai_usage.OpenAIUsageImporter(key),
                    "anthropic": lambda: anthropic_usage.AnthropicUsageImporter(key),
                    "openrouter": lambda: openrouter.OpenRouterImporter(key)}[provider]()
        try:
            records = importer.fetch(days=args.days)
        except ImportError_ as exc:
            print(f"Import failed: {exc}", file=sys.stderr)
            return 1
    n = _ws(args).add_usage(records)
    models = sorted({r.model for r in records})
    keys = sorted({r.api_key for r in records if r.api_key})
    print(f"Imported {n} usage records for {len(models)} model(s) from {provider}"
          + (f", across {len(keys)} API key(s)" if keys else "") + ".")
    return 0


IMPORT_SOURCES = ("file", "openai", "anthropic", "openrouter", "csv", "langfuse", "otel", "aws", "azure", "gcp")
KIND_NAMES = {"csv": "billing CSV", "aws": "AWS cost export", "azure": "Azure cost export", "gcp": "Google Cloud billing export",
              "langfuse": "Langfuse export", "otel": "OpenTelemetry traces", "openai-file": "OpenAI usage API file",
              "anthropic-file": "Anthropic usage API file", "openrouter-file": "OpenRouter activity file",
              "trainlab": "trainlab instrument() traces"}


def _import_file_auto(args) -> int:
    from .importers.auto import read_any
    from .importers.csv_generic import parse_map
    try:
        records, kind = read_any(args.file, "auto", default_provider=getattr(args, "provider", None),
                                 column_map=parse_map(getattr(args, "map", None)) or None)
    except (ValueError, OSError, KeyError) as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        return 1
    if not records:
        print(f"No LLM usage found in {args.file}. Check the service and period it covers.", file=sys.stderr)
        return 1
    n = _ws(args).add_usage(records)
    located = sum(1 for r in records if r.code_path)
    print(f"Imported {n} usage records from {args.file} (detected: {KIND_NAMES.get(kind, kind)})"
          + (f"; {located} carry source locations for exact per-line attribution" if located else "") + ".")
    return 0


def cmd_import(args) -> int:
    args.file = getattr(args, "file_opt", None) or getattr(args, "path_arg", None) or getattr(args, "file", None)
    if args.source not in IMPORT_SOURCES:
        # `trainlab import billing.csv` -> detect the file type
        if Path(args.source).is_file() or Path(args.source).is_dir():
            args.file, args.source = args.source, "file"
        else:
            print(f"'{args.source}' is neither a file nor a source ({', '.join(IMPORT_SOURCES)}).", file=sys.stderr)
            return 2
    if args.source == "file" or (args.source == "csv" and args.file
                                 and Path(args.file).suffix.lower() in (".json", ".jsonl", ".ndjson")):
        if not args.file:
            print("Pass the file: trainlab import <file>", file=sys.stderr)
            return 2
        return _import_file_auto(args)
    if args.source in ("openai", "anthropic", "openrouter"):
        return _import_api(args, args.source)
    if args.source in ("langfuse", "otel", "aws", "azure", "gcp"):
        if not args.file:
            print(f"Pass the export file: trainlab import {args.source} <file>", file=sys.stderr)
            return 2
        try:
            if args.source == "langfuse":
                from .importers.traces import read_langfuse
                records = read_langfuse(args.file)
            elif args.source == "otel":
                from .importers.traces import read_otel
                records = read_otel(args.file)
            else:
                from .importers.cloud_csv import read_cloud_csv
                records = read_cloud_csv(args.file, args.source)
        except (ValueError, OSError, KeyError) as exc:
            print(f"Import failed: {exc}", file=sys.stderr)
            return 1
        n = _ws(args).add_usage(records)
        located = sum(1 for r in records if r.code_path)
        print(f"Imported {n} usage records from {args.file}"
              + (f" ({located} with source locations: exact per-line attribution)" if located else "") + ".")
        if not records:
            print("  No LLM usage rows found. Check that the export covers the right service and period.")
        return 0
    if args.source == "csv":
        from .importers.csv_generic import read_csv
        if not args.file:
            print("Pass the CSV path: trainlab import csv billing.csv", file=sys.stderr)
            return 2
        try:
            from .importers.csv_generic import parse_map
            records = read_csv(args.file, default_provider=args.provider,
                               column_map=parse_map(getattr(args, "map", None)) or None)
        except (ValueError, OSError) as exc:
            print(f"Import failed: {exc}", file=sys.stderr)
            return 1
        n = _ws(args).add_usage(records)
        print(f"Imported {n} usage rows from {args.file}.")
        return 0
    return 2


def _print_summary(a) -> None:
    if a.spend.has_usage:
        print(f"Monthly LLM spend: ${a.spend.monthly_total:,.0f} ({a.spend.days} day{"s" if a.spend.days != 1 else ""} of data)")
        for o in a.spend.overlaps:
            print(f"  Note: {o}")
        if a.savings_high:
            print(f"Potential saving (estimated, untested): ${a.savings_low:,.0f} – ${a.savings_high:,.0f}/month")
    else:
        print("No usage imported: findings are from code only. Run `trainlab import ...` to price them.")
    shown = [f for f in a.findings if f.strategy != "coverage"][:8]
    for f in shown:
        save = f"  save up to ${f.savings_high_usd:,.0f}/mo" if f.savings_high_usd else ""
        print(f"  [{f.severity:<6}] {f.id} {f.title.replace('`', '')}{save}")
    rest = len([f for f in a.findings if f.strategy != "coverage"]) - len(shown)
    if rest > 0:
        print(f"  ... and {rest} more in the report")


def cmd_report(args) -> int:
    ws = _ws(args)
    result = ws.load_scan()
    if result is None:
        print("No scan found. Run `trainlab scan <path>` first.", file=sys.stderr)
        return 2
    from .experiment.runner import load_results
    a = analyze(result, ws.load_usage(), experiments=load_results(ws.path))
    out = Path(args.out) if args.out else ws.path / "report"
    try:
        paths = write_reports(a, out, formats=tuple(f.strip() for f in args.format.split(",") if f.strip()))
    except Exception as exc:   # PdfError, ValueError
        print(f"Report failed: {exc}", file=sys.stderr)
        return 1
    _print_summary(a)
    for p in paths:
        print(f"Wrote {p}")
    from .report import warnings as report_warnings
    for w in report_warnings:
        print(f"Warning: {w}", file=sys.stderr)
    fail_on = getattr(args, "fail_on", None)
    if fail_on and fail_on != "none":
        order = {"high": 0, "medium": 1, "low": 2}
        hits = [f for f in a.findings if f.strategy != "coverage"
                and order.get(f.severity, 9) <= order[fail_on]]
        if hits:
            print(f"{len(hits)} finding(s) at or above '{fail_on}' severity.", file=sys.stderr)
            return 3
    return 0


def cmd_run(args) -> int:
    rc = cmd_scan(args)
    if rc:
        return rc
    for f in args.usage_csv or []:
        args.source, args.file, args.provider = "file", f, None
        _import_file_auto(args)
    return cmd_report(args)


def cmd_ui(args) -> int:
    from .ui import run
    run(_ws(args), port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_prices(args) -> int:
    if args.action == "show":
        cat = PriceCatalog.load()
        p = cat.lookup(args.model)
        if not p:
            print(f"{args.model}: not in catalog ({len(cat)} models, built {cat.meta.get('built_at')})")
            return 1
        print(f"{p.model} ({p.provider}, {p.mode})")
        print(f"  input  ${p.input * 1e6:.3f} / 1M tokens")
        print(f"  output ${p.output * 1e6:.3f} / 1M tokens")
        if p.cache_read is not None:
            print(f"  cached input ${p.cache_read * 1e6:.3f} / 1M tokens")
        if p.batch_input:
            print(f"  batch input ${p.batch_input * 1e6:.3f} / 1M tokens")
        if p.deprecated:
            print(f"  deprecation date {p.deprecated}")
        return 0
    if args.action == "info":
        cat = PriceCatalog.load()
        print(json.dumps(cat.meta, indent=1))
        return 0
    if args.action == "update":
        from .catalog.build import update
        from .pricing import user_catalog_path
        try:
            cat = update(user_catalog_path())
        except OSError as exc:
            print(f"Could not download prices: {exc}. The bundled catalog stays in use.", file=sys.stderr)
            return 1
        print(f"Updated {cat['_meta']['model_count']} model prices ({cat['_meta']['built_at']}) "
              f"-> {user_catalog_path()}")
        return 0
    return 2


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="trainlab", description="Find and prove LLM cost savings in your code.")
    ap.add_argument("--version", action="version", version=f"trainlab-optimize {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workspace", default=argparse.SUPPRESS,
                        help="where scan results and usage are stored (default .trainlab)")
    ap.add_argument("--workspace", default=".trainlab", help="where scan results and usage are stored")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", parents=[common], help="scan a repository for LLM calls")
    s.add_argument("path", nargs="*", default=["."], help="one or more repository folders")
    s.add_argument("--append", action="store_true", help="add to the existing scan instead of replacing it")
    s.add_argument("--include-tests", action="store_true")
    s.add_argument("--exclude", action="append", metavar="GLOB",
                   help="skip paths, e.g. --exclude examples --exclude 'docs/*' (repeatable)")
    s.set_defaults(fn=cmd_scan)

    i = sub.add_parser("import", parents=[common], help="import usage/billing data")
    i.add_argument("source", metavar="source|file",
                   help="a file path (type detected automatically), or one of: openai/anthropic/openrouter "
                        "(usage API, needs an admin or management key), csv, langfuse, otel, aws, azure, gcp "
                        "(force a file type)")
    i.add_argument("path_arg", nargs="?", metavar="file", help="the export file when a source is named")
    i.add_argument("--file", dest="file_opt", help="read a saved usage API response instead of calling the API")
    i.add_argument("--key", help="admin/management key (default: OPENAI_ADMIN_KEY / ANTHROPIC_ADMIN_KEY / "
                                 "OPENROUTER_MANAGEMENT_KEY)")
    i.add_argument("--days", type=int, default=30)
    i.add_argument("--embeddings", action="store_true", help="the --file holds embeddings usage (OpenAI)")
    i.add_argument("--provider", help="default provider for CSV rows without one")
    i.add_argument("--map", metavar="FIELD=COLUMN,...",
                   help='name the columns of an unfamiliar CSV, e.g. --map model="Model",cost_usd="Amount (USD)"')
    i.set_defaults(fn=cmd_import)

    r = sub.add_parser("report", parents=[common], help="analyze and write the report")
    r.add_argument("--out", help="output directory (default .trainlab/report)")
    r.add_argument("--format", default="html,json", help="comma-separated: html,json,md,pdf")
    r.add_argument("--fail-on", choices=["none", "high", "medium", "low"], default="none",
                   help="exit with code 3 if findings at or above this severity exist (for CI)")
    r.set_defaults(fn=cmd_report)

    run = sub.add_parser("run", parents=[common], help="scan + import CSVs + report in one step")
    run.add_argument("path", nargs="*", default=["."])
    run.add_argument("--usage", "--usage-csv", dest="usage_csv", action="append", metavar="FILE",
                     help="usage/billing file to import (any supported type; repeatable)")
    run.add_argument("--include-tests", action="store_true")
    run.add_argument("--exclude", action="append", metavar="GLOB")
    run.add_argument("--out")
    run.add_argument("--format", default="html,json")
    run.add_argument("--fail-on", choices=["none", "high", "medium", "low"], default="none")
    run.set_defaults(fn=cmd_run)

    u = sub.add_parser("ui", parents=[common], help="open the local web interface")
    u.add_argument("--port", type=int, default=0, help="port on 127.0.0.1 (default: random free port)")
    u.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    u.set_defaults(fn=cmd_ui)

    p = sub.add_parser("prices", parents=[common], help="look up model prices")
    p.add_argument("action", choices=["show", "info", "update"])
    p.add_argument("model", nargs="?")
    p.set_defaults(fn=cmd_prices)

    from .experiment.cli import add_parser as add_experiment
    add_experiment(sub, common)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
