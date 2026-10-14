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
    result = scan(args.path, include_tests=args.include_tests, exclude=args.exclude or [])
    p = _ws(args).save_scan(result)
    by_kind: dict[str, int] = {}
    for s in result.call_sites:
        by_kind[s.kind] = by_kind.get(s.kind, 0) + 1
    unresolved = sum(1 for s in result.call_sites if s.kind in ("chat", "embedding") and not s.model)
    print(f"Scanned {result.files_scanned} files in {result.duration_s:.1f}s: "
          f"{len(result.call_sites)} LLM call sites {by_kind or ''}")
    if unresolved:
        print(f"  {unresolved} call site(s) choose the model at runtime")
    if result.model_refs:
        print(f"  {len(result.model_refs)} model name(s) referenced in config")
    if result.parse_errors:
        print(f"  {len(result.parse_errors)} Python file(s) could not be parsed (skipped)")
    print(f"Saved to {p}")
    return 0


def _import_api(args, provider: str) -> int:
    from .importers import anthropic_usage, openai_usage
    mod = openai_usage if provider == "openai" else anthropic_usage
    if args.file:
        pages = json.loads(Path(args.file).read_text())
        pages = pages if isinstance(pages, list) else [pages]
        if provider == "openai":
            records = mod.parse_pages(pages, kind="embeddings" if args.embeddings else "completions")
            for r in records:
                r.source = f"openai_usage_file:{Path(args.file).name}"
        else:
            records = mod.parse_pages(pages)
            for r in records:
                r.source = f"anthropic_usage_file:{Path(args.file).name}"
    else:
        env = "OPENAI_ADMIN_KEY" if provider == "openai" else "ANTHROPIC_ADMIN_KEY"
        key = args.key or os.environ.get(env)
        if not key:
            print(f"No admin key. Set {env} or pass --key. Usage APIs need an organization admin key "
                  "(read-only usage access), not a normal API key.", file=sys.stderr)
            return 2
        importer = (mod.OpenAIUsageImporter(key) if provider == "openai" else mod.AnthropicUsageImporter(key))
        try:
            records = importer.fetch(days=args.days)
        except ImportError_ as exc:
            print(f"Import failed: {exc}", file=sys.stderr)
            return 1
    n = _ws(args).add_usage(records)
    models = sorted({r.model for r in records})
    print(f"Imported {n} usage records for {len(models)} model(s) from {provider}.")
    return 0


def cmd_import(args) -> int:
    args.file = getattr(args, "file_opt", None) or getattr(args, "path_arg", None) or getattr(args, "file", None)
    if args.source in ("openai", "anthropic"):
        return _import_api(args, args.source)
    if args.source == "csv":
        from .importers.csv_generic import read_csv
        if not args.file:
            print("Pass the CSV path: trainlab import csv billing.csv", file=sys.stderr)
            return 2
        try:
            records = read_csv(args.file, default_provider=args.provider)
        except (ValueError, OSError) as exc:
            print(f"Import failed: {exc}", file=sys.stderr)
            return 1
        n = _ws(args).add_usage(records)
        print(f"Imported {n} usage rows from {args.file}.")
        return 0
    return 2


def _print_summary(a) -> None:
    if a.spend.has_usage:
        print(f"Monthly LLM spend: ${a.spend.monthly_total:,.0f} ({a.spend.days} days of data)")
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
    a = analyze(result, ws.load_usage())
    out = Path(args.out) if args.out else ws.path / "report"
    paths = write_reports(a, out, formats=tuple(args.format.split(",")))
    _print_summary(a)
    for p in paths:
        print(f"Wrote {p}")
    return 0


def cmd_run(args) -> int:
    rc = cmd_scan(args)
    if rc:
        return rc
    for f in args.usage_csv or []:
        args.source, args.file, args.provider = "csv", f, None
        cmd_import(args)
    return cmd_report(args)


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
    s.add_argument("path", nargs="?", default=".")
    s.add_argument("--include-tests", action="store_true")
    s.add_argument("--exclude", action="append", metavar="GLOB",
                   help="skip paths, e.g. --exclude examples --exclude 'docs/*' (repeatable)")
    s.set_defaults(fn=cmd_scan)

    i = sub.add_parser("import", parents=[common], help="import usage/billing data")
    i.add_argument("source", choices=["openai", "anthropic", "csv"])
    i.add_argument("path_arg", nargs="?", metavar="file", help="CSV path (for csv)")
    i.add_argument("--file", dest="file_opt", help="read a saved usage API response instead of calling the API")
    i.add_argument("--key", help="admin key (default: OPENAI_ADMIN_KEY / ANTHROPIC_ADMIN_KEY)")
    i.add_argument("--days", type=int, default=30)
    i.add_argument("--embeddings", action="store_true", help="the --file holds embeddings usage (OpenAI)")
    i.add_argument("--provider", help="default provider for CSV rows without one")
    i.set_defaults(fn=cmd_import)

    r = sub.add_parser("report", parents=[common], help="analyze and write the report")
    r.add_argument("--out", help="output directory (default .trainlab/report)")
    r.add_argument("--format", default="html,json")
    r.set_defaults(fn=cmd_report)

    run = sub.add_parser("run", parents=[common], help="scan + import CSVs + report in one step")
    run.add_argument("path", nargs="?", default=".")
    run.add_argument("--usage-csv", action="append")
    run.add_argument("--include-tests", action="store_true")
    run.add_argument("--exclude", action="append", metavar="GLOB")
    run.add_argument("--out")
    run.add_argument("--format", default="html,json")
    run.set_defaults(fn=cmd_run)

    p = sub.add_parser("prices", parents=[common], help="look up model prices")
    p.add_argument("action", choices=["show", "info"])
    p.add_argument("model", nargs="?")
    p.set_defaults(fn=cmd_prices)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
