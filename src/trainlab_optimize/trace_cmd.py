"""`trainlab trace -- <command>`: Profile 2 in one command, with no code change.

Runs any Python command (python app.py, uvicorn, pytest, a CLI entry point) with
instrument() switched on through a generated sitecustomize.py on PYTHONPATH, then scans the
code, imports the traces and writes the report. Child Python processes the app starts are
recorded too. Ctrl+C stops the app and still produces the report, so a server can be
exercised by hand and then stopped.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

BOOTSTRAP = '''\
# Written by `trainlab trace`. Starts trainlab_optimize.instrument() in this Python process,
# then runs any sitecustomize the environment already had. Never stops the app on failure.
def _trainlab_boot():
    import json, os, sys
    if getattr(sys, "_trainlab_trace_boot", False):     # never run twice in one process
        return
    sys._trainlab_trace_boot = True

    def norm(p):
        return os.path.normcase(os.path.realpath(os.path.abspath(p or os.curdir)))
    here = norm(os.path.dirname(__file__))
    me = sys.modules.pop("sitecustomize", None)
    sys.path[:] = [p for p in sys.path if norm(p) != here]
    try:
        import sitecustomize  # noqa: F401  the original one, if there is one
    except ImportError as exc:
        if getattr(exc, "name", None) != "sitecustomize":
            sys.stderr.write("trainlab trace: your sitecustomize failed: %s\\n" % exc)
        if me is not None:
            sys.modules["sitecustomize"] = me
    except Exception as exc:
        sys.stderr.write("trainlab trace: your sitecustomize failed: %s\\n" % exc)
    cfg = json.loads(os.environ.get("TRAINLAB_TRACE_CONFIG") or "{}")
    try:
        try:
            import trainlab_optimize
        except ImportError:
            if not cfg.get("pkg_parent"):
                raise
            sys.path.append(cfg["pkg_parent"])          # lowest priority: the app's own packages win
            import trainlab_optimize
        trainlab_optimize.instrument(path=cfg.get("traces"), project_root=cfg.get("root"),
                                     service=cfg.get("service"), tags=cfg.get("tags") or None,
                                     capture=cfg.get("capture"))
    except Exception as exc:
        sys.stderr.write("trainlab trace: recording is off in this process (%s: %s)\\n"
                         % (type(exc).__name__, exc))
    finally:
        sys.path[:] = [p for p in sys.path if norm(p) != here]


_trainlab_boot()
del _trainlab_boot
'''


def _count_calls(traces: Path) -> int:
    n = 0
    if traces.is_dir():
        for f in traces.glob("*.jsonl"):
            try:
                with open(f, "rb") as fh:
                    n += sum(1 for _ in fh)
            except OSError:
                pass
    return n


def _parse_tags(items) -> dict:
    tags = {}
    for item in items or []:
        k, sep, v = item.partition("=")
        if not sep or not k.strip():
            raise ValueError(f"--tag needs key=value, got {item!r}")
        tags[k.strip()] = v.strip()
    return tags


def _resolve(cmd: list[str]) -> list[str]:
    exe = shutil.which(cmd[0])
    return [exe or cmd[0], *cmd[1:]]


def run_traced(cmd: list[str], *, workspace: Path, root: Path, service: str | None, tags: dict,
               capture: dict | None) -> tuple[int, int, bool]:
    """Run cmd with recording on. Returns (exit code, calls recorded in this run, stopped by Ctrl+C)."""
    traces = workspace / "traces"
    boot = workspace / "run" / "bootstrap"
    boot.mkdir(parents=True, exist_ok=True)
    (boot / "sitecustomize.py").write_text(BOOTSTRAP, encoding="utf-8")
    import trainlab_optimize
    cfg = {"traces": str(traces), "root": str(root), "service": service, "tags": tags, "capture": capture,
           "pkg_parent": str(Path(trainlab_optimize.__file__).resolve().parent.parent)}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(boot) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["TRAINLAB_TRACE_CONFIG"] = json.dumps(cfg)
    env.pop("TRAINLAB_INSTRUMENT", None)
    before = _count_calls(traces)
    try:
        proc = subprocess.Popen(_resolve(cmd), env=env)
    except OSError as exc:
        print(f"Could not start {cmd[0]!r}: {exc}", file=sys.stderr)
        return 127, 0, False
    interrupted = False
    try:
        rc = proc.wait()
    except KeyboardInterrupt:
        interrupted = True
        # The app got the same Ctrl+C; give it time to shut down and flush its traces.
        print("\nStopping the app ...", file=sys.stderr)
        try:
            rc = proc.wait(timeout=20)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            proc.send_signal(signal.SIGTERM) if os.name != "nt" else proc.terminate()
            try:
                rc = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = proc.wait()
    return rc, _count_calls(traces) - before, interrupted


def cmd_trace(args) -> int:
    from .cli import _import_file_auto, _ws, cmd_report
    from .scanner import scan

    cmd = list(args.command or [])
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("Give the command to run after --, e.g.  trainlab trace -- python app.py", file=sys.stderr)
        return 2
    try:
        tags = _parse_tags(args.tag)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    ws = _ws(args).ensure()
    roots = [Path(p).resolve() for p in (args.path or ["."])]
    for r in roots:
        if not r.is_dir():
            print(f"Not a directory: {r}", file=sys.stderr)
            return 2
    if args.fresh:
        shutil.rmtree(ws.path / "traces", ignore_errors=True)
    capture = None
    if args.capture is not None:
        capture = {"sample_rate": args.capture_rate, "max_per_site": args.capture_max,
                   "path": str(ws.path / "samples")}
        if args.capture:
            capture["sites"] = args.capture
    service = args.service or roots[0].name

    print(f"trainlab trace: recording LLM calls from `{' '.join(cmd)}`" + (" (Ctrl+C to stop)" if sys.stdin.isatty() else ""))
    t0 = time.time()
    rc, n, interrupted = run_traced(cmd, workspace=ws.path.resolve(), root=roots[0], service=service, tags=tags,
                                    capture=capture)
    total = _count_calls(ws.path / "traces")
    how = "was stopped with Ctrl+C" if interrupted else f"exited with code {rc}"
    if interrupted:
        rc = 0                                  # stopping the app was the plan, not a failure
    print(f"\ntrainlab trace: the app {how} after {time.time() - t0:.0f}s; "
          f"recorded {n} LLM call(s)" + (f" ({total} in total with earlier runs; --fresh to start over)"
                                       if total > n else "") + ".")
    if capture is not None:
        samples = ws.path / "samples"
        files = list(samples.glob("*.jsonl")) if samples.is_dir() else []
        print(f"  captured samples from {len(files)} call site(s) in {samples} (redacted) for `trainlab experiment run`")
    if total == 0:
        print("  Nothing to report. trace records Python apps that call an LLM SDK or HTTP API "
              "(OpenAI, Anthropic, LiteLLM, Bedrock, Gemini, raw HTTP). For other languages use "
              "`trainlab run --usage <billing export>`.", file=sys.stderr)
        return rc or 1
    if args.no_report:
        return rc

    result = None
    for r in roots:
        sr = scan(r, include_tests=args.include_tests, exclude=args.exclude or [])
        result = sr if result is None else result.merge(sr)
    ws.save_scan(result)
    args.source, args.provider, args.map = "file", None, None
    args.file = str(ws.path / "traces")
    _import_file_auto(args)
    for f in args.usage or []:
        args.file = f
        _import_file_auto(args)
    report_rc = cmd_report(args)
    if report_rc == 0 and args.open:
        open_report(args)
    return rc or report_rc


def open_report(args) -> None:
    import webbrowser
    out = Path(args.out) if getattr(args, "out", None) else Path(args.workspace) / "report"
    html = (out / "trainlab-report.html").resolve()
    if html.exists():
        webbrowser.open(html.as_uri())


def add_parser(sub, common) -> None:
    import argparse
    t = sub.add_parser(
        "trace", parents=[common],
        help="run a Python app with recording on, then report cost per line (no code change)",
        description="Run a Python command with every LLM call recorded, then scan, import and report. "
                    "Example: trainlab trace -- python app.py",
        epilog="Everything after -- is the command. Ctrl+C stops the app and still writes the report.")
    t.add_argument("--path", action="append", metavar="DIR",
                   help="code to scan and attribute calls to (default: current folder; repeatable)")
    t.add_argument("--service", help="service name stored on each call (default: the folder name)")
    t.add_argument("--tag", action="append", metavar="KEY=VALUE", help="tag every call, e.g. --tag env=staging")
    t.add_argument("--capture", nargs="*", metavar="SITE",
                   help="also save real inputs and outputs for `trainlab experiment run` (redacted, local). "
                        "No value = every call site; or list sites as path:line")
    t.add_argument("--capture-rate", type=float, default=1.0, help="fraction of calls to capture (default 1.0)")
    t.add_argument("--capture-max", type=int, default=200, help="most samples kept per call site (default 200)")
    t.add_argument("--fresh", action="store_true", help="drop traces from earlier runs first")
    t.add_argument("--usage", action="append", metavar="FILE", help="also import a billing/usage file (repeatable)")
    t.add_argument("--include-tests", action="store_true")
    t.add_argument("--exclude", action="append", metavar="GLOB")
    t.add_argument("--no-report", action="store_true", help="only record; don't scan or report")
    t.add_argument("--open", action="store_true", help="open the HTML report when done")
    t.add_argument("--out")
    t.add_argument("--format", default="html,json")
    t.add_argument("--fail-on", choices=["none", "high", "medium", "low"], default="none")
    t.add_argument("command", nargs=argparse.REMAINDER, help="-- then the command, e.g. -- python app.py")
    t.set_defaults(fn=cmd_trace)
