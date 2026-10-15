"""`trainlab ui`: a local, browser-based app (standard library only): Overview, Spend, Live,
Experiments and Setup. The page lives in ui_page.py, the data behind it in ui_data.py.

Security: binds to 127.0.0.1, requires a random session token on every API call, rejects
non-localhost Host headers (DNS rebinding), never stores API keys, and only serves report
files from the workspace.
"""
from __future__ import annotations

import json
import os
import secrets
import tempfile
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import __version__
from .analyzer.engine import analyze
from .importers import ImportError_
from .report import warnings as report_warnings
from .report import write_reports
from .scanner import scan
from .workspace import Workspace

REPORT_FILES = {"trainlab-report.html": "text/html; charset=utf-8",
                "trainlab-report.json": "application/json",
                "trainlab-report.md": "text/markdown; charset=utf-8",
                "trainlab-report.pdf": "application/pdf"}
MAX_UPLOAD = 200 * 1024 * 1024


class App:
    def __init__(self, workspace: Workspace, start_dir: Path):
        from .ui_data import Jobs, LiveFeed
        self.ws = workspace
        self.token = secrets.token_urlsafe(18)
        self.start_dir = start_dir
        self.lock = threading.Lock()
        self.live = LiveFeed(workspace.path / "traces")
        self.jobs = Jobs(workspace)

    # ---- actions
    def state(self) -> dict:
        sc = self.ws.load_scan()
        usage = self.ws.load_usage()
        sources: dict[str, int] = {}
        for r in usage:
            sources[r.source] = sources.get(r.source, 0) + 1
        report_dir = self.ws.path / "report"
        return {
            "version": __version__,
            "start_dir": str(self.start_dir),
            "scan": None if sc is None else {
                "repos": sc.repos or [Path(sc.root).name], "files": sc.files_scanned,
                "call_sites": len(sc.call_sites),
                "runtime_models": sum(1 for s in sc.call_sites if s.kind in ("chat", "embedding") and not s.model),
                "languages": sc.files_by_language},
            "usage": [{"source": k, "records": v} for k, v in sorted(sources.items())],
            "reports": [n for n in REPORT_FILES if (report_dir / n).exists()],
        }

    def do_scan(self, body: dict) -> dict:
        paths = [p for p in body.get("paths", []) if str(p).strip()]
        if not paths:
            raise ValueError("Choose at least one folder to scan.")
        exclude = [e.strip() for e in body.get("exclude", []) if str(e).strip()]
        result = None
        for p in paths:
            path = Path(os.path.expanduser(p))
            if not path.is_dir():
                raise ValueError(f"Folder not found: {p}")
            r = scan(path, exclude=exclude, include_tests=bool(body.get("include_tests")))
            result = r if result is None else result.merge(r)
        self.ws.save_scan(result)
        return self.state()

    def do_import_file(self, filename: str, kind: str, data: bytes,
                       column_map: dict[str, str] | None = None) -> dict:
        from .importers.auto import read_any
        suffix = Path(filename).suffix.lower() or ".csv"
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / (Path(filename).name or f"upload{suffix}")
            tmp.write_bytes(data)
            try:
                records, _ = read_any(tmp, "auto" if kind == "csv" else kind, column_map=column_map or None)
            except ValueError as exc:
                if suffix not in (".json", ".jsonl", ".ndjson") and "Columns" in str(exc):
                    import csv
                    with open(tmp, newline="", encoding="utf-8-sig", errors="replace") as fh:
                        header = next(csv.reader(fh), [])
                    raise ColumnsNeeded("This file's column names aren't ones we recognise. Tell us which column is which.",
                                        header) from exc
                raise
        if not records:
            raise ValueError("No LLM usage found in that file. Check the service and period it covers.")
        self.ws.add_usage(records)
        return self.state()

    def do_import_api(self, body: dict) -> dict:
        from .importers.anthropic_usage import AnthropicUsageImporter
        from .importers.openai_usage import OpenAIUsageImporter
        from .importers.openrouter import OpenRouterImporter
        provider, key = body.get("provider"), (body.get("key") or "").strip()
        days = int(body.get("days") or 30)
        if not key:
            raise ValueError("Paste an admin key (read-only usage access).")
        importers = {"openai": OpenAIUsageImporter, "anthropic": AnthropicUsageImporter,
                     "openrouter": OpenRouterImporter}
        if provider not in importers:
            raise ValueError(f"Unknown provider: {provider}")
        imp = importers[provider](key)
        records = imp.fetch(days=days)          # the key is used once and never stored
        self.ws.add_usage(records)
        return self.state()

    def do_clear_usage(self) -> dict:
        self.ws.clear_usage()
        return self.state()

    def do_report(self, body: dict) -> dict:
        sc = self.ws.load_scan()
        if sc is None:
            raise ValueError("Scan your code first.")
        from .experiment.runner import load_results
        a = analyze(sc, self.ws.load_usage(), experiments=load_results(self.ws.path))
        formats = tuple(f for f in body.get("formats", ["html", "json", "md"]) if f in ("html", "json", "md", "pdf"))
        write_reports(a, self.ws.path / "report", formats=formats or ("html",))
        st = self.state()
        st["summary"] = {"monthly_total": a.spend.monthly_total if a.spend.has_usage else None,
                         "savings_low": a.savings_low, "savings_high": a.savings_high,
                         "findings": len([f for f in a.findings if f.strategy != "coverage"])}
        st["warnings"] = list(report_warnings)
        return st

    def list_dirs(self, path: str) -> dict:
        p = Path(os.path.expanduser(path or str(self.start_dir))).resolve()
        if not p.is_dir():
            raise ValueError(f"Not a folder: {path}")
        dirs = []
        try:
            for c in sorted(p.iterdir(), key=lambda x: x.name.lower()):
                if c.is_dir() and not c.name.startswith("."):
                    dirs.append(c.name)
        except PermissionError:
            pass
        return {"path": str(p), "parent": str(p.parent), "dirs": dirs[:500],
                "looks_like_repo": (p / ".git").exists()}


class ColumnsNeeded(ValueError):
    def __init__(self, message: str, columns: list[str]):
        super().__init__(message)
        self.columns = columns


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "trainlab-ui"

        def log_message(self, *args):   # quiet
            pass

        # -- helpers
        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").split(":")[0]
            return host in ("127.0.0.1", "localhost")

        def _token_ok(self) -> bool:
            q = parse_qs(urlparse(self.path).query)
            return secrets.compare_digest(self.headers.get("X-Trainlab-Token") or (q.get("t") or [""])[0],
                                          app.token)

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_UPLOAD:
                raise ValueError("File too large (200 MB limit).")
            return self.rfile.read(n) if n else b""

        # -- routes
        def do_GET(self):
            if not self._host_ok():
                return self._send(HTTPStatus.FORBIDDEN, b"forbidden", "text/plain")
            url = urlparse(self.path)
            if url.path == "/":
                if not self._token_ok():
                    return self._send(HTTPStatus.FORBIDDEN, b"Open the link printed by `trainlab ui`.",
                                      "text/plain")
                page = PAGE.replace("__TOKEN__", app.token).replace("__VERSION__", __version__)
                return self._send(200, page.encode(), "text/html; charset=utf-8",
                                  {"Content-Security-Policy": "default-src 'self'; style-src 'unsafe-inline'; "
                                   "script-src 'unsafe-inline'; frame-src 'self'"})
            if not self._token_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "missing or wrong session token"})
            try:
                if url.path == "/api/state":
                    return self._json(200, app.state())
                if url.path == "/api/ls":
                    return self._json(200, app.list_dirs(parse_qs(url.query).get("path", [""])[0]))
                from . import ui_data
                q = parse_qs(url.query)
                if url.path == "/api/overview":
                    return self._json(200, ui_data.overview(app.ws, app.jobs.list()))
                if url.path == "/api/spend":
                    return self._json(200, ui_data.spend(app.ws))
                if url.path == "/api/source":
                    return self._json(200, ui_data.source(app.ws, (q.get("repo") or [None])[0], q.get("path", [""])[0]))
                if url.path == "/api/live":
                    return self._json(200, app.live.snapshot(int((q.get("minutes") or ["60"])[0])))
                if url.path == "/api/experiments":
                    return self._json(200, ui_data.experiments(app.ws, app.jobs.list()))
                if url.path.startswith("/api/experiments/"):
                    return self._json(200, ui_data.experiment_detail(app.ws, unquote(url.path.rsplit("/", 1)[-1])))
                if url.path.startswith("/report/"):
                    name = url.path.rsplit("/", 1)[-1]
                    f = app.ws.path / "report" / name
                    if name not in REPORT_FILES or not f.exists():
                        return self._json(404, {"error": "not found"})
                    extra = {} if name.endswith(".html") else {"Content-Disposition": f'attachment; filename="{name}"'}
                    return self._send(200, f.read_bytes(), REPORT_FILES[name], extra)
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._host_ok() or not self._token_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
            url = urlparse(self.path)
            try:
                raw = self._body()
                with app.lock:
                    if url.path == "/api/import/file":
                        cmap = json.loads(unquote(self.headers.get("X-Map") or "") or "{}")
                        return self._json(200, app.do_import_file(
                            unquote(self.headers.get("X-Filename") or "") or "upload.csv",
                            self.headers.get("X-Kind") or "auto", raw,
                            {k: v for k, v in cmap.items() if v} if isinstance(cmap, dict) else None))
                    body = json.loads(raw or b"{}")
                    if url.path == "/api/scan":
                        return self._json(200, app.do_scan(body))
                    if url.path == "/api/import/api":
                        return self._json(200, app.do_import_api(body))
                    if url.path == "/api/usage/clear":
                        return self._json(200, app.do_clear_usage())
                    if url.path == "/api/report":
                        return self._json(200, app.do_report(body))
                if url.path == "/api/experiments/plan":           # outside the lock: may take a moment
                    return self._json(200, app.jobs.plan(body))
                if url.path == "/api/experiments/run":
                    return self._json(200, app.jobs.start(body))
            except ColumnsNeeded as exc:
                return self._json(400, {"error": str(exc), "columns": exc.columns})
            except (ValueError, ImportError_, KeyError, OSError) as exc:
                return self._json(400, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover - surface unexpected errors to the page
                return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return self._json(404, {"error": "not found"})

    return Handler


def serve(workspace: Workspace, port: int = 0, open_browser: bool = True, start_dir: Path | None = None):
    workspace.ensure()
    app = App(workspace, (start_dir or Path.cwd()).resolve())
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    url = f"http://127.0.0.1:{httpd.server_address[1]}/?t={app.token}"
    return httpd, app, url


def run(workspace: Workspace, port: int = 0, open_browser: bool = True) -> None:
    httpd, _, url = serve(workspace, port)
    print(f"TrainLab Optimize is running at:\n  {url}\nKeep this window open. Press Ctrl+C to stop.", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


from .ui_page import PAGE  # noqa: E402
