import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from trainlab_optimize.ui import serve
from trainlab_optimize.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def server(tmp_path):
    httpd, app, url = serve(Workspace(tmp_path / "ws"), port=0, start_dir=ROOT)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = url.split("/?")[0]
    yield base, app.token
    httpd.shutdown()
    httpd.server_close()


def call(base, path, token=None, body=None, headers=None, host=None):
    h = {"X-Trainlab-Token": token} if token else {}
    h.update(headers or {})
    if host:
        h["Host"] = host
    data = body if isinstance(body, bytes) else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(base + path, data=data, headers=h, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_requires_token_and_localhost(server):
    base, token = server
    assert call(base, "/api/state")[0] == 403
    assert call(base, "/")[0] == 403
    assert call(base, "/api/state", token, host="evil.example.com")[0] == 403
    assert call(base, "/api/state", token)[0] == 200


def test_full_flow(server):
    base, token = server
    code, body = call(base, "/api/scan", token, {"paths": [str(ROOT / "examples/demo_app")]})
    assert code == 200 and json.loads(body)["scan"]["call_sites"] == 7
    csv = (ROOT / "examples/demo_usage/billing_export.csv").read_bytes()
    code, body = call(base, "/api/import/file", token, csv, {"X-Filename": "billing.csv", "X-Kind": "auto"})
    assert code == 200 and json.loads(body)["usage"]
    code, body = call(base, "/api/report", token, {"formats": ["html", "json", "md"]})
    st = json.loads(body)
    assert code == 200 and round(st["summary"]["monthly_total"]) == 23501
    code, html = call(base, "/report/trainlab-report.html?t=" + token)
    assert code == 200 and b"LLM cost report" in html
    assert call(base, "/report/../../etc/passwd?t=" + token)[0] == 404


def test_errors_are_readable(server):
    base, token = server
    code, body = call(base, "/api/scan", token, {"paths": ["/definitely/not/here"]})
    assert code == 400 and b"Folder not found" in body
    code, body = call(base, "/api/report", token, {})
    assert code == 400 and b"Scan your code first" in body
