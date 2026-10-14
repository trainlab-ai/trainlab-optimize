"""`trainlab ui`: a local, browser-based wizard (standard library only).

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
        self.ws = workspace
        self.token = secrets.token_urlsafe(18)
        self.start_dir = start_dir
        self.lock = threading.Lock()

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
        a = analyze(sc, self.ws.load_usage())
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
    print(f"TrainLab Optimize is running at:\n  {url}\nKeep this window open. Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TrainLab Optimize</title>
<style>
:root{--paper:#F7F9FB;--sheet:#fff;--ink:#13233A;--muted:#5B6B7F;--rule:#D7DEE6;--soft:#EEF2F6;--save:#0F7B6C;--err:#B42318;--accent:#24476B}
@media (prefers-color-scheme:dark){:root{--paper:#0E1621;--sheet:#132030;--ink:#E6EDF5;--muted:#9AA9BA;--rule:#2A3A4E;--soft:#1A2A3C;--save:#4CC2AE;--err:#F97066;--accent:#7FB2D6}}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:920px;margin:0 auto;padding:36px 20px 64px}
h1,h2{font-family:Charter,"Bitstream Charter","Sitka Text",Cambria,Georgia,serif;font-weight:600;letter-spacing:-.01em}
h1{font-size:30px;margin:0 0 4px}h2{font-size:20px;margin:0 0 8px}
.sub{color:var(--muted);margin:0 0 28px}
.step{background:var(--sheet);border:1px solid var(--rule);border-radius:10px;padding:20px 22px;margin-bottom:16px;display:grid;grid-template-columns:34px 1fr;gap:6px 14px}
.num{width:28px;height:28px;border-radius:50%;border:2px solid var(--ink);display:grid;place-items:center;font-weight:600}
.done .num{background:var(--ink);color:var(--sheet)}
p{margin:0 0 10px}.note{color:var(--muted);font-size:13px}
label{display:block;font-size:13px;color:var(--muted);margin:10px 0 4px}
input[type=text],input[type=password],select{width:100%;padding:9px 10px;border:1px solid var(--rule);border-radius:7px;background:var(--paper);color:var(--ink);font:inherit}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.row>input{flex:1;min-width:200px}
button{font:inherit;padding:8px 14px;border-radius:7px;border:1px solid var(--ink);background:var(--ink);color:var(--sheet);cursor:pointer}
button.ghost{background:transparent;color:var(--ink);border-color:var(--rule)}
button:disabled{opacity:.5;cursor:wait}button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.picker{border:1px solid var(--rule);border-radius:7px;max-height:220px;overflow:auto;margin-top:8px;display:none}
.picker div{padding:6px 10px;cursor:pointer;border-bottom:1px solid var(--soft)}.picker div:hover{background:var(--soft)}
.chips span{display:inline-block;border:1px solid var(--rule);border-radius:999px;padding:1px 10px;margin:4px 6px 0 0;font-size:13px}
.msg{margin-top:10px;font-size:14px}.err{color:var(--err)}.ok{color:var(--save)}
.tabs{display:flex;gap:6px;margin:8px 0}.tabs button{background:transparent;color:var(--ink);border-color:var(--rule)}.tabs button[aria-selected=true]{border-color:var(--ink);font-weight:600}
.big{font-size:22px;font-weight:600}.save{color:var(--save)}
#mapfields{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:0 14px}
iframe{width:100%;height:70vh;border:1px solid var(--rule);border-radius:8px;margin-top:12px;background:#fff}
a{color:var(--accent)}
</style></head><body><main class="wrap">
<h1>TrainLab Optimize</h1>
<p class="sub">See where your LLM spend goes and what to change. Everything runs on this computer; nothing is uploaded.</p>

<section class="step" id="s1"><div class="num">1</div><div>
<h2>Choose the code to analyse</h2>
<p class="note">Pick one or more project folders. Tests are skipped automatically.</p>
<div class="row"><input type="text" id="path" placeholder="/path/to/your/project"><button class="ghost" id="browse">Browse…</button><button class="ghost" id="add">Add folder</button></div>
<div class="picker" id="picker"></div>
<div class="chips" id="chosen"></div>
<label for="exclude">Skip these folders (optional, comma-separated)</label>
<input type="text" id="exclude" placeholder="examples, docs, scripts">
<div class="row" style="margin-top:12px"><button id="scan">Scan code</button></div>
<div class="msg" id="m1"></div></div></section>

<section class="step" id="s2"><div class="num">2</div><div>
<h2>Add what you spend <span class="note">(optional, but needed for savings in dollars)</span></h2>
<div class="tabs" role="tablist"><button role="tab" aria-selected="true" data-tab="file">Upload a billing or trace file</button><button role="tab" aria-selected="false" data-tab="key">Connect with an admin key</button></div>
<div id="tab-file">
<p class="note">Billing or usage exports from any provider (OpenAI, Anthropic, OpenRouter, Alibaba Cloud, DeepSeek, Mistral, AWS, Azure, Google Cloud and others), usage API JSON, gateway logs, Langfuse exports or OpenTelemetry traces. The file type is detected automatically. Prompt text in traces is never kept.</p>
<div class="row"><input type="file" id="file" accept=".csv,.json,.jsonl"><select id="kind" style="max-width:220px"><option value="auto">Detect automatically</option><option value="aws">AWS cost report</option><option value="azure">Azure cost export</option><option value="gcp">Google Cloud billing</option><option value="langfuse">Langfuse export</option><option value="otel">OpenTelemetry traces</option></select><button id="upload">Import file</button></div>
<div id="mapper" hidden><p id="mapmsg"></p><p class="note">Model and either cost or tokens are needed. Leave the rest blank if the file doesn't have them.</p><div id="mapfields"></div><div class="row" style="margin-top:8px"><button id="mapgo">Import with these columns</button></div></div></div>
<div id="tab-key" hidden>
<p class="note">Needs an organization <b>admin</b> key (OpenAI, Anthropic) or a <b>management</b> key (OpenRouter), with read access to usage. Spend is broken down per API key. The key is used once for this import and never saved.</p>
<div class="row"><select id="prov" style="max-width:160px"><option value="openai">OpenAI</option><option value="anthropic">Anthropic</option><option value="openrouter">OpenRouter</option></select><input type="password" id="key" placeholder="Admin or management key" autocomplete="off"><select id="days" style="max-width:120px"><option>7</option><option selected>30</option><option>60</option><option>90</option></select><button id="connect">Import usage</button></div></div>
<div class="chips" id="sources"></div>
<div class="row" style="margin-top:8px"><button class="ghost" id="clear">Remove imported usage</button></div>
<div class="msg" id="m2"></div></div></section>

<section class="step" id="s3"><div class="num">3</div><div>
<h2>Get the report</h2>
<div class="row"><button id="report">Create report</button><span class="note" id="dl"></span></div>
<div class="msg" id="m3"></div>
<div id="result"></div></div></section>
<p class="note">trainlab-optimize __VERSION__</p>
</main>
<script>
const T="__TOKEN__", H={"X-Trainlab-Token":T};
const $=id=>document.getElementById(id);
let chosen=[];
async function api(path,opts={}){const r=await fetch(path,{...opts,headers:{...H,...(opts.headers||{})}});const j=await r.json().catch(()=>({error:"Unexpected response"}));if(!r.ok){const e=new Error(j.error||r.statusText);e.data=j;throw e}return j}
function busy(btn,on){btn.disabled=on}
function msg(id,text,cls){const el=$(id);el.textContent=text||"";el.className="msg "+(cls||"")}
function money(v){return v==null?"—":"$"+Math.round(v).toLocaleString()}
function render(st){
  $("chosen").innerHTML=chosen.map((p,i)=>`<span>${esc(p)} <a href="#" data-i="${i}" aria-label="remove">×</a></span>`).join("");
  document.querySelectorAll("#chosen a").forEach(a=>a.onclick=e=>{e.preventDefault();chosen.splice(+a.dataset.i,1);render(st)});
  if(st.scan){$("s1").classList.add("done");msg("m1",`Found ${st.scan.call_sites} LLM call sites in ${st.scan.files} files (${st.scan.repos.join(", ")}).`+(st.scan.runtime_models?` ${st.scan.runtime_models} pick their model at runtime.`:""),"ok")}
  $("sources").innerHTML=st.usage.map(u=>`<span>${esc(u.source)} (${u.records} rows)</span>`).join("");
  if(st.usage.length)$("s2").classList.add("done");else $("s2").classList.remove("done");
  $("dl").innerHTML=st.reports.filter(n=>!n.endsWith(".html")).map(n=>`<a href="/report/${n}?t=${T}">${n.split(".").pop().toUpperCase()}</a>`).join(" ");
}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
async function refresh(){const st=await api("/api/state");if(!chosen.length&&!st.scan)$("path").value=st.start_dir;render(st);return st}
$("add").onclick=()=>{const v=$("path").value.trim();if(v&&!chosen.includes(v))chosen.push(v);$("path").value="";refresh()};
$("browse").onclick=async()=>{const pk=$("picker");if(pk.style.display==="block"){pk.style.display="none";return}await openDir($("path").value)};
async function openDir(p){try{const r=await api("/api/ls?path="+encodeURIComponent(p));$("path").value=r.path;const pk=$("picker");pk.style.display="block";
 pk.innerHTML=`<div data-p="${esc(r.parent)}">⬑ ..</div>`+r.dirs.map(d=>`<div data-p="${esc(r.path+"/"+d)}">${esc(d)}</div>`).join("");
 pk.querySelectorAll("div").forEach(el=>el.onclick=()=>openDir(el.dataset.p))}catch(e){msg("m1",e.message,"err")}}
$("scan").onclick=async e=>{const b=e.target;const paths=chosen.length?chosen:[$("path").value.trim()];busy(b,true);msg("m1","Scanning…");
 try{render(await api("/api/scan",{method:"POST",body:JSON.stringify({paths,exclude:$("exclude").value.split(",")})}));$("picker").style.display="none"}catch(err){msg("m1",err.message,"err")}finally{busy(b,false)}};
document.querySelectorAll(".tabs button").forEach(t=>t.onclick=()=>{document.querySelectorAll(".tabs button").forEach(x=>x.setAttribute("aria-selected",x===t));$("tab-file").hidden=t.dataset.tab!=="file";$("tab-key").hidden=t.dataset.tab!=="key"});
const MAPF=[["model","Model (required)"],["cost_usd","Cost"],["input_tokens","Input tokens"],["output_tokens","Output tokens"],["date","Date"],["api_key","API key"],["project","Project / team"],["provider","Provider"]];
async function upload(btn,map){const f=$("file").files[0];if(!f){msg("m2","Choose a file first.","err");return}busy(btn,true);msg("m2","Importing…");
 const hd={"X-Filename":encodeURIComponent(f.name),"X-Kind":$("kind").value};if(map)hd["X-Map"]=encodeURIComponent(JSON.stringify(map));
 try{render(await api("/api/import/file",{method:"POST",headers:hd,body:await f.arrayBuffer()}));$("mapper").hidden=true;msg("m2","Imported "+f.name,"ok")}
 catch(err){if(err.data&&err.data.columns){msg("m2","");$("mapmsg").textContent=err.message;showMapper(err.data.columns)}else msg("m2",err.message,"err")}finally{busy(btn,false)}}
function showMapper(cols){const opts=`<option value="">—</option>`+cols.map(c=>`<option>${esc(c)}</option>`).join("");
 $("mapfields").innerHTML=MAPF.map(([k,l])=>`<div><label for="map-${k}">${l}</label><select id="map-${k}">${opts}</select></div>`).join("");$("mapper").hidden=false}
$("upload").onclick=e=>upload(e.target,null);
$("mapgo").onclick=e=>{const m={};MAPF.forEach(([k])=>{const v=$("map-"+k).value;if(v)m[k]=v});if(!m.model){msg("m2","Pick the model column.","err");return}upload(e.target,m)};
$("connect").onclick=async e=>{busy(e.target,true);msg("m2","Fetching usage…");
 try{render(await api("/api/import/api",{method:"POST",body:JSON.stringify({provider:$("prov").value,key:$("key").value,days:$("days").value})}));$("key").value="";msg("m2","Usage imported.","ok")}catch(err){msg("m2",err.message,"err")}finally{busy(e.target,false)}};
$("clear").onclick=async()=>{render(await api("/api/usage/clear",{method:"POST"}));msg("m2","Usage removed.","ok")};
$("report").onclick=async e=>{busy(e.target,true);msg("m3","Analysing…");
 try{const st=await api("/api/report",{method:"POST",body:JSON.stringify({formats:["html","json","md","pdf"]})});render(st);const s=st.summary;
  msg("m3",(st.warnings||[]).join(" "),"note");
  $("result").innerHTML=(s.monthly_total!=null?`<p><span class="big">${money(s.monthly_total)}</span> per month, <span class="big save">${money(s.savings_low)}–${money(s.savings_high)}</span> potential saving, ${s.findings} findings</p>`:`<p>${s.findings} findings from code. Add usage in step 2 to put a price on them.</p>`)+
  `<iframe title="Report" src="/report/trainlab-report.html?t=${T}&v=${Date.now()}"></iframe><p class="note"><a target="_blank" href="/report/trainlab-report.html?t=${T}">Open report in a new tab</a></p>`;
  $("s3").classList.add("done")}catch(err){msg("m3",err.message,"err")}finally{busy(e.target,false)}};
refresh();
</script></body></html>"""
