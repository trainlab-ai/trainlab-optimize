"""The `trainlab ui` page: Overview (briefing), Spend (code map), Live (code map), Experiments
(ledger with a sample review) and Setup (the scan / import / report wizard). One self-contained
document; no external fonts or scripts, so it works offline and under a strict CSP."""

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TrainLab Optimize</title>
<style>
:root{--bg:#F4F6F5;--sheet:#FFFFFF;--well:#F0F3F1;--ink:#17201C;--body:#3A4540;--muted:#5D6862;--line:#D9DFDB;--soft:#E9EDEA;
--save:#0E7A5C;--save-soft:#DDF1E8;--warn:#9A5300;--warn-soft:#FBEBD9;--bad:#A63A3A;--bad-soft:#F7E3E3;--act:#2448C4;--hot:#B35A06;
--code-hl:rgba(179,90,6,.09);--focus:#2448C4;
--serif:"Iowan Old Style","Charter","Sitka Text","Palatino Linotype",Cambria,Georgia,serif;
--sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
--mono:ui-monospace,"SFMono-Regular","Cascadia Mono",Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#121715;--sheet:#1A201D;--well:#151A18;--ink:#E5EAE7;--body:#C3CCC7;--muted:#9AA5A0;--line:#2E3733;--soft:#232A27;
--save:#5BC8A8;--save-soft:#173A31;--warn:#F0A35A;--warn-soft:#3A2A17;--bad:#F08A8A;--bad-soft:#3A2228;--act:#8FB0FF;--hot:#F0A35A;--code-hl:rgba(240,163,90,.10);--focus:#8FB0FF}}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--sans)}
button,input,select{font:inherit;color:inherit}
a{color:var(--act)}
:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.num,.mono{font-variant-numeric:tabular-nums}.mono{font-family:var(--mono)}
.muted{color:var(--muted)}.save{color:var(--save)}.warn{color:var(--warn)}.bad{color:var(--bad)}
header.top{position:sticky;top:0;z-index:5;background:var(--sheet);border-bottom:1px solid var(--line)}
.top-in{max-width:1280px;margin:0 auto;display:flex;flex-wrap:wrap;gap:6px 22px;align-items:center;padding:0 max(16px,env(safe-area-inset-left))}
.brand{font-weight:600;padding:14px 0}
nav.tabs{display:flex;flex-wrap:wrap}
nav.tabs a{padding:15px 12px 13px;color:var(--muted);text-decoration:none;border-bottom:2px solid transparent;font-size:14px}
nav.tabs a[aria-current=page]{color:var(--ink);font-weight:600;border-bottom-color:var(--ink)}
.ws{margin-left:auto;font-size:13px;color:var(--muted)}
main{max-width:1280px;margin:0 auto;padding:0 max(16px,env(safe-area-inset-left)) 64px}
section.view[hidden]{display:none}
.btn{display:inline-flex;align-items:center;gap:6px;min-height:38px;padding:7px 14px;border-radius:6px;border:1px solid var(--ink);background:var(--ink);color:var(--sheet);cursor:pointer;text-decoration:none;font-size:14px}
.btn.ghost{background:var(--sheet);color:var(--ink);border-color:var(--line)}
.btn.act{background:var(--act);border-color:var(--act);color:#fff}
.btn:disabled{opacity:.5;cursor:not-allowed}
.empty{background:var(--sheet);border:1px dashed var(--line);border-radius:8px;padding:22px;margin-top:24px}
.empty code,.code-inline{font-family:var(--mono);background:var(--soft);padding:2px 6px;border-radius:4px;font-size:13px}
.err{color:var(--bad)}.ok{color:var(--save)}

/* ---- Overview: the briefing */
.brief{max-width:780px;margin:0 auto;padding:44px 0 24px}
.brief h1{font-family:var(--serif);font-size:38px;line-height:1.16;font-weight:600;margin:0;text-wrap:balance;letter-spacing:-.005em}
.lede{font-size:18px;line-height:1.6;color:var(--body);margin:16px 0 0;max-width:66ch}
.brief h2{font-family:var(--serif);font-size:23px;font-weight:600;margin:0 0 14px}
.brief section{margin-top:40px}.brief section.rule{padding-top:28px;border-top:1px solid var(--line)}
ol.steps{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:12px}
.step{display:grid;grid-template-columns:30px minmax(0,1fr) auto;gap:14px;padding:18px;border:1px solid var(--line);border-radius:8px;background:var(--sheet);align-items:start}
.step .i{font-family:var(--serif);font-size:22px;color:var(--muted)}
.step .t{font-size:17px;font-weight:600}.step .d{font-size:15px;color:var(--body);margin-top:4px}
.step .acts{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap}
.amt{text-align:right;font-family:var(--mono);font-size:19px;color:var(--save);white-space:nowrap}
.amt.m{color:var(--muted)}.amt small{display:block;font-family:var(--sans);font-size:13px;color:var(--muted)}
.three{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:18px}
.three .n{font-family:var(--mono);font-size:22px}.three .x{font-size:14px;color:var(--body);margin-top:4px}
.prog{height:6px;background:var(--soft);border-radius:3px;margin-top:8px}.prog i{display:block;height:6px;background:var(--warn);border-radius:3px}

/* ---- Spend + Live: the code map */
.map{display:grid;grid-template-columns:minmax(220px,280px) minmax(0,1fr);gap:22px;padding-top:22px}
@media (max-width:820px){.map{grid-template-columns:minmax(0,1fr)}}
aside.tree{background:var(--sheet);border:1px solid var(--line);border-radius:8px;padding:12px;align-self:start;position:sticky;top:64px;max-height:calc(100vh - 84px);overflow:auto}
.tree h3{margin:2px 4px 10px;font-size:12px;font-weight:500;color:var(--muted);letter-spacing:.04em}
.tree button{display:flex;justify-content:space-between;gap:8px;width:100%;text-align:left;border:0;background:none;padding:6px 8px;border-radius:5px;cursor:pointer;font-family:var(--mono);font-size:12.5px}
.tree button:hover{background:var(--soft)}.tree button[aria-current=true]{background:var(--soft);font-weight:600}
.tree .ln{padding-left:20px;color:var(--muted)}.tree .hot{color:var(--hot)}
.panel{background:var(--sheet);border:1px solid var(--line);border-radius:8px;padding:16px;min-width:0}
.panel h2{margin:0;font-size:15px;font-weight:600}
.row-sb{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;align-items:baseline}
.code{font-family:var(--mono);font-size:13px;line-height:1.7;background:var(--well);border-radius:6px;padding:8px 0;overflow-x:auto;margin-top:12px}
.code .l{display:grid;grid-template-columns:96px 46px minmax(max-content,1fr);white-space:pre}
.code .g{text-align:right;padding-right:10px;color:var(--hot)}.code .n{text-align:right;padding-right:12px;color:var(--muted);opacity:.7}
.code .hl{background:var(--code-hl)}.code .gap{color:var(--muted);padding-left:150px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin-top:14px}
.card{border:1px solid var(--line);border-radius:6px;padding:12px;background:var(--sheet)}
.card .loc{font-family:var(--mono);font-size:12px;color:var(--muted)}
.card .big{font-family:var(--mono);font-size:20px;margin:6px 0 2px}
.chip{display:inline-block;font-size:12px;padding:1px 8px;border-radius:10px;margin:4px 4px 0 0;border:1px solid var(--line);color:var(--muted)}
.chip.ok{background:var(--save-soft);border-color:transparent;color:var(--save)}.chip.mid{background:var(--warn-soft);border-color:transparent;color:var(--warn)}
.chip.no{background:var(--bad-soft);border-color:transparent;color:var(--bad)}
.stats{display:flex;flex-wrap:wrap;gap:10px 30px;align-items:baseline;padding:22px 0 4px}
.stat .k{font-size:13px;color:var(--muted)}.stat .v{font-family:var(--mono);font-size:24px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:var(--muted);margin-right:6px}.dot.on{background:var(--save)}
table{border-collapse:collapse;width:100%}th{text-align:left;font-size:12px;font-weight:500;color:var(--muted);padding:8px 10px;white-space:nowrap}
td{padding:9px 10px;border-top:1px solid var(--soft);vertical-align:middle}.r{text-align:right}
.tblbox{overflow-x:auto}.tblbox table{min-width:720px}
.tail{font-family:var(--mono);font-size:12.5px;line-height:1.75;background:var(--well);border-radius:6px;padding:10px 12px;overflow-x:auto;white-space:pre;color:var(--muted);margin-top:10px;max-height:340px;overflow-y:auto}
.tail b{color:var(--ink);font-weight:400}.tail .e{color:var(--bad)}
svg.spark{display:block}

/* ---- Experiments: ledger + sample review */
.ledger{padding-top:22px;display:flex;flex-direction:column;gap:18px}
.form{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-top:12px}
.form label{display:flex;flex-direction:column;gap:6px;font-size:13px;color:var(--muted)}
.form select,.form input[type=text],.form input[type=number]{min-height:38px;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:var(--sheet)}
.checks{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:14px;color:var(--ink)}
.planbox{margin-top:12px;background:var(--well);border-radius:6px;padding:12px 14px;font-size:14px}
.planbox dl{display:grid;grid-template-columns:minmax(0,180px) minmax(0,1fr);gap:4px 16px;margin:0}.planbox dt{color:var(--muted)}.planbox dd{margin:0}
.verdict{display:inline-block;font-size:12px;font-weight:600;padding:2px 8px;border-radius:4px;line-height:1.5}
.v-PASS{background:var(--save);color:var(--sheet)}td.nw{white-space:nowrap}.v-FAIL{background:var(--bad-soft);color:var(--bad)}.v-INCONCLUSIVE{background:var(--warn-soft);color:var(--warn)}
.v-ERROR,.v-NEEDS-JUDGE{border:1px solid var(--line);color:var(--muted)}
.split{display:grid;grid-template-columns:minmax(220px,320px) minmax(0,1fr);gap:14px;margin-top:12px}
@media (max-width:820px){.split{grid-template-columns:minmax(0,1fr)}}
.slist{display:flex;flex-direction:column;gap:2px;max-height:460px;overflow-y:auto}
.slist button{display:flex;flex-direction:column;align-items:flex-start;gap:2px;text-align:left;padding:8px 10px;border:0;border-radius:5px;background:transparent;cursor:pointer}
.slist button[aria-current=true]{background:var(--soft)}.slist .f{font-family:var(--mono);font-size:12px;color:var(--muted)}
.slist .s{font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:290px}
pre.box{margin:0;font-family:var(--mono);font-size:12.5px;line-height:1.6;background:var(--well);border-radius:6px;padding:10px 12px;white-space:pre-wrap;word-break:break-word}
.lbl{font-size:12px;color:var(--muted);margin:10px 0 6px}
.cmp{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.fields{font-size:13px;color:var(--body);margin-top:6px;line-height:1.7}

/* ---- Setup: the original wizard */
.wiz{max-width:880px;margin:0 auto;padding-top:28px}
.wstep{background:var(--sheet);border:1px solid var(--line);border-radius:10px;padding:20px 22px;margin-bottom:16px;display:grid;grid-template-columns:34px minmax(0,1fr);gap:6px 14px}
.wstep>div{min-width:0}.wiz input[type=file]{max-width:100%}
.wnum{width:28px;height:28px;border-radius:50%;border:2px solid var(--ink);display:grid;place-items:center;font-weight:600}
.done .wnum{background:var(--ink);color:var(--sheet)}
.wiz h2{font-size:18px;margin:0 0 8px}.wiz p{margin:0 0 10px}.note{color:var(--muted);font-size:13px}
.wiz label{display:block;font-size:13px;color:var(--muted);margin:10px 0 4px}
.wiz input[type=text],.wiz input[type=password],.wiz select{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:6px;background:var(--bg)}
.wrow{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.wrow>input{flex:1;min-width:200px}
.picker{border:1px solid var(--line);border-radius:6px;max-height:220px;overflow:auto;margin-top:8px;display:none}
.picker div{padding:6px 10px;cursor:pointer;border-bottom:1px solid var(--soft)}.picker div:hover{background:var(--soft)}
.chips span{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:1px 10px;margin:4px 6px 0 0;font-size:13px}
.msg{margin-top:10px;font-size:14px}
.wtabs{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}.wtabs button{background:transparent;border:1px solid var(--line);border-radius:6px;padding:6px 12px;cursor:pointer}.wtabs button[aria-selected=true]{border-color:var(--ink);font-weight:600}
#mapfields{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:0 14px}
iframe{width:100%;height:70vh;border:1px solid var(--line);border-radius:8px;margin-top:12px;background:#fff}
</style></head><body>
<header class="top"><div class="top-in">
  <div class="brand">TrainLab Optimize</div>
  <nav class="tabs" aria-label="Screens">
    <a href="#overview" data-v="overview">Overview</a><a href="#spend" data-v="spend">Spend</a>
    <a href="#live" data-v="live">Live</a><a href="#experiments" data-v="experiments">Experiments</a>
    <a href="#setup" data-v="setup">Setup</a></nav>
  <div class="ws" id="wsname"></div>
</div></header>
<main>

<section class="view" id="v-overview" hidden><div class="brief" id="brief"></div></section>

<section class="view" id="v-spend" hidden><div class="map">
  <aside class="tree" aria-label="Call sites by file"><h3>CALL SITES, SPEND PER MONTH</h3><div id="tree"></div></aside>
  <div id="spendmain"></div></div></section>

<section class="view" id="v-live" hidden><div id="livehead"></div><div id="livebody"></div></section>

<section class="view" id="v-experiments" hidden><div class="ledger">
  <div class="panel"><div class="row-sb"><h2>Experiments</h2><span class="muted" style="font-size:13px">real inputs from a call site, replayed on cheaper models with your API keys</span></div>
    <div class="tblbox"><table><thead><tr><th>Call site</th><th>Current model</th><th class="r">Samples</th><th>Candidates</th><th>Result</th><th class="r">Spent</th><th></th></tr></thead><tbody id="xrows"></tbody></table></div></div>
  <div class="panel" id="newx"><div class="row-sb"><h2>New experiment</h2><span class="muted" style="font-size:13px">nothing is sent until you press Run</span></div>
    <div class="form">
      <label for="x-site">Call site<select id="x-site"></select></label>
      <label for="x-judge">Judge for free text and prose fields<select id="x-judge"><option value="">None</option><option>gpt-4.1</option><option>gpt-4o</option><option>claude-sonnet-4-5</option><option>gemini-2.5-pro</option></select></label>
      <label for="x-max">Maximum cost (USD)<input id="x-max" type="number" min="0.05" step="0.05" value="1.00"></label>
    </div>
    <div class="lbl">Candidates</div><div class="checks" id="x-cands"></div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:14px"><button class="btn ghost" id="x-plan">Check cost</button><button class="btn" id="x-run" disabled>Run experiment</button></div>
    <div id="x-planbox"></div><div class="msg" id="x-msg"></div></div>
  <div class="panel" id="xdetail" hidden></div>
</div></section>

<section class="view" id="v-setup" hidden><div class="wiz">
<section class="wstep" id="s1"><div class="wnum">1</div><div>
<h2>Choose the code to analyse</h2>
<p class="note">Pick one or more project folders. Tests are skipped automatically.</p>
<div class="wrow"><input type="text" id="path" placeholder="/path/to/your/project" aria-label="Folder"><button class="btn ghost" id="browse">Browse…</button><button class="btn ghost" id="add">Add folder</button></div>
<div class="picker" id="picker"></div><div class="chips" id="chosen"></div>
<label for="exclude">Skip these folders (optional, comma-separated)</label><input type="text" id="exclude" placeholder="examples, docs, scripts">
<div class="wrow" style="margin-top:12px"><button class="btn" id="scan">Scan code</button></div><div class="msg" id="m1"></div></div></section>
<section class="wstep" id="s2"><div class="wnum">2</div><div>
<h2>Add what you spend <span class="note">(optional, but needed for savings in dollars)</span></h2>
<div class="wtabs" role="tablist"><button role="tab" aria-selected="true" data-tab="file">Upload a billing or trace file</button><button role="tab" aria-selected="false" data-tab="key">Connect with an admin key</button></div>
<div id="tab-file"><p class="note">Billing or usage exports from any provider, usage API JSON, gateway logs, Langfuse exports, OpenTelemetry or <code class="code-inline">trainlab trace</code> traces. The type is detected. Prompt text in traces is never kept.</p>
<div class="wrow"><input type="file" id="file" accept=".csv,.json,.jsonl" aria-label="Usage file"><select id="kind" style="max-width:220px" aria-label="File type"><option value="auto">Detect automatically</option><option value="aws">AWS cost report</option><option value="azure">Azure cost export</option><option value="gcp">Google Cloud billing</option><option value="langfuse">Langfuse export</option><option value="otel">OpenTelemetry traces</option></select><button class="btn" id="upload">Import file</button></div>
<div id="mapper" hidden><p id="mapmsg"></p><p class="note">Model and either cost or tokens are needed.</p><div id="mapfields"></div><div class="wrow" style="margin-top:8px"><button class="btn" id="mapgo">Import with these columns</button></div></div></div>
<div id="tab-key" hidden><p class="note">Needs an organization <b>admin</b> key (OpenAI, Anthropic) or a <b>management</b> key (OpenRouter) with read access to usage. Used once for this import and never saved.</p>
<div class="wrow"><select id="prov" style="max-width:160px" aria-label="Provider"><option value="openai">OpenAI</option><option value="anthropic">Anthropic</option><option value="openrouter">OpenRouter</option></select><input type="password" id="key" placeholder="Admin or management key" autocomplete="off" aria-label="Key"><select id="days" style="max-width:120px" aria-label="Days"><option>7</option><option selected>30</option><option>60</option><option>90</option></select><button class="btn" id="connect">Import usage</button></div></div>
<div class="chips" id="sources"></div><div class="wrow" style="margin-top:8px"><button class="btn ghost" id="clear">Remove imported usage</button></div><div class="msg" id="m2"></div></div></section>
<section class="wstep" id="s3"><div class="wnum">3</div><div>
<h2>Get the report</h2><div class="wrow"><button class="btn" id="report">Create report</button><span class="note" id="dl"></span></div>
<div class="msg" id="m3"></div><div id="result"></div></div></section>
<p class="note">trainlab-optimize __VERSION__</p></div></section>
</main>
<script>
const T="__TOKEN__",H={"X-Trainlab-Token":T};
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?"":s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
async function api(path,opts={}){const r=await fetch(path,{...opts,headers:{...H,...(opts.headers||{})}});const j=await r.json().catch(()=>({error:"Unexpected response"}));if(!r.ok){const e=new Error(j.error||r.statusText);e.data=j;throw e}return j}
function money(v,dp){if(v==null)return"—";const a=Math.abs(v);if(dp==null)dp=a>=100?0:a>=1?2:a>=0.01?3:4;return"$"+v.toLocaleString("en-US",{minimumFractionDigits:dp,maximumFractionDigits:dp})}
const pct=v=>v==null?"—":Math.round(v*100)+"%";
const k=v=>v==null?"—":v>=1e6?(v/1e6).toFixed(1)+"M":v>=1e3?(v/1e3).toFixed(1)+"k":String(Math.round(v));
const fn=s=>s?s.split(":")[0].split("/").pop()+":"+s.split(":").pop():"";

// ---------------------------------------------------------------- routing
const VIEWS=["overview","spend","live","experiments","setup"];let current=null,liveTimer=null,xTimer=null;
function route(){let v=(location.hash||"#overview").slice(1).split("?")[0];if(!VIEWS.includes(v))v="overview";current=v;
 VIEWS.forEach(n=>{$("v-"+n).hidden=n!==v});document.querySelectorAll("nav.tabs a").forEach(a=>a.setAttribute("aria-current",a.dataset.v===v?"page":"false"));
 clearInterval(liveTimer);clearInterval(xTimer);
 if(v==="overview")loadOverview();if(v==="spend")loadSpend();if(v==="live"){loadLive();liveTimer=setInterval(loadLive,2000)}
 if(v==="experiments"){loadExperiments();xTimer=setInterval(()=>{if(runningJobs)loadExperiments(true)},2000)}if(v==="setup")refresh()}
window.addEventListener("hashchange",route);

// ---------------------------------------------------------------- Overview: the briefing
async function loadOverview(){const el=$("brief");let o;try{o=await api("/api/overview")}catch(e){el.innerHTML=`<p class="err">${esc(e.message)}</p>`;return}
 if(!o.ready){el.innerHTML=`<h1>Start by choosing the code to analyse.</h1><p class="lede">TrainLab finds every place your code calls an LLM, prices it from your billing or traces, and tests cheaper models on your real inputs.</p><p style="margin-top:22px"><a class="btn" href="#setup">Choose your code</a></p>`;return}
 $("wsname").textContent=o.repos.join(", ");
 const nT=o.tested.length;let h1,lede;
 if(!o.has_usage){h1=`${o.call_sites} places in your code call an LLM. Add what you spend to see what they cost.`;lede=`Import a billing export, or record real calls with <code class="code-inline">trainlab trace -- python app.py</code>.`}
 else if(nT){h1=`You spend ${money(o.monthly)} a month on LLM calls. ${nT===1?"One tested change":nT+" tested changes"} would cut ${money(o.validated)} of it.`;
  lede=`Each change swaps a model for a cheaper one on a single line of code, and was run on real inputs from that line before being recommended.`}
 else if(o.untested.length){h1=`You spend ${money(o.monthly)} a month on LLM calls. ${o.untested.length===1?"One call looks":o.untested.length+" calls look"} overpriced, but nothing is tested yet.`;
  lede=`Estimated saving ${money(o.estimated_low)}–${money(o.estimated_high)} a month. An experiment replays real inputs on a cheaper model and tells you whether quality holds.`}
 else{h1=`You spend ${money(o.monthly)} a month on LLM calls, and nothing obvious is overpriced.`;lede=`Every call site is on a model that fits its task, or has been tested.`}
 if(o.has_usage&&o.days<=2)lede+=` <span class="muted">Based on ${o.days} day${o.days===1?"":"s"} of data, scaled to a month.</span>`;
 let steps=[],i=0;
 for(const t of o.tested){const e=t.evidence||{};steps.push(`<li class="step"><span class="i">${++i}</span><div><div class="t">Switch ${esc(t.function||t.site)} to ${esc(e.tested_model||"the tested model")}</div>
   <div class="d"><span class="mono">${esc(t.site)}</span>. Matched the current model on ${e.pass_rate!=null?pct(e.pass_rate):"—"} of ${e.samples||"—"} real inputs, ${pct(e.measured_cost_reduction)} cheaper per call.</div></div><div class="amt">${money(t.high)}<small>a month</small></div></li>`)}
 for(const j of o.running){steps.push(`<li class="step"><span class="i">${++i}</span><div><div class="t">Wait for the test on ${esc(fn(j.site))}</div><div class="d">Running: ${j.calls} of about ${j.total} calls, ${money(j.spent)} spent.</div><div class="prog"><i style="width:${Math.min(100,100*j.calls/Math.max(1,j.total))}%"></i></div></div><div class="amt m">…</div></li>`)}
 for(const u of o.untested){steps.push(`<li class="step"><span class="i">${++i}</span><div><div class="t">Test a cheaper model for ${esc(u.function||u.site)}</div><div class="d">${esc(u.title)}. Estimated from price differences; quality not tested yet.</div>
   <div class="acts"><a class="btn ghost" href="#experiments?site=${encodeURIComponent(u.site)}">Set up the test</a></div></div><div class="amt m">up to ${money(u.high)}<small>if it holds</small></div></li>`)}
 for(const x of o.other.filter(x=>x.kind==="kept")){steps.push(`<li class="step"><span class="i">${++i}</span><div><div class="t">Keep ${esc(x.function||x.site)} as it is</div><div class="d">${esc(x.summary)}</div></div><div class="amt m">tested<small>no saving</small></div></li>`)}
 for(const x of o.other.filter(x=>x.kind==="other").slice(0,3)){steps.push(`<li class="step"><span class="i">${++i}</span><div><div class="t">${esc(x.title)}</div><div class="d">${esc(x.recommendation)}</div></div><div class="amt m">${x.high?"up to "+money(x.high):""}</div></li>`)}
 el.innerHTML=`<h1>${esc(h1)}</h1><p class="lede">${lede}</p>`+(steps.length?`<section><h2>What to do next</h2><ol class="steps">${steps.join("")}</ol></section>`:"")+
  (o.has_usage?`<section class="rule"><h2>How sure are these numbers?</h2><div class="three">
   <div><div class="n">${money(o.monthly)}</div><div class="x">measured: ${pct(o.traced_share)} traced to the line of code that made the call</div></div>
   <div><div class="n save">${money(o.validated)}</div><div class="x">tested on real inputs, cost per call measured</div></div>
   <div><div class="n muted">$0 so far</div><div class="x">confirmed in billing after a change ships</div></div></div></section>`:"")}

// ---------------------------------------------------------------- Spend: the code map
let SP=null,selFile=null;
async function loadSpend(){try{SP=await api("/api/spend")}catch(e){$("spendmain").innerHTML=`<p class="err">${esc(e.message)}</p>`;return}
 if(!SP.ready){$("tree").innerHTML="";$("spendmain").innerHTML=`<div class="empty">No scan yet. <a href="#setup">Choose your code</a> first.</div>`;return}
 const top=SP.files[0]?SP.files[0].spend:0;
 $("tree").innerHTML=SP.files.map((f,fi)=>`<button data-f="${fi}" aria-current="${selFile===fi}"><span>${esc(f.file.split("/").pop())}</span><span class="${f.spend>0&&f.spend>=top*0.3?"hot":""}">${SP.has_usage?money(f.spend):""}</span></button>`+
  f.sites.map(s=>`<button class="ln" data-f="${fi}" data-l="${s.line}"><span>:${s.line} ${esc(s.function||"")}</span><span>${s.spend!=null?money(s.spend):""}</span></button>`).join("")).join("");
 document.querySelectorAll("#tree button").forEach(b=>b.onclick=()=>showFile(+b.dataset.f,b.dataset.l?+b.dataset.l:null));
 if(selFile==null||selFile>=SP.files.length)selFile=0;if(SP.files.length)showFile(selFile);else $("spendmain").innerHTML=`<div class="empty">The scan found no LLM calls.</div>`}
async function showFile(fi,line){selFile=fi;const f=SP.files[fi];document.querySelectorAll("#tree button").forEach(b=>b.setAttribute("aria-current",+b.dataset.f===fi&&!b.dataset.l));
 const sites=Object.fromEntries(f.sites.map(s=>[s.line,s]));let code="";
 try{const src=await api(`/api/source?path=${encodeURIComponent(f.path)}${f.repo?"&repo="+encodeURIComponent(f.repo):""}`);
  const keep=new Set();f.sites.forEach(s=>{for(let i=s.line-6;i<=s.line+3;i++)keep.add(i)});let last=0;
  src.lines.forEach((t,ix)=>{const n=ix+1;if(!keep.has(n))return;if(last&&n>last+1)code+=`<div class="gap">⋯</div>`;last=n;const s=sites[n];
   code+=`<div class="l${s?" hl":""}" ${s?`id="L${n}"`:""}><span class="g">${s&&s.spend!=null?money(s.spend)+"/mo":""}</span><span class="n">${n}</span><span>${esc(t)||" "}</span></div>`})}
 catch(e){code=`<div class="gap">${esc(e.message)}</div>`}
 const cards=f.sites.map(s=>{const per=s.spend!=null&&s.requests?s.spend/s.requests*1000:null;
  return `<div class="card"><div class="loc">${esc(f.path)}:${s.line} ${esc(s.function||"")}</div>
  <div class="big">${per!=null?money(per,2)+` <span class="muted" style="font-size:12px">per 1k calls</span>`:s.spend!=null?money(s.spend):"no usage"}</div>
  <div class="muted" style="font-size:13px">${esc(s.model||"model set at runtime")} · ${s.requests!=null?k(s.requests)+" calls/mo":"—"}${s.method==="exact_trace"?" · traced":s.method==="split_model"?" · shared estimate":""}</div>
  <div>${s.findings.map(x=>`<span class="chip ${x.tier==="validated"&&x.high?"ok":x.tier==="validated"?"no":"mid"}" title="${esc(x.title)}">${esc(x.id)} ${x.tier==="validated"&&x.high?"validated "+money(x.high):x.tier==="validated"?"tested, keep":x.high?"est. "+money(x.high):esc(x.severity)}</span>`).join("")}</div></div>`}).join("");
 $("spendmain").innerHTML=`<div class="row-sb"><span class="mono">${esc(f.file)}</span><span class="muted" style="font-size:13px">${SP.has_usage?money(f.spend)+" a month across "+f.sites.length+" call site"+(f.sites.length>1?"s":"")+"; cost beside the line that makes the call":"no usage imported: add it in Setup to see costs"}</span></div>
  <div class="code">${code}</div><div class="cards">${cards}</div>`+tagTable();
 if(line){const el=$("L"+line);if(el)el.scrollIntoView({block:"center"})}}
function tagTable(){const t=SP.by_tag||{};const rows=Object.entries(t).sort((a,b)=>a[0].split("=")[0].localeCompare(b[0].split("=")[0])||b[1]-a[1]);if(!rows.length)return"";
 return `<div class="panel" style="margin-top:16px"><h2>By tag</h2><div class="tblbox"><table style="min-width:360px"><tbody>${rows.map(([kk,v])=>`<tr><td class="mono">${esc(kk)}</td><td class="r mono">${money(v)}</td></tr>`).join("")}</tbody></table></div></div>`}

// ---------------------------------------------------------------- Live
function spark(a,w=170,h=30){const m=Math.max(1,...a);const x=i=>2+(w-4)*i/(a.length-1),y=v=>h-3-(h-6)*v/m;const p=a.map((v,i)=>x(i).toFixed(1)+","+y(v).toFixed(1)).join(" ");
 return `<svg class="spark" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" aria-hidden="true"><polygon points="2,${h} ${p} ${w-2},${h}" fill="currentColor" fill-opacity=".15"/><polyline points="${p}" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>`}
const hhmm=t=>new Date(t*1000).toLocaleTimeString([], {hour:"2-digit",minute:"2-digit",second:"2-digit"});
async function loadLive(){let L;try{L=await api("/api/live")}catch(e){$("livehead").innerHTML=`<p class="err">${esc(e.message)}</p>`;return}
 if(!L.events){$("livehead").innerHTML="";$("livebody").innerHTML=`<div class="empty"><b>No calls recorded yet.</b><p>Run your app with recording on, in another terminal, from your project folder:</p><p><code>trainlab trace -- python app.py</code></p><p class="muted">This screen follows <code>.trainlab/traces</code> and updates every 2 seconds. No prompt text is recorded.</p></div>`;return}
 const st=L.live?`<span class="dot on"></span>live, updating every 2 s`:`<span class="dot"></span>no calls in the last 2 minutes; showing the hour up to the last call at ${hhmm(L.last_call)}`;
 $("livehead").innerHTML=`<div class="stats"><div class="stat"><div class="k">${st}</div></div></div><div class="stats" style="padding-top:6px">
  <div class="stat"><div class="k">Calls, last minute</div><div class="v">${k(L.calls_per_min)}</div></div>
  <div class="stat"><div class="k">Spend per hour</div><div class="v">${money(L.cost_per_hour)}</div></div>
  <div class="stat"><div class="k">Spend today</div><div class="v">${money(L.cost_today)}</div></div>
  <div class="stat"><div class="k">Errors, ${L.minutes} min</div><div class="v ${L.errors_window?"bad":""}">${L.errors_window}</div></div>
  <div class="stat"><div class="k">Calls, ${L.minutes} min</div><div class="v">${k(L.calls_window)}</div></div></div>`;
 const rows=L.sites.map(s=>`<tr><td><div class="mono" style="font-size:13px">${esc(s.site)}</div><div class="muted" style="font-size:12px">${esc(s.func||"")}${s.tags.feature?" · "+esc(s.tags.feature):""}</div></td>
  <td>${esc(s.model||"")}</td><td style="color:${s.error_rate>0.02?"var(--warn)":"var(--save)"}">${spark(s.per_min)}</td><td class="r mono">${k(s.per_min[s.per_min.length-1])}</td>
  <td class="r mono">${s.p95!=null?(s.p95/1000).toFixed(2)+" s":"—"}</td><td class="r mono ${s.error_rate>0.02?"warn":"muted"}">${(s.error_rate*100).toFixed(1)}%</td><td class="r mono">${money(s.cost_per_hour)}</td></tr>`).join("");
 const tail=L.recent.map(r=>`${hhmm(r.ts)}  <b>${esc(r.site.padEnd(28).slice(0,28))}</b> ${esc((r.model||"").padEnd(22).slice(0,22))} ${String(r.in).padStart(6)} in ${String(r.out).padStart(5)} out ${r.ms!=null?String(Math.round(r.ms)).padStart(6)+" ms":"        "} ${r.cost!=null?money(r.cost,5).padStart(9):""}  ${r.ok?"ok":`<span class="e">${esc(r.error||"error")}</span>`}`).join("\n");
 $("livebody").innerHTML=`<div class="panel" style="padding:0;margin-top:14px"><div class="tblbox"><table><thead><tr><th style="padding-left:16px">Call site</th><th>Model</th><th>Calls per minute, ${L.minutes} min</th><th class="r">Last min</th><th class="r">p95</th><th class="r">Errors</th><th class="r" style="padding-right:16px">$/hour</th></tr></thead><tbody>${rows}</tbody></table></div></div>
  <div class="panel" style="margin-top:14px"><div class="row-sb"><h2>Latest calls</h2><span class="muted" style="font-size:13px">counts and names only; no prompt text is recorded</span></div><div class="tail">${tail}</div></div>`}

// ---------------------------------------------------------------- Experiments
let XS=null,runningJobs=0,planned=null,selX=null;
async function loadExperiments(quiet){try{XS=await api("/api/experiments")}catch(e){$("x-msg").innerHTML=`<span class="err">${esc(e.message)}</span>`;return}
 runningJobs=XS.jobs.filter(j=>j.status==="running").length;
 const jobRows=XS.jobs.filter(j=>j.status!=="done").map(j=>`<tr><td class="mono">${esc(j.site)}</td><td colspan="3">${j.status==="running"?`running: ${j.calls} of about ${j.total} calls, ${money(j.spent)} of ${money(j.budget,2)}<div class="prog"><i style="width:${Math.min(100,100*j.calls/Math.max(1,j.total))}%"></i></div>`:`<span class="err">${esc(j.error)}</span>`}</td><td></td><td></td><td></td></tr>`).join("");
 const seen={},older={};const latest=XS.experiments.filter(x=>{const key=x.site||x.id;if(seen[key]){older[key]=(older[key]||0)+1;return false}seen[key]=1;return true});
 const rows=latest.map(x=>{const best=x.candidates.find(c=>c.model===x.recommended);
  const passMore=x.candidates.some(c=>c.verdict==="PASS"&&c.costs_more);
  const res=best?`<span class="verdict v-PASS">${esc(best.model)} validated</span>`
   :x.candidates.every(c=>c.verdict==="FAIL"||(c.verdict==="PASS"&&c.costs_more))?`<span class="verdict v-FAIL">keep current${passMore?": cheaper options cost more":""}</span>`
   :`<span class="verdict v-INCONCLUSIVE">${x.stopped?"stopped: "+esc(x.stopped):"no clear winner yet"}</span>`;
  const n=older[x.site||x.id];
  return `<tr><td><div class="mono" style="font-size:13px">${esc(x.site||"samples file")}</div><div class="muted" style="font-size:12px">${esc(x.created||"")}${x.judge?" · judge "+esc(x.judge):""}${n?` · ${n} earlier run${n>1?"s":""}`:""}</div></td><td class="nw">${esc(x.baseline)}</td><td class="r mono">${x.samples}</td>
   <td>${x.candidates.map(c=>`<span class="chip ${c.verdict==="PASS"&&!c.costs_more?"ok":c.verdict==="FAIL"?"no":"mid"}">${esc(c.model)} ${c.passed!=null?c.passed+"/"+c.scored:""}</span>`).join("")}</td>
   <td>${res}</td><td class="r mono">${money(x.spent)}</td><td class="r"><button class="btn ghost" data-x="${esc(x.id)}">View</button></td></tr>`}).join("");
 $("xrows").innerHTML=jobRows+rows||`<tr><td colspan="7" class="muted">No experiments yet. Set one up below.</td></tr>`;
 document.querySelectorAll("#xrows button[data-x]").forEach(b=>b.onclick=()=>showX(b.dataset.x));
 if(quiet)return;
 const want=new URLSearchParams((location.hash.split("?")[1]||"")).get("site");
 $("x-site").innerHTML=XS.sites.map(s=>`<option value="${esc(s.site)}" ${s.samples?"":"disabled"} ${want===s.site?"selected":""}>${esc(s.site)}: ${esc(s.model)} · ${s.samples?s.samples+" samples":"no samples captured"}${s.spend!=null?" · "+money(s.spend)+"/mo":""}</option>`).join("")||`<option disabled>No call site with a cheaper model to test</option>`;
 if(!want){const first=XS.sites.find(s=>s.samples);if(first)$("x-site").value=first.site}
 candBoxes();if(!XS.sites.some(s=>s.samples))$("x-msg").innerHTML=`<span class="muted">No captured samples yet. Record some with <code class="code-inline">trainlab trace --capture -- python app.py</code>.</span>`}
function candBoxes(){const s=(XS.sites||[]).find(x=>x.site===$("x-site").value);planned=null;$("x-run").disabled=true;$("x-planbox").innerHTML="";
 $("x-cands").innerHTML=s?s.candidates.map(c=>`<label><input type="checkbox" value="${esc(c)}" checked> ${esc(c)}</label>`).join(""):""}
$("x-site").onchange=candBoxes;$("x-judge").onchange=()=>{planned=null;$("x-run").disabled=true};$("x-max").oninput=$("x-judge").onchange;
function xbody(){return{site:$("x-site").value,judge:$("x-judge").value,max_cost:+$("x-max").value,candidates:[...document.querySelectorAll("#x-cands input:checked")].map(i=>i.value)}}
$("x-plan").onclick=async()=>{$("x-msg").textContent="";$("x-planbox").innerHTML="";try{const p=await api("/api/experiments/plan",{method:"POST",body:JSON.stringify(xbody())});planned=p;
 const block=(p.blocking||[]).length||p.problems.some(x=>/no API key|no price/.test(x));
 $("x-planbox").innerHTML=`<div class="planbox"><dl><dt>Samples</dt><dd>${p.samples} (${p.with_reference} with the current model's answer)</dd>
  <dt>Estimated cost</dt><dd class="mono">${money(p.est_cost_usd)}, at most ${money(p.max_cost_usd)} (budget ${money(p.budget_usd,2)})</dd>
  <dt>Calls</dt><dd>${p.models.filter(m=>m.calls).map(m=>esc(m.model)+" "+m.calls+(m.role==="judge"?" (judge, at most)":"")).join(", ")}</dd>
  <dt>Samples go to</dt><dd>${Object.keys(p.data_goes_to).map(esc).join(", ")}</dd></dl>
  ${p.problems.map(x=>`<p class="${/no API key|no price|needed|--judge/.test(x)?"err":"muted"}" style="margin:8px 0 0">${esc(x)}</p>`).join("")}
  ${p.new_hosts.length?`<p style="margin:10px 0 0"><label><input type="checkbox" id="x-hosts"> Send samples to ${p.new_hosts.map(esc).join(", ")}, which doesn't serve this call today</label></p>`:""}</div>`;
 $("x-run").disabled=!!block}catch(e){$("x-msg").innerHTML=`<span class="err">${esc(e.message)}</span>`}};
$("x-run").onclick=async()=>{const b={...xbody(),confirm_hosts:!!($("x-hosts")&&$("x-hosts").checked)};$("x-run").disabled=true;
 try{await api("/api/experiments/run",{method:"POST",body:JSON.stringify(b)});$("x-msg").innerHTML=`<span class="ok">Started. Progress shows in the list above.</span>`;$("x-planbox").innerHTML="";loadExperiments(true)}
 catch(e){$("x-msg").innerHTML=`<span class="err">${esc(e.message)}</span>`;$("x-run").disabled=false}};
async function showX(id){let x;try{x=await api("/api/experiments/"+encodeURIComponent(id))}catch(e){return}selX=x;const d=$("xdetail");d.hidden=false;
 const ROLE={fact:"facts (must match)",prose:"prose",judgment:"judgment, not scored",ignore:"ignored"};
 const cands=x.candidates.map(c=>{const fl=c.fields?["fact","prose","judgment"].map(r=>{const fs=Object.entries(c.fields).filter(([,v])=>v.role===r);return fs.length?`${ROLE[r]}${r==="prose"?(x.prose_scoring==="judge"?" (judged)":" (word overlap)"):""}: `+fs.map(([kk,v])=>`${esc(kk)} ${v.agree}/${v.n}`).join(", "):""}).filter(Boolean).join("<br>"):"";
  return `<tr><td>${esc(c.model)}</td><td><span class="verdict v-${esc(c.verdict)}">${esc(c.verdict)}</span>${c.costs_more?` <span class="chip no">costs more</span>`:""}${c.stopped_early?` <span class="chip">stopped early</span>`:""}</td><td class="r mono">${c.passed}/${c.scored}</td><td class="r mono">${pct(c.lower_95)}</td><td class="r mono">${money(c.cost_per_call_usd,5)}</td><td class="r mono">${c.savings_pct!=null?(c.savings_pct>0?pct(c.savings_pct):"+"+pct(-c.savings_pct)+" cost"):"—"}</td></tr>`+(fl?`<tr><td></td><td colspan="5" class="fields">${fl}</td></tr>`:"")}).join("");
 const ex=[];x.candidates.forEach(c=>Object.entries(c.examples||{}).forEach(([f,items])=>items.forEach(it=>ex.push({model:c.model,field:f,role:((c.fields||{})[f]||{}).role,...it}))));
 d.innerHTML=`<div class="row-sb"><div><div class="mono muted" style="font-size:12px">${esc(x.site||"")} · ${x.samples} samples · ${esc(x.task)} · reference: ${esc(x.reference)}</div>
  <h2 style="margin-top:4px;font-size:17px">${x.recommended?`${esc(x.recommended)} keeps quality against ${esc(x.baseline)}`:`No cheaper model kept quality against ${esc(x.baseline)}`}</h2></div><button class="btn ghost" id="xclose">Close</button></div>
  <div class="tblbox"><table><thead><tr><th>Candidate</th><th>Verdict</th><th class="r">Matched</th><th class="r">95% low (bar ${pct(x.threshold)})</th><th class="r">Cost/call</th><th class="r">Saving</th></tr></thead><tbody>${cands}</tbody></table></div>
  ${ex.length?`<div class="lbl" style="margin-top:16px">Where they disagreed (${ex.length} shown)</div><div class="split"><div class="slist" id="xlist">${ex.map((e,i)=>`<button data-i="${i}" aria-current="${i===0}"><span class="f">${esc(e.model)} · ${esc(e.field)}${e.role?" ("+esc(e.role)+")":""}</span><span class="s">${esc(e.candidate)}</span></button>`).join("")}</div><div id="xone"></div></div>`:`<p class="muted">No disagreements stored for this experiment.</p>`}`;
 $("xclose").onclick=()=>{d.hidden=true};
 document.querySelectorAll("#xlist button").forEach(b=>b.onclick=()=>{document.querySelectorAll("#xlist button").forEach(z=>z.setAttribute("aria-current",z===b));one(ex[+b.dataset.i],x)});
 if(ex.length)one(ex[0],x);d.scrollIntoView({behavior:"smooth",block:"start"})}
function one(e,x){$("xone").innerHTML=`<div class="lbl">Input (as captured, redacted)</div><pre class="box">${esc(e.input)}</pre>
 <div class="cmp"><div><div class="lbl">${esc(x.reference==="expected answers"?"expected":x.baseline+" (current)")}</div><pre class="box">${esc(e.reference)}</pre></div>
 <div><div class="lbl">${esc(e.model)}${e.why?" · "+esc(e.why):""}</div><pre class="box">${esc(e.candidate)}</pre></div></div>
 ${e.role?`<p class="muted" style="font-size:13px">Field <span class="mono">${esc(e.field)}</span>${e.role==="judgment"?": a judgment call, shown but not scored":e.role==="prose"?": compared on meaning":": must match"}.</p>`:""}`}

// ---------------------------------------------------------------- Setup (scan, import, report)
let chosen=[];
function msg(id,text,cls){const el=$(id);el.textContent=text||"";el.className="msg "+(cls||"")}
function render(st){$("chosen").innerHTML=chosen.map((p,i)=>`<span>${esc(p)} <a href="#setup" data-i="${i}" aria-label="remove">×</a></span>`).join("");
 document.querySelectorAll("#chosen a").forEach(a=>a.onclick=e=>{e.preventDefault();chosen.splice(+a.dataset.i,1);render(st)});
 if(st.scan){$("s1").classList.add("done");$("wsname").textContent=st.scan.repos.join(", ");msg("m1",`Found ${st.scan.call_sites} LLM call sites in ${st.scan.files} files (${st.scan.repos.join(", ")}).`+(st.scan.runtime_models?` ${st.scan.runtime_models} pick their model at runtime.`:""),"ok")}
 $("sources").innerHTML=st.usage.map(u=>`<span>${esc(u.source)} (${u.records} rows)</span>`).join("");
 $("s2").classList.toggle("done",!!st.usage.length);
 $("dl").innerHTML=st.reports.filter(n=>!n.endsWith(".html")).map(n=>`<a href="/report/${n}?t=${T}">${n.split(".").pop().toUpperCase()}</a>`).join(" ")}
async function refresh(){const st=await api("/api/state");if(!chosen.length&&!st.scan)$("path").value=st.start_dir;render(st);return st}
$("add").onclick=()=>{const v=$("path").value.trim();if(v&&!chosen.includes(v))chosen.push(v);$("path").value="";refresh()};
$("browse").onclick=async()=>{const pk=$("picker");if(pk.style.display==="block"){pk.style.display="none";return}await openDir($("path").value)};
async function openDir(p){try{const r=await api("/api/ls?path="+encodeURIComponent(p));$("path").value=r.path;const pk=$("picker");pk.style.display="block";
 pk.innerHTML=`<div data-p="${esc(r.parent)}">⬑ ..</div>`+r.dirs.map(d=>`<div data-p="${esc(r.path+"/"+d)}">${esc(d)}</div>`).join("");
 pk.querySelectorAll("div").forEach(el=>el.onclick=()=>openDir(el.dataset.p))}catch(e){msg("m1",e.message,"err")}}
$("scan").onclick=async e=>{const b=e.target;const paths=chosen.length?chosen:[$("path").value.trim()];b.disabled=true;msg("m1","Scanning…");
 try{render(await api("/api/scan",{method:"POST",body:JSON.stringify({paths,exclude:$("exclude").value.split(",")})}));$("picker").style.display="none"}catch(err){msg("m1",err.message,"err")}finally{b.disabled=false}};
document.querySelectorAll(".wtabs button").forEach(t=>t.onclick=()=>{document.querySelectorAll(".wtabs button").forEach(x=>x.setAttribute("aria-selected",x===t));$("tab-file").hidden=t.dataset.tab!=="file";$("tab-key").hidden=t.dataset.tab!=="key"});
const MAPF=[["model","Model (required)"],["cost_usd","Cost"],["input_tokens","Input tokens"],["output_tokens","Output tokens"],["date","Date"],["api_key","API key"],["project","Project / team"],["provider","Provider"]];
async function upload(btn,map){const f=$("file").files[0];if(!f){msg("m2","Choose a file first.","err");return}btn.disabled=true;msg("m2","Importing…");
 const hd={"X-Filename":encodeURIComponent(f.name),"X-Kind":$("kind").value};if(map)hd["X-Map"]=encodeURIComponent(JSON.stringify(map));
 try{render(await api("/api/import/file",{method:"POST",headers:hd,body:await f.arrayBuffer()}));$("mapper").hidden=true;msg("m2","Imported "+f.name,"ok")}
 catch(err){if(err.data&&err.data.columns){msg("m2","");$("mapmsg").textContent=err.message;showMapper(err.data.columns)}else msg("m2",err.message,"err")}finally{btn.disabled=false}}
function showMapper(cols){const opts=`<option value="">—</option>`+cols.map(c=>`<option>${esc(c)}</option>`).join("");
 $("mapfields").innerHTML=MAPF.map(([kk,l])=>`<div><label for="map-${kk}">${l}</label><select id="map-${kk}">${opts}</select></div>`).join("");$("mapper").hidden=false}
$("upload").onclick=e=>upload(e.target,null);
$("mapgo").onclick=e=>{const m={};MAPF.forEach(([kk])=>{const v=$("map-"+kk).value;if(v)m[kk]=v});if(!m.model){msg("m2","Pick the model column.","err");return}upload(e.target,m)};
$("connect").onclick=async e=>{e.target.disabled=true;msg("m2","Fetching usage…");
 try{render(await api("/api/import/api",{method:"POST",body:JSON.stringify({provider:$("prov").value,key:$("key").value,days:$("days").value})}));$("key").value="";msg("m2","Usage imported.","ok")}catch(err){msg("m2",err.message,"err")}finally{e.target.disabled=false}};
$("clear").onclick=async()=>{render(await api("/api/usage/clear",{method:"POST"}));msg("m2","Usage removed.","ok")};
$("report").onclick=async e=>{e.target.disabled=true;msg("m3","Analysing…");
 try{const st=await api("/api/report",{method:"POST",body:JSON.stringify({formats:["html","json","md","pdf"]})});render(st);const s=st.summary;msg("m3",(st.warnings||[]).join(" "),"note");
  $("result").innerHTML=(s.monthly_total!=null?`<p><b>${money(s.monthly_total)}</b> per month, <b class="save">${money(s.savings_low)}–${money(s.savings_high)}</b> potential saving, ${s.findings} findings</p>`:`<p>${s.findings} findings from code. Add usage in step 2 to put a price on them.</p>`)+
  `<iframe title="Report" src="/report/trainlab-report.html?t=${T}&v=${Date.now()}"></iframe><p class="note"><a target="_blank" href="/report/trainlab-report.html?t=${T}">Open report in a new tab</a></p>`;$("s3").classList.add("done")}
 catch(err){msg("m3",err.message,"err")}finally{e.target.disabled=false}};
api("/api/state").then(st=>{if(st.scan)$("wsname").textContent=st.scan.repos.join(", ")}).catch(()=>{});
route();
</script></body></html>"""
