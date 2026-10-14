# TrainLab Optimize

Find out where your LLM money goes, line by line, and what to change.

`trainlab` scans your repositories for LLM calls to any provider, imports your actual usage
(usage APIs, billing exports, cloud cost exports, gateway logs or traces), links the spend
to the code that causes it, and writes a report of concrete changes with estimated monthly
savings.

- **Runs locally.** Your code and usage data never leave your machine. The HTML report is a
  single file with no external requests.
- **Zero dependencies.** Pure Python standard library (3.10+).
- **Read-only.** It never executes your code and never calls an LLM. No LLM key is needed.

## Quick start

```bash
pip install -e .                      # PyPI release pending

cd your-repo
trainlab scan .                       # find LLM call sites
trainlab import billing.csv           # any usage file; the type is detected
trainlab report                       # writes .trainlab/report/trainlab-report.html
```

Without usage data the report still lists every call site and code-level issues; it just
can't put a price on them.

**No terminal experience?** Run `trainlab ui`. A page opens in your browser with three
steps: pick the code folder, drop in usage files (or paste an admin key), get the report.
It only listens on 127.0.0.1 and keys are used once, never stored.

**One step:** `trainlab run . --usage billing.csv --usage traces.json`

**Demo** (a small app with deliberate waste; output is in `examples/demo_report/`):

```bash
trainlab run examples/demo_app --usage examples/demo_usage/billing_export.csv
```

## Usage data it understands

| Source | How | What you need |
|---|---|---|
| OpenAI Usage API | `trainlab import openai` | `OPENAI_ADMIN_KEY` (organization admin key, read-only usage) |
| Anthropic Usage & Cost API | `trainlab import anthropic` | `ANTHROPIC_ADMIN_KEY` (admin key) |
| OpenRouter activity API | `trainlab import openrouter` | `OPENROUTER_MANAGEMENT_KEY` (management key) |
| Any provider's billing or usage CSV | `trainlab import FILE.csv` | columns for model and tokens or cost (names matched loosely) |
| OpenRouter activity export | `trainlab import activity.csv` | CSV from the OpenRouter dashboard |
| AWS Cost and Usage Report (Bedrock) | `trainlab import cur.csv` | CUR legacy or 2.0 export |
| Azure cost export (Azure OpenAI / AI Foundry) | `trainlab import cost.csv` | Cost Management export |
| Google Cloud billing (Vertex AI / Gemini) | `trainlab import billing.csv` | BigQuery billing export saved as CSV |
| Langfuse | `trainlab import observations.json` | observations export (JSON, JSONL or CSV) |
| OpenTelemetry | `trainlab import spans.json` | OTLP JSON spans with GenAI attributes |
| Saved usage API responses | `trainlab import response.json` | the JSON returned by the APIs above |

The file type is detected from its contents; name it explicitly if needed
(`trainlab import azure cost.csv`). Prompt text inside traces is never stored.

**Exports with unfamiliar column names** (Alibaba Cloud bills, gateway logs, a finance
team's sheet): the UI lists the file's columns and asks which is which. On the command line:
`trainlab import bill.csv --map model="Billing Item",cost_usd="Pretax Amount (USD)",api_key="Instance ID"`.
Bill lines such as `qwen-max input_tokens` are read as model `qwen-max`.

### Spend per API key

The OpenAI, Anthropic and OpenRouter imports break spend down per API key, as do CSVs with
a key column. The report has a "Spend by API key" table.

One key used by several applications can't be split by billing. How exact the numbers
are depends on how keys are set up:

| Setup | What the report can say |
|---|---|
| One key or project per application | Exact cost per application, from billing |
| Shared key, but each application uses different models | Exact: the code shows which application uses which model |
| Shared key, same model in several scanned repos | Spend is split evenly between the call sites and flagged (R904), with the keys involved |
| Shared key, but only one of the repos was scanned | The rest shows as spend with no matching code (R902), with the keys that carry it |
| Any of the above with `trainlab.instrument()` (v0.2) | Exact per line, whatever the key setup |

**When sources overlap**, nothing is counted twice:

- Billing sources (usage APIs, CSVs, cloud exports) are the authority on totals. If two of
  them cover the same model on the same day through the same channel, one is counted, in
  this order: billed cost, then usage API, then token-only CSV. The report says which
  source was set aside. The same model bought directly, on Azure, on Bedrock or through
  OpenRouter is different traffic and is always counted separately.
- Traces only decide *where* billed spend comes from. Models that appear only in traces
  are added to the total.
- Each source is scaled to 30 days on its own period (30 days of billing plus 7 days of
  traces is not a 37-day window).

## What it finds

| Rule | Finding | Typical fix |
|---|---|---|
| R101 | Premium model on a simple task (extraction, classification, structured output) | Test an economy model; fine-tuned small model for high volume |
| R102 | Model at or near its provider deprecation date | Migrate before shutdown |
| R103 | Large system prompt that is never cached (dynamic content first, or no `cache_control`) | Reorder prompt, add cache breakpoint |
| R104 | Sequential LLM calls in a `for` loop or comprehension | Batch API (~50% cheaper) if latency-tolerant |
| R105 | No output token limit | Set `max_completion_tokens` / `max_tokens` |
| R106 | Unbounded retries around an LLM call | Add a stop condition |
| R107 | The same long prompt duplicated across files | One shared, cached prompt |
| R108 | Large average context per request with little caching | Trim retrieval, cache the stable part |
| R109 | Embeddings computed one item at a time in a loop | Send a list per request |
| R904 | One model's spend shared by several repositories (one key, can't be split by billing) | A key or project per application, or instrument |
| R905 | A model priced from another host's list (its maker publishes no price) | Import billing with the cost column |
| R9xx | Other coverage gaps: runtime-selected models, spend with no matching code, unpriced models | Instrument, or scan the other repos |

Cheaper-model suggestions stay with the provider you call: a Groq user gets Groq models, a
DashScope user gets Qwen models. Calls through a router (OpenRouter, Vercel AI Gateway,
Portkey) get suggestions from any vendor, since switching is one line there.

Findings in examples, notebooks and docs are capped at low severity. Tests are skipped
unless `--include-tests` is given.

## How savings are counted

Every number carries a confidence tier:

| Tier | Meaning |
|---|---|
| From code | Pattern found in source, no cost attached |
| Estimated | Measured spend × price difference. Quality not yet tested |
| Validated | Tested on your real inputs (`trainlab experiment`, coming in v0.3) |
| Realized | Confirmed in production billing |

Recommendations on the same call site overlap, so the headline total only counts the
largest saving per call site.

**Attribution** links spend to code in three ways, best first:

1. **Exact (traced):** traces carrying `code.file.path` / `code.line.number` put spend on
   that line.
2. **By model:** one call site uses a model, so it gets all of that model's spend.
3. **Split:** several call sites share a model; spend is split evenly and marked "shared".

## Providers

The scanner works out where each call goes, in this order:

1. The SDK or framework class: `cohere.ClientV2`, `Groq()`, `Mistral()`, `ChatTongyi`, `ChatOllama`,
   LlamaIndex `OpenRouter`, Vercel AI `openrouter("...")`, ...
2. The client's base URL or API-key variable: `OpenAI(base_url="https://openrouter.ai/api/v1")`,
   `api_key=os.environ["DASHSCOPE_API_KEY"]`, including values set in constants or env defaults.
3. Hosts, key variables and provider packages elsewhere in the same file, when they point at
   exactly one provider.
4. The model name: `openrouter/...`, `groq:...` (aisuite), or the model family's maker.

Recognised providers include OpenAI, Azure OpenAI and AI Foundry, Anthropic, Google Gemini and
Vertex AI, AWS Bedrock, OpenRouter, Vercel AI Gateway, Portkey, Alibaba DashScope (Qwen), DeepSeek,
Moonshot (Kimi), Zhipu (GLM), MiniMax, ByteDance Volcengine (Doubao), Baidu Qianfan (ERNIE),
Tencent Hunyuan, iFlytek Spark, SiliconFlow, Mistral, Cohere, xAI, Perplexity, Groq, Together,
Fireworks, DeepInfra, Novita, Cerebras, SambaNova, NVIDIA NIM, Hugging Face, Replicate,
Cloudflare Workers AI, Voyage, AI21, Databricks, watsonx, and self-hosted Ollama and LM Studio.
Self-hosted calls are listed but never priced (there's no per-token bill). Adding a provider
is one line in `scanner/providers.py`.

**Prices** cover about 3,100 text and embedding models across all of these. A model is priced
at its maker's list price when the catalog has it. Otherwise the same model's price at
another host is used, and the report says so (R905). `trainlab prices update` refreshes the
list and also merges OpenRouter's public model list (about 300 models at OpenRouter's prices).

## Languages and SDKs

| | Python (AST, high confidence) | Other languages (pattern-based) |
|---|---|---|
| OpenAI-compatible APIs (any provider above) | chat, responses, embeddings, batches | JS/TS; Go (openai-go, go-openai); Java/Kotlin (openai-java, Azure, aallam); C# (OpenAI .NET, Azure.AI.OpenAI, Azure.AI.Inference); Ruby (ruby-openai); PHP (openai-php); Rust (async-openai) |
| Anthropic | messages, batches | JS/TS; Go; Java; Ruby; PHP |
| AWS Bedrock | invoke_model, converse (boto3, aioboto3) | JS commands; Go; Ruby; Rust |
| Google Gemini / Vertex | generate_content, embed_content, `GenerativeModel` | JS/TS; Go genai |
| Native vendor SDKs | DashScope (`Generation.call`), Mistral, Cohere, Ollama, Hugging Face, Replicate, Voyage, xAI, Zhipu, Volcengine Ark, Groq, Together, Fireworks, Cerebras | Mistral, Cohere, Ollama, Hugging Face, Replicate (JS/TS) |
| LiteLLM / aisuite | completion, embedding (incl. aliases, `functools.partial`) | — |
| Frameworks | LangChain and LlamaIndex (all provider classes), Semantic Kernel, wrapper configs | Vercel AI SDK (all providers), LangChain.js, LangChain4j, Spring AI, langchaingo, Semantic Kernel (.NET), RubyLLM, langchainrb, Prism, LLPhant, rig, genai |
| Raw HTTP to provider APIs | `/chat/completions`, `/v1/messages`, `/v1/responses`, `/embeddings`, `:generateContent`, DashScope, Ollama and Cohere native paths | same |
| Injected or factory-built clients | `client.chat.completions.create(...)` without an SDK import | — |

Also handled:

- **Jupyter notebooks:** each cell is parsed on its own, so a broken cell or top-level
  `await` doesn't hide the rest. Findings point to the cell (`nb.ipynb:cell 4:12`).
- **Prompts and model names defined in other files** (`from .prompts import SYSTEM`).
- Model names in config files (YAML, JSON, TOML, `.env`).
- Server routes that *serve* an OpenAI-compatible API are not counted as calls.
- Generated, minified, vendored and oversized files (>1 MB) are skipped. Files in any
  encoding are read without failing.

## Several repositories

```bash
trainlab scan ../api ../worker ../web          # one combined analysis
trainlab scan ../ml-jobs --append              # add another repo later
```

The report shows spend and findings per repository.

## Reports

`trainlab report --format html,json,md,pdf`

- **HTML:** self-contained, light and dark mode, prints cleanly.
- **JSON:** stable schema (`schema_version`) for other tools.
- **Markdown:** for pull request comments and wikis.
- **PDF:** uses Chrome, Edge or Chromium if one is installed. Otherwise open the HTML and
  print to PDF.

## In CI

`trainlab report --fail-on high` exits with code 3 when findings at or above that severity
exist. A ready-made GitHub Action posts the summary on pull requests: see
[docs/github-action.md](docs/github-action.md).

## Commands

```
trainlab scan [PATH ...] [--append] [--exclude GLOB] [--include-tests]
trainlab import FILE [--map FIELD=COLUMN,...]          # type detected
trainlab import {csv,aws,azure,gcp,langfuse,otel} FILE # force a type
trainlab import {openai,anthropic,openrouter} [--days 30] [--key KEY | --file saved.json]
trainlab report [--out DIR] [--format html,json,md,pdf] [--fail-on high|medium|low]
trainlab run [PATH ...] [--usage FILE ...] [--exclude GLOB] [--format ...] [--fail-on ...]
trainlab ui [--port N] [--no-browser]
trainlab prices show MODEL | prices info | prices update
```

All state lives in `.trainlab/` (git-ignored automatically). Use `--workspace DIR` to
keep several analyses apart.

Prices ship with the package. `trainlab prices update` downloads the latest list to
`~/.cache/trainlab/prices.json` (or `$TRAINLAB_PRICES`), including OpenRouter's public
model list. That and the usage API imports are the only network calls the tool makes,
each only when you run it.

## Benchmark

`scripts/eval_repos.py` scans real repositories and compares detected call sites with a
plain-text search for SDK call patterns. It was run on 41 open-source projects in October
2026 (shallow clones, 2 CPU cores):

- 9 typical AI apps and the 12 largest AI codebases we could find;
- 12 corner cases: cookbooks full of notebooks, SDK repos, and Go, Java, Kotlin, C#, Ruby,
  PHP and Rust projects;
- 8 multi-provider projects: Qwen-Agent (DashScope), Cohere Toolkit, the Mistral cookbook,
  ollama-python, OpenRouter's examples, CAMEL, NextChat and opencode.

| Repo | Files | Scan time | Call sites |
|---|---|---|---|
| lobe-chat | 17,765 | 20.7s | 100 |
| dify | 12,957 | 14.2s | 15 |
| onyx | 7,595 | 7.2s | 28 |
| opencode (40 providers) | 5,675 | 6.0s | 50 |
| llama_index | 5,278 | 12.3s | 1,100 (about 900 in notebooks) |
| semantic-kernel (C#, Python) | 4,060 | 5.5s | 658 |
| langchain | 2,762 | 6.0s | 77 |
| camel | 1,268 | 2.7s | 34 |
| openai-cookbook (notebooks) | 1,237 | 4.4s | 544 |
| Mistral cookbook (notebooks) | 246 | 2.2s | 220 |
| …31 more | | | |

Every real SDK call found by the text search was also found by the scanner. The
remaining text matches were docstrings and markdown, server route definitions, a
framework's own internal helper methods, generated clients for the app's own API, other
APIs with the same method names (Twilio and Plivo `messages.create`), and mocks in test
files.

## Known limits

- Pattern-based languages (everything except Python) find calls and literal model names
  reliably, but can't follow values across files.
- Models chosen at runtime (from a database or user input) can't be priced per call site.
  The report lists them as a coverage gap; traces close it.
- Model names in cloud cost exports are parsed from meter or SKU names, which vendors
  change. If a name can't be matched to a model, its billed cost still counts, but it gets
  no model-switch recommendation.
- The usage API importers have been tested against the documented response formats, not
  yet against live organization accounts.
- Frameworks that pick the provider at runtime from configuration (CAMEL's model factory,
  a gateway service that serves many tenants) show one call site for all providers. Traces
  or `trainlab.instrument()` split them.
- OpenRouter's activity API covers the last 30 days. BYOK inference (billed on your own
  provider keys through OpenRouter) is counted as spend; if you also import that provider's
  own billing, the same calls appear in both.
- Prices are list prices. Negotiated discounts aren't known unless the billing export
  carries the billed cost, which then takes precedence.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
python scripts/build_catalog.py        # refresh bundled prices from LiteLLM's price map
python scripts/make_demo_usage.py      # regenerate demo usage fixtures
python scripts/eval_repos.py DIR REPO...   # benchmark against cloned repos
```

Price data comes from [BerriAI/litellm](https://github.com/BerriAI/litellm)'s
`model_prices_and_context_window.json` (MIT); attribution and build date are stored in
`src/trainlab_optimize/catalog/prices.json`.

## Roadmap

- **v0.2** `trainlab.instrument()`: one line in your app tags each LLM call with its code
  location for exact attribution
- **v0.3** `trainlab experiment`: replay real inputs against cheaper candidates with
  quality gates; `trainlab export` for a TrainLab optimization engagement

## License

Apache 2.0
