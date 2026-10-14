# TrainLab Optimize

Find out where your LLM money goes, line by line, and what to change.

`trainlab` scans your repositories for LLM calls, imports your actual usage (usage APIs,
billing exports, cloud cost exports or traces), links the spend to the code that causes
it, and writes a report of concrete changes with estimated monthly savings.

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
| Any billing CSV | `trainlab import FILE.csv` | columns for model and tokens or cost (names are matched loosely) |
| AWS Cost and Usage Report (Bedrock) | `trainlab import cur.csv` | CUR legacy or 2.0 export |
| Azure cost export (Azure OpenAI / AI Foundry) | `trainlab import cost.csv` | Cost Management export |
| Google Cloud billing (Vertex AI / Gemini) | `trainlab import billing.csv` | BigQuery billing export saved as CSV |
| Langfuse | `trainlab import observations.json` | observations export (JSON, JSONL or CSV) |
| OpenTelemetry | `trainlab import spans.json` | OTLP JSON spans with GenAI attributes |
| Saved usage API responses | `trainlab import response.json` | the JSON returned by the APIs above |

The file type is detected from its contents; name it explicitly if needed
(`trainlab import azure cost.csv`). Prompt text inside traces is never stored.

**When sources overlap**, nothing is counted twice:

- Billing sources (usage APIs, CSVs, cloud exports) are the authority on totals. If two of
  them cover the same model on the same day, one is counted, in this order: billed cost,
  then usage API, then token-only CSV. The report says which source was set aside.
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
| R9xx | Coverage gaps: runtime-selected models, spend with no matching code | Instrument, or scan the other repos |

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

## Languages and SDKs

| | Python (AST, high confidence) | Other languages (pattern-based) |
|---|---|---|
| OpenAI / Azure OpenAI | chat, responses, embeddings, batches | JS/TS; Go (openai-go, go-openai); Java/Kotlin (openai-java, Azure, aallam); C# (OpenAI .NET, Azure.AI.OpenAI, Azure.AI.Inference); Ruby (ruby-openai); PHP (openai-php); Rust (async-openai) |
| Anthropic | messages, batches | JS/TS; Go; Java; Ruby; PHP |
| AWS Bedrock | invoke_model, converse (boto3, aioboto3) | JS commands; Go; Ruby; Rust |
| Google Gemini / Vertex | generate_content, embed_content | JS/TS; Go genai |
| LiteLLM | completion, embedding (incl. aliases, `functools.partial`) | — |
| Frameworks | LangChain, LlamaIndex, Semantic Kernel, wrapper configs | Vercel AI SDK, LangChain4j, Spring AI, langchaingo, Semantic Kernel (.NET), RubyLLM, langchainrb, Prism, LLPhant, rig, genai |
| Raw HTTP to provider APIs | `/chat/completions`, `/v1/messages`, `/v1/responses`, `/embeddings`, `:generateContent` | same |
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
trainlab import FILE                                   # type detected
trainlab import {csv,aws,azure,gcp,langfuse,otel} FILE # force a type
trainlab import {openai,anthropic} [--days 30] [--key KEY | --file saved.json]
trainlab report [--out DIR] [--format html,json,md,pdf] [--fail-on high|medium|low]
trainlab run [PATH ...] [--usage FILE ...] [--exclude GLOB] [--format ...] [--fail-on ...]
trainlab ui [--port N] [--no-browser]
trainlab prices show MODEL | prices info | prices update
```

All state lives in `.trainlab/` (git-ignored automatically). Use `--workspace DIR` to
keep several analyses apart.

Prices ship with the package. `trainlab prices update` downloads the latest list to
`~/.cache/trainlab/prices.json` (or `$TRAINLAB_PRICES`). That and the usage API imports
are the only network calls the tool makes, each only when you run it.

## Benchmark

`scripts/eval_repos.py` scans real repositories and compares detected call sites with a
plain-text search for SDK call patterns. It was run on 33 open-source projects in October
2026 (shallow clones, 2 CPU cores). The set includes 9 typical AI apps, the 12 largest AI
codebases we could find, and 12 corner cases: cookbooks full of notebooks, SDK repos, and
Go, Java, Kotlin, C#, Ruby, PHP and Rust projects.

| Repo | Files | Scan time | Call sites |
|---|---|---|---|
| lobe-chat | 17,765 | 19.0s | 99 |
| dify | 12,957 | 11.6s | 15 |
| onyx | 7,595 | 5.9s | 23 |
| LibreChat | 6,156 | 8.4s | 9 |
| llama_index | 5,278 | 9.8s | 902 (752 in notebooks) |
| ragflow (Go) | 4,480 | 3.4s | 11 |
| semantic-kernel (C#, Python) | 4,060 | 4.4s | 650 |
| rig (Rust) | 2,911 | 4.4s | 106 |
| langchain | 2,762 | 5.1s | 52 |
| openai-cookbook (notebooks) | 1,237 | 5.3s | 544 |
| langchaingo (Go) | 718 | 0.6s | 100 |
| langchain4j-examples (Java) | 307 | 0.2s | 149 |
| …21 more | | | |

Every real SDK call found by the text search was also found by the scanner. The
remaining text matches were docstrings and markdown, server route definitions, a
framework's own internal helper methods, generated clients for the app's own API, and test
mocks.

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
