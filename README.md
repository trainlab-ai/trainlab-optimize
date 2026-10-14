# TrainLab Optimize

Find out where your LLM money goes, line by line, and what to change.

`trainlab` scans your repository for LLM calls, imports your actual usage from OpenAI,
Anthropic or any billing CSV, links the spend to the code that causes it, and writes a
report of concrete changes with estimated monthly savings.

- **Runs locally.** Your code and usage data never leave your machine. The report is a
  single HTML file with no external requests.
- **Zero dependencies.** Pure Python standard library (3.10+).
- **Read-only.** It never executes your code and never calls an LLM.

## Quick start

```bash
pip install -e .                      # PyPI release coming with v0.1

cd your-repo
trainlab scan .                       # find LLM call sites
trainlab import openai                # needs OPENAI_ADMIN_KEY (read-only usage access)
trainlab import anthropic             # needs ANTHROPIC_ADMIN_KEY
trainlab import csv billing.csv       # or any export with model + tokens or cost columns
trainlab report                       # writes .trainlab/report/trainlab-report.html
```

Without usage data the report still lists every call site and code-level issues; it just
can't put a price on them.

One-step version: `trainlab run . --usage-csv billing.csv`

Try it on the bundled demo (a small app with deliberate waste):

```bash
cd examples/demo_app
trainlab scan . && trainlab import csv ../demo_usage/billing_export.csv && trainlab report
```

## What it finds

| Rule | Finding | Typical fix |
|---|---|---|
| R101 | Premium model on a simple task (extraction, classification, structured output) | Test an economy model; fine-tuned small model for high volume |
| R102 | Model at or near its provider deprecation date | Migrate before shutdown |
| R103 | Large system prompt that is never cached (dynamic content first, or no `cache_control`) | Reorder prompt, add cache breakpoint |
| R104 | Sequential LLM calls in a loop | Batch API (~50% cheaper) if latency-tolerant |
| R105 | No output token limit | Set `max_completion_tokens` |
| R106 | Unbounded retries around an LLM call | Add a stop condition |
| R9xx | Coverage gaps: runtime-selected models, spend with no matching code | Instrument, or scan the other repos |

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

**Attribution (v0.1):** spend is linked to code by model name. If one call site uses a
model, it gets all of that model's spend; if several do, it is split evenly and marked
"shared". Runtime instrumentation (v0.2) replaces this with exact per-line attribution.

## Supported

| | Python (AST, high confidence) | JS/TS, Go, Java/Kotlin, C# (heuristic) |
|---|---|---|
| OpenAI / Azure OpenAI | chat, responses, embeddings, batches | JS SDK; openai-go, go-openai; openai-java; OpenAI .NET, Azure.AI.OpenAI, Azure.AI.Inference |
| Anthropic | messages, batches | JS SDK; anthropic-sdk-go; anthropic-java |
| AWS Bedrock | invoke_model, converse (boto3, aioboto3) | JS `ConverseCommand`; Go bedrockruntime |
| Google Gemini | generate_content, embed_content | JS `generateContent`; Go genai |
| LiteLLM | completion, embedding (incl. aliases, `functools.partial`) | — |
| LangChain / LlamaIndex / wrappers | model configuration objects | LangChain4j builders, Spring AI ChatClient, langchaingo |
| Vercel AI SDK | — | generateText, streamText, generateObject |
| Raw HTTP to provider APIs | `/chat/completions`, `/v1/messages`, `/v1/responses`, `/embeddings`, `:generateContent` | same |
| Injected or factory-built clients | `client.chat.completions.create(...)` without an SDK import | — |

Server routes that *serve* an OpenAI-compatible API (`@app.post("/v1/chat/completions")`,
`/api/chat/completions`) are not counted as calls.

Usage import: OpenAI Usage API, Anthropic Usage & Cost API, generic CSV (Azure and
Bedrock cost exports work through CSV in v0.1).

Scope a scan to product code: `trainlab scan . --exclude examples --exclude docs`.
Tests are skipped unless `--include-tests` is given.

## Benchmark

`scripts/eval_repos.py` scans real repositories and compares detected call sites with a
plain-text search for SDK call patterns. Results on 21 open-source AI projects
(October 2026, shallow clones, 2 CPU cores):

| Repo | Files | Scan time | Call sites |
|---|---|---|---|
| lobe-chat | 17,763 | 16.7s | 99 |
| dify | 12,958 | 9.7s | 15 |
| onyx | 7,587 | 5.5s | 24 |
| LibreChat | 6,157 | 7.8s | 9 |
| llama_index | 4,524 | 6.6s | 150 |
| ragflow (Go) | 4,481 | 3.0s | 11 |
| langchain | 2,762 | 5.3s | 52 |
| crewAI | 2,258 | 5.9s | 47 |
| pipecat | 1,691 | 2.3s | 70 |
| …12 more | | | |

Every SDK call found by the text search was also found by the scanner; the remaining
text matches were docstrings, server route definitions, type names, generated clients for
the app's own API, and test mocks. Peak memory stayed under 60 MB.

## Commands

```
trainlab scan [PATH] [--exclude GLOB] [--include-tests]
trainlab import {openai,anthropic} [--days 30] [--key KEY | --file saved_response.json]
trainlab import csv FILE [--provider NAME]
trainlab report [--out DIR] [--format html,json]
trainlab run [PATH] [--usage-csv FILE] [--exclude GLOB]
trainlab prices show MODEL | prices info
```

All state lives in `.trainlab/` (git-ignored automatically). Use `--workspace DIR` to
keep several analyses apart.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
python scripts/build_catalog.py        # refresh prices from LiteLLM's price map
python scripts/make_demo_usage.py      # regenerate demo usage fixtures
```

Price data comes from [BerriAI/litellm](https://github.com/BerriAI/litellm)'s
`model_prices_and_context_window.json` (MIT); attribution and build date are stored in
`src/trainlab_optimize/catalog/prices.json`.

## Roadmap

- **v0.2** `trainlab.instrument()`: exact code-to-cost mapping from live traces; Langfuse/OTel import; PDF report
- **v0.3** `trainlab experiment`: replay real inputs against cheaper candidates with quality gates; `trainlab export`

## License

Apache 2.0
