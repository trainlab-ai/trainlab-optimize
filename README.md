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

| | Python (AST, high confidence) | JS / TS (heuristic) |
|---|---|---|
| OpenAI / Azure OpenAI | chat, responses, embeddings, batches | chat, responses, embeddings |
| Anthropic | messages, batches | messages |
| AWS Bedrock | invoke_model, converse | — |
| Google Gemini (google-genai) | generate_content, embed_content | generateContent |
| LiteLLM | completion, embedding | — |
| LangChain / LlamaIndex | model configuration objects | — |
| Vercel AI SDK | — | generateText, streamText, generateObject |
| OpenAI-compatible (Groq, Together, vLLM…) | via `base_url` | — |

Usage import: OpenAI Usage API, Anthropic Usage & Cost API, generic CSV (Azure and
Bedrock cost exports work through CSV in v0.1).

Scope a scan to product code: `trainlab scan . --exclude examples --exclude docs`.
Tests are skipped unless `--include-tests` is given.

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
