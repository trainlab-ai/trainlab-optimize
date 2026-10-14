# Benchmark test cases

Run per repository by `bench/run.py`. Each test ends in **pass**, **fail**, **review** (a
threshold missed that needs a human to classify the misses) or **skip** (doesn't apply).

| ID | Test | Pass condition | Applies to |
|---|---|---|---|
| T01 | Scan completes | The scanner returns without an exception | all |
| T02 | Scan speed | ≤ 60 s, or ≥ 150 files/s on big repos | all |
| T03 | Deterministic | A second scan gives identical call sites (id, path, line, kind, provider, model) | scans ≤ 40 s |
| T04 | Recall vs text search | ≥ 95% of baseline lines found (else **review**: misses are classified by hand as real misses or baseline noise) | LLM repos with baseline matches |
| T05 | No false positives | Zero LLM call sites reported | negative controls |
| T06 | Provider consistency | No call site has provider `unknown` while its model's maker is known; share of calls naming a specific provider is reported | repos with call sites |
| T07 | Mock usage import | Imported spend equals the generated bill (±0.5%) | all |
| T08 | Re-import is idempotent | Importing the same file again doesn't change spend | all |
| T09 | Attribution | Every billed model that the code calls is fully attributed to call sites; the decoy model (billed, never called) is reported as spend with no matching code | all |
| T10 | Spend per API key | Keys and per-key totals equal the generated bill (±0.5%) | all |
| T11 | Pricing from tokens | A tokens-only file is priced exactly as the raw catalog numbers say (±0.5%) | all |
| T12 | Reports | HTML, JSON and Markdown are written; JSON has all top-level sections; HTML loads nothing external | all |
| T13 | Savings sanity | 0 ≤ low ≤ high ≤ monthly spend | all |
| T14 | Finding locations | ≥ 95% of cited lines (±2) contain the call or model they claim | repos with findings |
| T99 | No crash after scan | Import, analysis and reporting run without an exception | all |

## Mock usage

`bench/mock_usage.py` bills the up-to-five most used models found in the repo's own code
(or the provider's usual model when none can be resolved), plus one **decoy** model the code
never calls, for 30 days across two API keys (`key-prod` 70%, `key-batch` 30%). One of
three CSV layouts is used per repo, chosen from the repo name, so each import path is
exercised about 33 times:

- **generic**: one row per day × model × key, with token and cost columns
- **openrouter**: OpenRouter's activity export, per-request rows with `vendor/model` slugs
- **unfamiliar**: cloud-bill columns trainlab doesn't know (`Billing Item`, `Pretax Amount (USD)`, ...),
  with bill lines like `gpt-4o input_tokens`, imported with an explicit column map

Expected totals are computed from the generated rows, not by trainlab.

## Recall baseline

A regular expression over source lines (tests excluded) that matches common SDK call
shapes and provider endpoint strings across Python, JS/TS, Go, Java/Kotlin, C#, Ruby, PHP
and Rust. It over-matches on purpose: docstrings, type names, server routes and other
APIs with the same method names show up, which is why T04 below 95% is a **review**, not a
failure. Each review is triaged in `bench/triage.json`.
