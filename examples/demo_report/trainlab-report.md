## LLM cost report: demo_app

**Monthly LLM spend:** $23,501 (30 days of data)  
_Overlap: csv:billing_export.csv overlaps anthropic_usage_file:anthropic_usage.json, openai_usage_file:openai_usage_completions.json, openai_usage_file:openai_usage_embeddings.json for the same models and days; counted once._  
**Potential saving:** $5,629 – $12,412/month (estimated, not yet tested)  
**Spend traced to code:** 97%

7 LLM call sites in 7 files, 9 findings.

| | Finding | Where | Saving/month |
|---|---|---|---|
| 🔴 high | Premium model `gpt-4o` on a simple task (extraction, structured_output) | `services/invoice_extractor.py:19`, `services/invoice_extractor.py:34` | $3,747–$7,494 |
| 🔴 high | ~1,902-token system prompt is not being cached | `services/support_rag.py:33` | $1,742–$3,484 |
| 🔴 high | `gpt-4` is near its deprecation date (2026-10-23) | `services/legacy_summary.py:10` |  |
| 🟠 medium | Sequential calls in a loop in `process_batch` | `services/invoice_extractor.py:34` | $0–$1,993 |
| 🟠 medium | Sequential calls in a loop in `summarize_notes` | `services/legacy_summary.py:10` | $0–$1,154 |
| 🟠 medium | Premium model `gpt-4.1` on a simple task (classification) | `services/ticket_classifier.py:19` | $140–$280 |
| 🟠 medium | `claude-sonnet-4-5` is near its deprecation date (2026-11-30) | `services/support_rag.py:33` |  |
| 🟠 medium | Unbounded retries around an LLM call in `extract_invoice` | `services/invoice_extractor.py:19` |  |
| ⚪ low | No output token limit | `services/invoice_extractor.py:19`, `services/invoice_extractor.py:34` |  |

**Gaps:** $761/month of spend has no matching call site

<sub>trainlab-optimize 0.1.3, generated locally. No code or usage data left the machine.</sub>
