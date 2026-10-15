# Northwind support desk

A realistic OpenAI app to try trainlab on. It processes a day of customer support tickets for
Northwind Cloud (a fictional file-sync company) and uses the API the way production apps do:

| Feature | File | Model | How it calls |
|---|---|---|---|
| Route each ticket to a queue | `support_desk/triage.py` | gpt-4o | 8 threads |
| Pull CRM fields out as JSON | `support_desk/extract.py` | gpt-4o | JSON mode, 8 threads |
| Score customer mood | `support_desk/sentiment.py` | gpt-4o-mini | `AsyncOpenAI`, `asyncio.gather` |
| Search the help centre | `support_desk/kb.py` | text-embedding-3-small | batched embeddings |
| Draft a grounded reply | `support_desk/reply.py` | gpt-4o-mini | streaming with usage |
| Answer shipping questions | `support_desk/agent.py` | gpt-4o-mini | tool calling, multi-turn |
| Write the daily digest | `support_desk/digest.py` | gpt-4.1 | Responses API |

The app doesn't depend on trainlab. If trainlab is installed, `trainlab.tag()` labels calls
by feature and team; otherwise the tags do nothing.

## Run it

Install once (from the repository root):

```bash
pip install -e .
pip install openai
cd examples/support_desk
```

**Offline and free** (a local mock of the OpenAI API answers):

```bash
trainlab trace --capture -- python run.py --mock
```

**With the real OpenAI API** (about $0.30 for the default 100 tickets):

```bash
export OPENAI_API_KEY=sk-...            # Windows PowerShell:  $env:OPENAI_API_KEY="sk-..."
trainlab trace --capture --fresh -- python run.py         # --fresh also drops old samples
```

`--fresh` drops traces from earlier runs, so mock and real calls don't mix.

You get a report of the run in `.trainlab/report/trainlab-report.html`, with cost per line, per
feature and per team. Expect two findings: gpt-4o is used for routing and for field extraction,
both simple tasks.

## Prove the saving

```bash
trainlab experiment run --max-cost 1 --judge gpt-4.1 --open
```

This replays the captured tickets (redacted: emails and phone numbers are replaced) against
cheaper models with your key, scores every answer against what gpt-4o said, and updates the
report. Costs a few cents. What to expect:

- **Routing and extraction:** tested against gpt-4.1-mini, gpt-5-mini and gpt-4o-mini. A model
  that matches on enough tickets becomes a *validated* saving. In extraction, `order_id`,
  `product` and `wants_refund` must match; `summary` is prose (with `--judge gpt-4.1` a
  reworded summary is judged on meaning; the judge also lets the free-text reply drafts be
  tested, so expect about $0.30 in total instead of a few cents); `urgency` is a judgment call, so its
  agreement with gpt-4o is shown but doesn't decide the verdict.
- **Reply drafts and the digest:** skipped, because they are free text and need a judge. Add
  `--judge gpt-4.1` to score them (costs more).
- **The shipping agent:** not captured, because tool-calling turns can't be replayed from text
  alone.
- **Mood scoring:** already on gpt-4o-mini; the only cheaper candidate is a reasoning model whose
  hidden reasoning tokens can make it cost more per call. If so, it's reported and not
  recommended.

With 100 tickets, 97 are unique after redaction. To pass, a model has to clear a 95%
lower confidence bound: 97 out of 97 or 96 out of 97 passes, while two misses leave the result
*inconclusive*, and the output says how many more samples would settle it
(`python run.py --tickets 300` captures more).

## Read the examples before trusting a FAIL

An experiment compares cheaper models with what the current model said, not with the truth. When
the prompt leaves a field open to interpretation, they disagree because the question is vague,
not because one model is worse. `trainlab experiment show <id> --examples` shows each
disagreement: the ticket, gpt-4o's answer and the candidate's.

The app keeps its first extraction prompt to show this: `python run.py --prompt v1`. It asks for
a "product" without saying what one is. On the real API, gpt-4o answered "SSO", "Starter" and
"2-port hub" as products; the cheaper models picked other words from the ticket, and every one of
them failed. The default prompt (`v2`) defines each field: one of Northwind's five products,
"refund" means asking for money back, rules for urgency, a summary that names the customer.

```bash
trainlab trace --capture --fresh -- python run.py --prompt v1
trainlab experiment run --max-cost 1 --judge gpt-4.1
trainlab experiment show <extraction experiment id> --examples      # product: "SSO" vs "Starter"...
trainlab trace --capture --fresh -- python run.py                   # v2, the fields defined
trainlab experiment run --max-cost 1 --judge gpt-4.1
```

Tightening the prompt is usually the first saving: the cheaper model can only match what the
current one was asked clearly.

## Run the app on its own

```bash
python run.py --mock                    # or without --mock for the real API
python run.py --tickets 20 --seed 3     # fewer, different tickets
python mock_openai.py --port 8787       # the mock as a standalone server
```
