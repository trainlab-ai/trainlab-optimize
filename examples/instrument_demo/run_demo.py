"""Run a small app under trainlab_optimize.instrument(), offline.

    python examples/instrument_demo/run_demo.py
    cd examples/instrument_demo
    trainlab scan app && trainlab import .trainlab/traces && trainlab report --format html,md
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import trainlab_optimize as trainlab  # noqa: E402

trainlab.instrument(project_root=HERE, service="acme-api", tags={"env": "demo"})

import mock_llm  # noqa: E402
from app import clients, invoices, search, support  # noqa: E402

clients.openai_client, clients.anthropic_client = mock_llm.clients()

TICKETS = ["I was charged twice", "App crashes on login", "How do I change my email?", "Refund please"]

with trainlab.tag(feature="support-bot", team="cx"):
    for i in range(120):
        t = TICKETS[i % len(TICKETS)]
        support.classify(t)
        if i % 3 == 0:
            support.answer(t)

with trainlab.tag(feature="invoice-extract", team="finance"):
    for i in range(40):
        invoices.extract(f"INVOICE #{i}\nAcme Ltd\nTotal: 12.50")

with trainlab.tag(feature="search", team="platform"):
    for i in range(60):
        search.embed([f"chunk {i}-{j}" for j in range(8)])
        if i % 4 == 0:
            search.summarise(["doc a", "doc b"])

trainlab.flush()
print(trainlab.status())
