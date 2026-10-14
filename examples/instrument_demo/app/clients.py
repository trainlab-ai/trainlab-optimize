"""Shared SDK clients. run_demo.py swaps in offline mock transports; a real app would not."""
from anthropic import Anthropic
from openai import OpenAI

openai_client: OpenAI | None = None
anthropic_client: Anthropic | None = None
