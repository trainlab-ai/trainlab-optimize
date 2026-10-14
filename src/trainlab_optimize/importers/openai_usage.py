"""OpenAI organization Usage API importer.

GET https://api.openai.com/v1/organization/usage/completions  (and /embeddings)
Auth: Authorization: Bearer <OpenAI admin key>.  Response: pages of daily buckets.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from ..schema import UsageRecord
from . import paginate

BASE = "https://api.openai.com/v1/organization/usage"


def _day(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def parse_pages(pages: list[dict], kind: str = "completions") -> list[UsageRecord]:
    out: list[UsageRecord] = []
    for page in pages:
        for bucket in page.get("data", []):
            day = _day(bucket.get("start_time", 0))
            for r in bucket.get("results", []):
                model = r.get("model")
                if not model:
                    continue
                if kind == "embeddings":
                    out.append(UsageRecord(
                        date=day, provider="openai", model=model,
                        input_tokens=int(r.get("input_tokens") or 0),
                        requests=int(r.get("num_model_requests") or 0),
                        project=r.get("project_id"), api_key=r.get("api_key_id"), source="openai_usage_api"))
                    continue
                out.append(UsageRecord(
                    date=day, provider="openai", model=model,
                    input_tokens=int(r.get("input_tokens") or 0),
                    output_tokens=int(r.get("output_tokens") or 0),
                    cached_input_tokens=int(r.get("input_cached_tokens") or 0),
                    cache_write_tokens=int(r.get("input_cache_write_tokens") or 0),
                    requests=int(r.get("num_model_requests") or 0),
                    batch=r.get("batch"),
                    project=r.get("project_id"), api_key=r.get("api_key_id"),
                    source="openai_usage_api"))
    return out


class OpenAIUsageImporter:
    name = "openai"

    def __init__(self, admin_key: str, base_url: str = BASE):
        self.admin_key = admin_key
        self.base_url = base_url.rstrip("/")

    def fetch(self, days: int = 30) -> list[UsageRecord]:
        start = int(time.time()) - days * 86400
        headers = {"Authorization": f"Bearer {self.admin_key}"}
        params = [("start_time", str(start)), ("bucket_width", "1d"), ("limit", "31"),
                  ("group_by", "model"), ("group_by", "project_id"), ("group_by", "api_key_id"),
                  ("group_by", "batch")]
        records = parse_pages(list(paginate(f"{self.base_url}/completions", params, headers)))
        emb_params = [p for p in params if p != ("group_by", "batch")]
        records += parse_pages(list(paginate(f"{self.base_url}/embeddings", emb_params, headers)),
                               kind="embeddings")
        return records
