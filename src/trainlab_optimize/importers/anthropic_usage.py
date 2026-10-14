"""Anthropic Usage & Cost Admin API importer.

GET https://api.anthropic.com/v1/organizations/usage_report/messages
Headers: x-api-key: <Admin key sk-ant-admin01-...>, anthropic-version: 2023-06-01
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from ..schema import UsageRecord
from . import paginate

URL = "https://api.anthropic.com/v1/organizations/usage_report/messages"


def parse_pages(pages: list[dict]) -> list[UsageRecord]:
    out: list[UsageRecord] = []
    for page in pages:
        for bucket in page.get("data", []):
            day = (bucket.get("starting_at") or "")[:10]
            for r in bucket.get("results", []):
                model = r.get("model")
                if not model:
                    continue
                uncached = int(r.get("uncached_input_tokens") or 0)
                cache_read = int(r.get("cache_read_input_tokens") or 0)
                creation = r.get("cache_creation") or {}
                cache_write = int(creation.get("ephemeral_5m_input_tokens") or 0) + int(
                    creation.get("ephemeral_1h_input_tokens") or 0)
                out.append(UsageRecord(
                    date=day, provider="anthropic", model=model,
                    input_tokens=uncached + cache_read + cache_write,
                    output_tokens=int(r.get("output_tokens") or 0),
                    cached_input_tokens=cache_read, cache_write_tokens=cache_write,
                    requests=0,  # the usage report does not include request counts
                    batch=(r.get("service_tier") == "batch") if r.get("service_tier") else None,
                    project=r.get("workspace_id") or r.get("api_key_id"),
                    source="anthropic_usage_api"))
    return out


class AnthropicUsageImporter:
    name = "anthropic"

    def __init__(self, admin_key: str, url: str = URL):
        self.admin_key = admin_key
        self.url = url

    def fetch(self, days: int = 30) -> list[UsageRecord]:
        start = datetime.fromtimestamp(time.time() - days * 86400, tz=timezone.utc)
        headers = {"x-api-key": self.admin_key, "anthropic-version": "2023-06-01"}
        params = [("starting_at", start.strftime("%Y-%m-%dT00:00:00Z")), ("bucket_width", "1d"),
                  ("limit", "31"), ("group_by[]", "model"), ("group_by[]", "workspace_id"),
                  ("group_by[]", "service_tier")]
        return parse_pages(list(paginate(self.url, params, headers)))
