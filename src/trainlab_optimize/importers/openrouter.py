"""OpenRouter activity importer.

GET https://openrouter.ai/api/v1/keys                       list API keys (hash, name, label)
GET https://openrouter.ai/api/v1/activity?api_key_hash=...   last 30 days, per day x model x provider
Auth: Authorization: Bearer <management key>. Regular inference keys get 403 on both.

Spend per API key comes from filtering activity by each key's hash. If keys can't be listed,
the account-wide activity is imported without a key column.
"""
from __future__ import annotations

from ..schema import UsageRecord
from . import ImportError_, http_get_json

BASE = "https://openrouter.ai/api/v1"


def parse_activity(pages: list[dict] | dict, api_key: str | None = None,
                   source: str = "openrouter_api") -> list[UsageRecord]:
    """Activity response(s) -> records. Accepts one response, a list of them, or a bare item list."""
    if isinstance(pages, dict):
        pages = [pages]
    out: list[UsageRecord] = []
    for page in pages:
        items = page.get("data", []) if isinstance(page, dict) else page if isinstance(page, list) else []
        for it in items:
            if not isinstance(it, dict) or not it.get("model"):
                continue
            credits = float(it.get("usage") or 0)
            byok = float(it.get("byok_usage_inference") or 0)
            out.append(UsageRecord(
                date=str(it.get("date") or "unknown")[:10], provider="openrouter", model=it["model"],
                input_tokens=int(it.get("prompt_tokens") or 0),
                output_tokens=int(it.get("completion_tokens") or 0),
                cached_input_tokens=int(it.get("cached_tokens") or 0),
                requests=int(it.get("requests") or 0),
                # OpenRouter credits + inference billed on the customer's own provider keys (BYOK)
                cost_usd=credits + byok,
                project=it.get("workspace_id"),
                api_key=api_key or it.get("api_key"),
                source=source))
    return out


class OpenRouterImporter:
    name = "openrouter"

    def __init__(self, management_key: str, base_url: str = BASE):
        self.headers = {"Authorization": f"Bearer {management_key}"}
        self.base = base_url.rstrip("/")

    def list_keys(self) -> list[dict]:
        keys: list[dict] = []
        for _ in range(50):
            page = http_get_json(f"{self.base}/keys", [("include_disabled", "true"),
                                                       ("offset", str(len(keys)))], self.headers)
            batch = page.get("data", [])
            keys += batch
            if not batch:
                break
        return keys

    def fetch(self, days: int = 30) -> list[UsageRecord]:
        # The activity API covers the last 30 days; `days` is accepted for interface symmetry.
        try:
            keys = self.list_keys()
        except ImportError_:
            keys = []
        if not keys:
            return parse_activity(http_get_json(f"{self.base}/activity", [], self.headers))
        records: list[UsageRecord] = []
        for k in keys:
            label = k.get("name") or k.get("label") or (k.get("hash") or "")[:10]
            if not k.get("hash"):
                continue
            page = http_get_json(f"{self.base}/activity", [("api_key_hash", k["hash"])], self.headers)
            records += parse_activity(page, api_key=label)
        return records
