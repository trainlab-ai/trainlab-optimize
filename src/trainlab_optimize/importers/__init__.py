"""Usage importer port. Each importer returns a list of UsageRecord."""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterator, Protocol

from ..schema import UsageRecord


class UsageImporter(Protocol):
    name: str

    def fetch(self, days: int = 30) -> list[UsageRecord]: ...


class ImportError_(RuntimeError):
    """Raised with a user-facing message when an import fails."""


def http_get_json(url: str, params: list[tuple[str, str]], headers: dict[str, str],
                  timeout: int = 60) -> dict:
    full = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(full, headers={"User-Agent": "trainlab-optimize/0.1", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read()[:300].decode(errors="replace")
        if e.code in (401, 403):
            raise ImportError_(
                f"{e.code} from {url}: the key is missing admin/usage permissions. "
                "Usage APIs need an organization admin key, not a regular API key.") from e
        raise ImportError_(f"HTTP {e.code} from {url}: {body}") from e
    except urllib.error.URLError as e:
        raise ImportError_(f"Could not reach {url}: {e.reason}") from e


def paginate(url: str, params: list[tuple[str, str]], headers: dict[str, str],
             max_pages: int = 50) -> Iterator[dict]:
    page = None
    for _ in range(max_pages):
        p = list(params) + ([("page", page)] if page else [])
        data = http_get_json(url, p, headers)
        yield data
        if not data.get("has_more"):
            return
        page = data.get("next_page")
        if not page:
            return
