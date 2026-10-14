"""Local workspace (.trainlab/): scan results and imported usage. Nothing leaves the machine."""
from __future__ import annotations

import json
from pathlib import Path

from .scanner import ScanResult
from .schema import UsageRecord


class Workspace:
    def __init__(self, path: str | Path = ".trainlab"):
        self.path = Path(path)

    def ensure(self) -> "Workspace":
        self.path.mkdir(parents=True, exist_ok=True)
        gi = self.path / ".gitignore"
        if not gi.exists():
            gi.write_text("*\n")  # never commit usage data or scan output by accident
        return self

    # scan
    def save_scan(self, result: ScanResult) -> Path:
        self.ensure()
        p = self.path / "scan.json"
        p.write_text(json.dumps(result.to_dict(), indent=1))
        return p

    def load_scan(self) -> ScanResult | None:
        p = self.path / "scan.json"
        return ScanResult.from_dict(json.loads(p.read_text())) if p.exists() else None

    # usage
    def load_usage(self) -> list[UsageRecord]:
        p = self.path / "usage.json"
        if not p.exists():
            return []
        return [UsageRecord.from_dict(d) for d in json.loads(p.read_text())]

    def add_usage(self, records: list[UsageRecord], replace_source: bool = True) -> int:
        """Store records; re-importing the same source replaces its previous records."""
        self.ensure()
        existing = self.load_usage()
        if replace_source:
            sources = {r.source for r in records}
            existing = [r for r in existing if r.source not in sources]
        merged = existing + records
        (self.path / "usage.json").write_text(json.dumps([r.to_dict() for r in merged]))
        return len(records)

    def clear_usage(self) -> None:
        p = self.path / "usage.json"
        if p.exists():
            p.unlink()
