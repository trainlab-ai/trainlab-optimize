"""Scanner port: walk a repository and return call sites + model references."""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..pricing import PriceCatalog
from ..schema import CallSite
from .generic import CONFIG_EXTS, JS_EXTS, scan_js, scan_model_refs
from .python_ast import PythonFileScanner

EXCLUDE_DIRS = {".git", "node_modules", "venv", ".venv", "env", ".env", "site-packages", "dist",
                "build", "__pycache__", ".tox", ".mypy_cache", ".pytest_cache", ".trainlab",
                ".next", ".nuxt", "coverage", "vendor", ".idea", ".vscode", "target", ".ruff_cache"}
MAX_FILE_BYTES = 1_000_000


@dataclass
class ScanResult:
    root: str
    call_sites: list[CallSite] = field(default_factory=list)
    model_refs: list[dict] = field(default_factory=list)   # {path, line, model, provider}
    files_scanned: int = 0
    files_by_language: dict[str, int] = field(default_factory=dict)
    parse_errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    excluded: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "call_sites": [s.to_dict() for s in self.call_sites],
            "model_refs": self.model_refs,
            "files_scanned": self.files_scanned,
            "files_by_language": self.files_by_language,
            "parse_errors": self.parse_errors,
            "duration_s": round(self.duration_s, 2),
            "excluded": self.excluded,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScanResult":
        r = cls(root=d["root"])
        r.call_sites = [CallSite.from_dict(s) for s in d.get("call_sites", [])]
        r.model_refs = d.get("model_refs", [])
        r.files_scanned = d.get("files_scanned", 0)
        r.files_by_language = d.get("files_by_language", {})
        r.parse_errors = d.get("parse_errors", [])
        r.duration_s = d.get("duration_s", 0.0)
        r.excluded = d.get("excluded", [])
        return r


def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")
                             or d in (".github",))
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield p


def _excluded(rel: str, patterns: list[str]) -> bool:
    import fnmatch
    for pat in patterns:
        p = pat.strip().rstrip("/")
        if not p:
            continue
        if fnmatch.fnmatch(rel, p) or rel.startswith(p + "/") or fnmatch.fnmatch(rel, p + "/*"):
            return True
    return False


def scan(root: str | Path, include_tests: bool = False, catalog: PriceCatalog | None = None,
         exclude: list[str] | None = None) -> ScanResult:
    t0 = time.time()
    root = Path(root).resolve()
    catalog = catalog or PriceCatalog.load()
    result = ScanResult(root=str(root))
    result.excluded = list(exclude or [])
    refs: list[tuple[str, int, str]] = []

    for path in iter_files(root):
        rel = path.relative_to(root).as_posix()
        if exclude and _excluded(rel, exclude):
            continue
        ext = path.suffix.lower()
        lang = ("python" if ext == ".py" else "javascript" if ext in JS_EXTS else
                "config" if ext in CONFIG_EXTS or path.name.startswith(".env") else None)
        if not lang:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        result.files_scanned += 1
        result.files_by_language[lang] = result.files_by_language.get(lang, 0) + 1

        if lang == "python":
            s = PythonFileScanner(rel, text)
            sites = s.scan()
            if not sites and not s.model_refs and "import" in text and _syntax_error(text):
                result.parse_errors.append(rel)
            result.call_sites.extend(sites)
            refs.extend((rel, ln, m) for ln, m in s.model_refs)
        elif lang == "javascript":
            result.call_sites.extend(scan_js(rel, text))
            refs.extend((rel, ln, m) for ln, m in scan_model_refs(rel, text))
        else:
            found = scan_model_refs(rel, text)
            # JSON/JSONL that repeats model names many times is data (exports, datasets), not config.
            if ext == ".json" and (len(text) > 50_000 or len(found) > 20):
                continue
            refs.extend((rel, ln, m) for ln, m in found)

    if not include_tests:
        result.call_sites = [s for s in result.call_sites if not s.is_test]

    # Keep only model references that resolve in the price catalog (cuts false positives).
    site_lines = {(s.path, s.line) for s in result.call_sites}
    site_models = {(s.path, (s.model or "").lower()) for s in result.call_sites}
    seen = set()
    for rel, ln, m in refs:
        if (rel, ln) in site_lines or (rel, m) in seen or (rel, m.lower()) in site_models:
            continue
        price = catalog.lookup(m)
        if not price:
            continue
        if not include_tests and _is_test_path(rel):
            continue
        seen.add((rel, m))
        result.model_refs.append({"path": rel, "line": ln, "model": m, "provider": price.provider})

    result.duration_s = time.time() - t0
    return result


def _is_test_path(rel: str) -> bool:
    from .python_ast import TEST_PATH_RE
    return bool(TEST_PATH_RE.search(rel))


def _syntax_error(text: str) -> bool:
    import ast
    try:
        ast.parse(text)
        return False
    except (SyntaxError, ValueError):
        return True


def repo_fingerprint(root: str | Path) -> str:
    return hashlib.sha1(str(Path(root).resolve()).encode()).hexdigest()[:10]
