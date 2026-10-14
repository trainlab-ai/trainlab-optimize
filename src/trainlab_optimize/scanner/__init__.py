"""Scanner port: walk a repository and return call sites + model references."""
from __future__ import annotations

import hashlib
import os
import re
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

# Cheap text pre-filters: a file without any of these tokens cannot contain a call site or a
# model reference, so it is never parsed. Keep in sync with python_ast/generic signatures.
PY_TRIGGER = re.compile(
    r"openai|anthropic|litellm|bedrock|genai|gemini|completions|embeddings|\.messages\.|"
    r"responses\.|converse|invoke_model|generate_content|langchain|llama_index|pydantic_ai|"
    r"init_chat_model|gpt-|chatgpt|claude|text-embedding|\bo[1-9]\b|o[1-9]-|mistral|deepseek|llama-?\d")
JS_TRIGGER = re.compile(
    r"completions|messages\.|responses\.|embeddings|generateText|streamText|generateObject|"
    r"streamObject|embedMany|\bembed\(|generateContent|Command\(|gpt-|claude|gemini|"
    r"text-embedding|model\s*:", re.I)


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


PARALLEL_MIN_FILES = 400


def _scan_python_text(rel: str, text: str):
    if not PY_TRIGGER.search(text):
        return [], [], False   # no LLM-related token anywhere in the file: skip the AST entirely
    s = PythonFileScanner(rel, text)
    sites = s.scan()
    return sites, s.model_refs, s.parse_error


def _scan_python_batch(items: list[tuple[str, str]]):
    out = {}
    for rel, path in items:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        sites, refs, perr = _scan_python_text(rel, text)
        if sites or refs or perr:
            out[rel] = ([s.to_dict() for s in sites], refs, perr)
    return out


def _scan_python_parallel(root: Path, files: list[Path], exclude) -> dict:
    """Parse Python files across CPU cores. Returns {rel: (sites, refs, parse_error)}; files that
    produced nothing are reported as empty results so they are not parsed again."""
    import concurrent.futures as cf
    items = []
    for p in files:
        if p.suffix.lower() != ".py":
            continue
        rel = p.relative_to(root).as_posix()
        if exclude and _excluded(rel, exclude):
            continue
        items.append((rel, str(p)))
    workers = max(1, min(8, (os.cpu_count() or 2)))
    if workers == 1 or len(items) < PARALLEL_MIN_FILES:
        return {}
    chunk = max(20, len(items) // (workers * 4))
    batches = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    results: dict = {rel: ([], [], False) for rel, _ in items}
    try:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            for part in ex.map(_scan_python_batch, batches):
                for rel, (sites, refs, perr) in part.items():
                    results[rel] = ([CallSite.from_dict(d) for d in sites], refs, perr)
    except (OSError, RuntimeError, cf.process.BrokenProcessPool):
        return {}   # fall back to sequential scanning
    return results


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

    files = list(iter_files(root))
    py_results = _scan_python_parallel(root, files, exclude) if len(files) >= PARALLEL_MIN_FILES else {}

    for path in files:
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
            if rel in py_results:
                sites, mrefs, perr = py_results[rel]
            else:
                sites, mrefs, perr = _scan_python_text(rel, text)
            if perr:
                result.parse_errors.append(rel)
            result.call_sites.extend(sites)
            refs.extend((rel, ln, m) for ln, m in mrefs)
        elif lang == "javascript":
            if not JS_TRIGGER.search(text):
                continue
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


def repo_fingerprint(root: str | Path) -> str:
    return hashlib.sha1(str(Path(root).resolve()).encode()).hexdigest()[:10]
