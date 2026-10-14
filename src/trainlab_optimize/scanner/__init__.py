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
from .generic import CONFIG_EXTS, lang_of, scan_heuristic, scan_model_refs
from .python_ast import PythonFileScanner, module_constants, quiet_parse

EXCLUDE_DIRS = {".git", "node_modules", "venv", ".venv", "env", ".env", "site-packages", "dist",
                "build", "__pycache__", ".tox", ".mypy_cache", ".pytest_cache", ".trainlab",
                ".next", ".nuxt", "coverage", "vendor", ".idea", ".vscode", "target", ".ruff_cache"}
MAX_FILE_BYTES = 1_000_000
MAX_NOTEBOOK_BYTES = 25_000_000      # notebooks carry outputs (images); only code cells are read
SKIP_FILE_RE = re.compile(r"\.min\.(js|css)$|\.bundle\.js$|\.map$|(^|/)(package-lock\.json|yarn\.lock|"
                          r"pnpm-lock\.yaml|poetry\.lock|Cargo\.lock|composer\.lock|uv\.lock)$")

# Cheap text pre-filters: a file without any of these tokens cannot contain a call site or a
# model reference, so it is never parsed. Plain substring checks on lowercased text are much
# faster than a large case-insensitive regex. Keep in sync with the scanner signatures.
PY_TOKENS = ("openai", "anthropic", "litellm", "bedrock", "genai", "gemini", "completions",
             "embeddings", ".messages.", "responses.", "converse", "invoke_model", "generate_content",
             "langchain", "llama_index", "pydantic_ai", "init_chat_model", "gpt-", "chatgpt", "claude",
             "text-embedding", "o1", "o3", "o4", "mistral", "deepseek", "llama-", "llama3", "llama4",
             "v1/messages", "generatecontent", "googleapis.com", "dashscope", "qwen", "cohere", "groq",
             "together", "fireworks", "cerebras", "zhipu", "zai", "volcengine", "huggingface_hub", "ollama",
             "replicate", "voyageai", "xai_sdk", "grok", "generativemodel", "openrouter", "perplexity",
             "sambanova", "ai21", "moonshot", "kimi", "glm-", "minimax", "doubao", "qianfan", "hunyuan",
             "aisuite", "vertexai", ":11434", "aliyuncs", "api.cohere", "services/aigc")
OTHER_TOKENS = ("completions", "messages", "responses", "embeddings", "generatetext", "streamtext",
                "generateobject", "streamobject", "embedmany", "embed(", "generatecontent", "command(",
                "gpt-", "claude", "gemini", "text-embedding", "model:", "model :", "chatmodel",
                "completechat", "chatcompletion", "converse", "invokemodel", "langchain4j",
                "springframework.ai", "api.openai", "api.anthropic", "parameters:", "->chat()",
                "->create(", "prism::", "rubyllm", "langchain::llm", "async_openai", "rig::", "exec_chat",
                "modelid(", "openaichat", "anthropicchat", "dashscope", "openrouter", "mistralai", "cohere",
                "groq", "together", "fireworks", "ollama", "qwen", "deepseek", "grok", "moonshot", "kimi",
                "glm-", "minimax", "doubao", "perplexity", "chat.complete", "aliyuncs", "volces", "bigmodel",
                "x.ai", "cerebras", "huggingface", "replicate")


class _Trigger:
    def __init__(self, tokens):
        self.tokens = tokens

    def search(self, text: str) -> bool:
        low = text.lower()
        return any(t in low for t in self.tokens)


PY_TRIGGER = _Trigger(PY_TOKENS)
OTHER_TRIGGER = _Trigger(OTHER_TOKENS)


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
    repos: list[str] = field(default_factory=list)

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
            "repos": self.repos,
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
        r.repos = d.get("repos", [])
        return r

    def merge(self, other: "ScanResult") -> "ScanResult":
        """Combine scans of several repositories; every call site keeps its repo name."""
        def tag(res: "ScanResult"):
            name = Path(res.root).name
            for s in res.call_sites:
                if not s.repo:
                    s.repo = name
                    # ids are path:line based; make them unique across repositories
                    s.id = hashlib.sha1(f"{name}:{s.id}".encode()).hexdigest()[:12]
            for m in res.model_refs:
                m.setdefault("repo", name)
            return name
        names = (self.repos or [tag(self)]) + [tag(other)]
        out = ScanResult(root=str(Path(self.root).parent if not self.repos else self.root))
        out.call_sites = self.call_sites + [s for s in other.call_sites
                                            if (s.repo, s.path, s.line) not in
                                            {(x.repo, x.path, x.line) for x in self.call_sites}]
        out.model_refs = self.model_refs + other.model_refs
        out.files_scanned = self.files_scanned + other.files_scanned
        for d in (self.files_by_language, other.files_by_language):
            for k, v in d.items():
                out.files_by_language[k] = out.files_by_language.get(k, 0) + v
        out.parse_errors = self.parse_errors + other.parse_errors
        out.duration_s = self.duration_s + other.duration_s
        out.excluded = sorted(set(self.excluded) | set(other.excluded))
        out.repos = list(dict.fromkeys(names))
        return out


def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")
                             or d in (".github",))
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            try:
                size = p.stat().st_size
            except OSError:
                continue   # broken symlink, permission error
            limit = MAX_NOTEBOOK_BYTES if fn.endswith(".ipynb") else MAX_FILE_BYTES
            if size > limit or SKIP_FILE_RE.search(fn):
                continue
            yield p


PARALLEL_MIN_FILES = 400


def _scan_python_text(rel: str, text: str):
    if not PY_TRIGGER.search(text):
        return [], [], False, []   # no LLM-related token anywhere in the file: skip the AST entirely
    s = PythonFileScanner(rel, text)
    sites = s.scan()
    return sites, s.model_refs, s.parse_error, sorted(s.wanted) if sites else []


def _scan_python_batch(items: list[tuple[str, str]]):
    out = {}
    for rel, path in items:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        sites, refs, perr, wanted = _scan_python_text(rel, text)
        if sites or refs or perr:
            out[rel] = ([s.to_dict() for s in sites], refs, perr, wanted)
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
    results: dict = {rel: ([], [], False, []) for rel, _ in items}
    try:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            for part in ex.map(_scan_python_batch, batches):
                for rel, (sites, refs, perr, wanted) in part.items():
                    results[rel] = ([CallSite.from_dict(d) for d in sites], refs, perr, wanted)
    except (OSError, RuntimeError, cf.process.BrokenProcessPool):
        return {}   # fall back to sequential scanning
    return results


NON_PYTHON_KERNELS = {"r", "julia", "scala", "javascript", "typescript", "sql", "c++", "c", "java", "go",
                      "rust", "csharp", "c#", ".net-csharp", "fsharp", "kotlin", "bash", "sh", "powershell",
                      "matlab", "octave", "ruby", "sas", "stata", "wolfram language", "haskell"}


# Kernel / cell language -> heuristic scanner language
_NB_LANG = {"javascript": "javascript", "typescript": "javascript", "csharp": "csharp", "c#": "csharp",
            ".net-csharp": "csharp", "java": "jvm", "kotlin": "jvm", "scala": "jvm", "go": "go", "rust": "rust",
            "ruby": "ruby"}


def _scan_notebook_heuristic(rel: str, nb: dict, kernel_lang: str):
    """Non-Python notebooks (.NET Interactive / Polyglot, Deno, IJava, evcxr, ...): scan the code cells
    with the matching pattern scanner and report cell + line."""
    groups: dict[str, list[tuple[int, list[str]]]] = {}
    code_cell = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        code_cell += 1
        meta = cell.get("metadata") or {}
        cl = ((meta.get("dotnet_interactive") or {}).get("language")
              or (meta.get("polyglot_notebook") or {}).get("kernelName") or kernel_lang)
        hl = _NB_LANG.get(str(cl).lower())
        if not hl:
            continue
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else str(src)
        groups.setdefault(hl, []).append((code_cell, src.splitlines()))
    sites = []
    for hl, cells in groups.items():
        lines, origin = [], []
        for n, cell_lines in cells:
            for i, ln in enumerate(cell_lines, 1):
                lines.append(ln)
                origin.append((n, i))
        for s in scan_heuristic(rel, "\n".join(lines), hl):
            if 1 <= s.line <= len(origin):
                s.cell, s.line = origin[s.line - 1]
                s.scope = "example"
                sites.append(s)
    return sites, []


def _scan_notebook(rel: str, path: Path):
    """Scan the Python code cells of a Jupyter notebook as one program; report cell + line."""
    import json as _json
    try:
        nb = _json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return [], []
    lang = ((nb.get("metadata") or {}).get("kernelspec") or {}).get("language") or \
        ((nb.get("metadata") or {}).get("language_info") or {}).get("name") or "python"
    # Kernel metadata is often wrong (e.g. "coconut" for plain Python), and cells that are not
    # Python are dropped individually below, so only skip clearly non-Python kernels.
    if str(lang).lower() in NON_PYTHON_KERNELS:
        return _scan_notebook_heuristic(rel, nb, str(lang).lower())
    import ast as _ast
    # Cells can declare their own language (VS Code Polyglot / .NET Interactive) under a Python kernel
    other_sites, _ = _scan_notebook_heuristic(rel, nb, "python")
    cells: list[tuple[int, list[str]]] = []
    code_cell = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        code_cell += 1
        meta = cell.get("metadata") or {}
        cell_lang = ((meta.get("dotnet_interactive") or {}).get("language")
                     or (meta.get("polyglot_notebook") or {}).get("kernelName"))
        if cell_lang and str(cell_lang).lower() in _NB_LANG:
            continue
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else str(src)
        if src.lstrip().startswith("%%"):           # cell magics (%%bash, %%writefile, ...) are not Python
            continue
        cell_lines = []
        for ln in src.splitlines():
            stripped = ln.lstrip()
            # IPython line magics, shell escapes and help (`obj?`) are not Python
            if stripped.startswith(("%", "!", "?")) or (stripped.endswith("?") and not stripped.startswith("#")
                                                        and "(" not in stripped):
                ln = ""
            cell_lines.append(ln)
        # Keep only cells that parse on their own (one broken cell must not hide the notebook)
        try:
            quiet_parse("\n".join(cell_lines), rel)
        except (SyntaxError, ValueError):
            continue
        cells.append((code_cell, cell_lines))
    lines: list[str] = []
    origin: list[tuple[int, int]] = []          # combined line -> (cell number, line in cell)
    for n, cell_lines in cells:
        for j, ln in enumerate(cell_lines, 1):
            lines.append(ln)
            origin.append((n, j))
        lines.append("")
        origin.append((n, 0))
    text = "\n".join(lines)
    if not PY_TRIGGER.search(text):
        return other_sites, []
    scanner = PythonFileScanner(rel, text)
    sites = scanner.scan()
    for s in sites:
        cell, line = origin[s.line - 1] if 0 < s.line <= len(origin) else (None, s.line)
        s.cell, s.line, s.scope = cell, line, "example"
        s.id = hashlib.sha1(f"{rel}:{cell}:{line}:{s.kind}".encode()).hexdigest()[:12]
    refs = [(rel, origin[ln - 1][1] if 0 < ln <= len(origin) else ln, m) for ln, m in scanner.model_refs]
    return sites + other_sites, refs


def _module_index(root: Path, files: list[Path]) -> dict[str, list[str]]:
    """Dotted-module suffix -> files. libs/core/app/prompts.py is indexed as prompts,
    app.prompts, core.app.prompts, libs.core.app.prompts (monorepos, src/ layouts)."""
    index: dict[str, list[str]] = {}
    for p in files:
        if p.suffix != ".py":
            continue
        rel = p.relative_to(root).as_posix()
        parts = rel[:-3].split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        for i in range(len(parts)):
            index.setdefault(".".join(parts[i:]), []).append(rel)
    return index


def _resolve_cross_file(root: Path, files: list[Path], wanted_by_file: dict, result) -> None:
    """Second pass: rescan files whose prompts/models are imported from other project modules."""
    index = _module_index(root, files)
    consts_cache: dict[str, dict] = {}

    def lookup(qualified: str):
        module, _, name = qualified.rpartition(".")
        cands = index.get(module)
        if not cands:
            return None
        rel = min(cands, key=len)
        if rel not in consts_cache:
            try:
                consts_cache[rel] = module_constants((root / rel).read_text(encoding="utf-8", errors="replace"))
            except OSError:
                consts_cache[rel] = {}
        return consts_cache[rel].get(name)

    # Replace sites back-to-front so earlier offsets stay valid.
    for rel, (wanted, start, count) in sorted(wanted_by_file.items(), key=lambda kv: -kv[1][1]):
        external = {}
        for q in wanted:
            node = lookup(q)
            if node is not None:
                external[q] = node
        if not external:
            continue
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        scanner = PythonFileScanner(rel, text, external=external)
        result.call_sites[start:start + count] = scanner.scan()


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
    wanted_by_file: dict[str, tuple[list[str], int, int]] = {}
    py_results = _scan_python_parallel(root, files, exclude) if len(files) >= PARALLEL_MIN_FILES else {}

    for path in files:
        rel = path.relative_to(root).as_posix()
        if exclude and _excluded(rel, exclude):
            continue
        ext = path.suffix.lower()
        if ext == ".ipynb":
            result.files_scanned += 1
            result.files_by_language["notebook"] = result.files_by_language.get("notebook", 0) + 1
            sites, nrefs = _scan_notebook(rel, path)
            result.call_sites.extend(sites)
            refs.extend(nrefs)
            continue
        lang = ("python" if ext == ".py" else lang_of(ext) or
                ("config" if ext in CONFIG_EXTS or path.name.startswith(".env") else None))
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
                sites, mrefs, perr, wanted = py_results[rel]
            else:
                sites, mrefs, perr, wanted = _scan_python_text(rel, text)
            if perr:
                result.parse_errors.append(rel)
            if wanted:
                wanted_by_file[rel] = (wanted, len(result.call_sites), len(sites))
            result.call_sites.extend(sites)
            refs.extend((rel, ln, m) for ln, m in mrefs)
        elif lang in ("javascript", "go", "jvm", "csharp", "ruby", "php", "rust"):
            if not OTHER_TRIGGER.search(text):
                continue
            result.call_sites.extend(scan_heuristic(rel, text, lang))
            refs.extend((rel, ln, m) for ln, m in scan_model_refs(rel, text))
        else:
            found = scan_model_refs(rel, text)
            # JSON/JSONL that repeats model names many times is data (exports, datasets), not config.
            if ext == ".json" and (len(text) > 50_000 or len(found) > 20):
                continue
            refs.extend((rel, ln, m) for ln, m in found)

    if wanted_by_file:
        _resolve_cross_file(root, files, wanted_by_file, result)

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
