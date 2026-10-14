"""Heuristic (regex) scanning for JS/TS sources and config files.

Lower confidence than the Python AST scanner; sites are marked confidence="heuristic".
"""
from __future__ import annotations

import hashlib
import re

from ..schema import CallSite
from .python_ast import EXAMPLE_PATH_RE, MODEL_NAME_RE, TEST_PATH_RE

JS_EXTS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
CONFIG_EXTS = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".env"}

JS_CALLS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\.(chat\.completions\.(?:create|parse|stream))\s*\("), "chat", "openai"),
    (re.compile(r"\.(responses\.(?:create|parse|stream))\s*\("), "chat", "openai"),
    (re.compile(r"\.(messages\.batches\.create)\s*\("), "batch", "anthropic"),
    (re.compile(r"\.(messages\.(?:create|stream))\s*\("), "chat", "anthropic"),
    (re.compile(r"\.(embeddings\.create)\s*\("), "embedding", "openai"),
    (re.compile(r"\b(generateText|streamText|generateObject|streamObject)\s*\("), "chat", "vercel_ai"),
    (re.compile(r"\b(embed|embedMany)\s*\(\s*\{"), "embedding", "vercel_ai"),
    (re.compile(r"\.(models\.generateContent(?:Stream)?)\s*\("), "chat", "gemini"),
    (re.compile(r"\bnew\s+(Converse(?:Stream)?Command|InvokeModel(?:WithResponseStream)?Command)\s*\("),
     "chat", "bedrock"),
]
JS_MODEL = [
    re.compile(r"""\b(?:model|modelId)\s*:\s*['"`]([^'"`]+)['"`]"""),
    re.compile(r"""\b(openai|anthropic|google|azure|bedrock|mistral|groq)\s*\(\s*['"`]([^'"`]+)['"`]"""),
    re.compile(r"""\bmodel\s*:\s*(\w+)\s*\(\s*['"`]([^'"`]+)['"`]"""),
]
QUOTED = re.compile(r"""['"]([A-Za-z0-9][\w.\-:/]{2,79})['"]|(?:=|:)\s*([A-Za-z0-9][\w.\-:/]{2,79})\s*$""")


def _sid(path: str, line: int, kind: str) -> str:
    return hashlib.sha1(f"{path}:{line}:{kind}".encode()).hexdigest()[:12]


def scan_js(rel_path: str, text: str) -> list[CallSite]:
    sites: list[CallSite] = []
    lines = text.splitlines()
    is_test = bool(TEST_PATH_RE.search(rel_path)) or bool(re.search(r"\.(test|spec)\.[jt]sx?$", rel_path))
    for i, line in enumerate(lines):
        for pat, kind, provider in JS_CALLS:
            hit = pat.search(line)
            if not hit:
                continue
            window = "\n".join(lines[i: i + 25])
            model = None
            for mp in JS_MODEL:
                m = mp.search(window)
                if m:
                    groups = [g for g in m.groups() if g]
                    model = groups[-1]
                    if provider == "vercel_ai" and len(groups) == 2 and groups[0] in (
                            "openai", "anthropic", "google", "azure", "bedrock", "mistral", "groq"):
                        provider = {"google": "gemini"}.get(groups[0], groups[0])
                    break
            model_expr = None
            if not model:
                mx = re.search(r"\bmodel\s*:\s*([^,\n}]+)", window)
                model_expr = mx.group(1).strip()[:80] if mx else None
            if provider == "anthropic" and "anthropic" not in text.lower():
                provider = "openai_compatible"
            params = {}
            for p in ("max_tokens", "maxTokens", "max_completion_tokens", "temperature", "stream",
                      "response_format", "tools"):
                if re.search(rf"\b{p}\s*:", window):
                    params[p] = "<expr>"
            sites.append(CallSite(
                id=_sid(rel_path, i + 1, kind), path=rel_path, line=i + 1, kind=kind,
                provider=provider, sdk_call=hit.group(1),
                language="javascript", model=model, model_expr=model_expr,
                model_source="literal" if model else "unresolved",
                params=params, uses_cache_control="cache_control" in window,
                is_test=is_test, confidence="heuristic",
                scope="example" if EXAMPLE_PATH_RE.search(rel_path) else "app",
            ))
            break
    return sites


def scan_model_refs(rel_path: str, text: str) -> list[tuple[int, str]]:
    refs = []
    for i, line in enumerate(text.splitlines()):
        if len(line) > 400:
            continue
        for m in QUOTED.finditer(line):
            v = (m.group(1) or m.group(2) or "").strip()
            if v and MODEL_NAME_RE.match(v):
                refs.append((i + 1, v))
    return refs
