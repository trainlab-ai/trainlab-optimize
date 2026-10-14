"""Heuristic (regex) scanning for JS/TS, Go, Java/Kotlin, C# sources and config files.

Lower confidence than the Python AST scanner; sites are marked confidence="heuristic".
"""
from __future__ import annotations

import hashlib
import re

from ..schema import CallSite
from .python_ast import EXAMPLE_PATH_RE, MODEL_NAME_RE, TEST_PATH_RE, http_endpoint

JS_EXTS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
GO_EXTS = {".go"}
JVM_EXTS = {".java", ".kt", ".kts"}
CS_EXTS = {".cs"}
CONFIG_EXTS = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".env"}

# (pattern, kind, provider, required token in file or None)
LANG_CALLS: dict[str, list[tuple[re.Pattern, str, str, str | None]]] = {
    "javascript": [
        (re.compile(r"\.(chat\.completions\.(?:create|parse|stream))\s*\("), "chat", "openai", None),
        (re.compile(r"\.(responses\.(?:create|parse|stream))\s*\("), "chat", "openai", None),
        (re.compile(r"\.(messages\.batches\.create)\s*\("), "batch", "anthropic", None),
        (re.compile(r"\.(messages\.(?:create|stream))\s*\("), "chat", "anthropic", None),
        (re.compile(r"\.(embeddings\.create)\s*\("), "embedding", "openai", None),
        (re.compile(r"\b(generateText|streamText|generateObject|streamObject)\s*\("), "chat", "vercel_ai", None),
        (re.compile(r"\b(embed|embedMany)\s*\(\s*\{"), "embedding", "vercel_ai", None),
        (re.compile(r"\.(models\.generateContent(?:Stream)?)\s*\("), "chat", "gemini", None),
        (re.compile(r"\bnew\s+(Converse(?:Stream)?Command|InvokeModel(?:WithResponseStream)?Command)\s*\("),
         "chat", "bedrock", None),
    ],
    "go": [
        (re.compile(r"\.(Chat\.Completions\.New(?:Streaming)?)\("), "chat", "openai", "openai-go"),
        (re.compile(r"\.(Responses\.New(?:Streaming)?)\("), "chat", "openai", "openai-go"),
        (re.compile(r"\.(Embeddings\.New)\("), "embedding", "openai", "openai-go"),
        (re.compile(r"\.(Messages\.New(?:Streaming)?)\("), "chat", "anthropic", "anthropic-sdk-go"),
        (re.compile(r"\.(CreateChatCompletion(?:Stream)?)\("), "chat", "openai", "go-openai"),
        (re.compile(r"\.(CreateEmbeddings)\("), "embedding", "openai", "go-openai"),
        (re.compile(r"\.(Converse(?:Stream)?)\(\s*ctx"), "chat", "bedrock", "bedrockruntime"),
        (re.compile(r"\.(InvokeModel(?:WithResponseStream)?)\(\s*ctx"), "chat", "bedrock", "bedrockruntime"),
        (re.compile(r"\.(GenerateContent(?:Stream)?)\(\s*ctx"), "chat", "gemini", "genai"),
        (re.compile(r"\b(llms\.GenerateFromSinglePrompt)\("), "chat", "unknown", "langchaingo"),
    ],
    "jvm": [
        (re.compile(r"\.(chat\(\)\s*\.completions\(\)\s*\.create)\("), "chat", "openai", "com.openai"),
        (re.compile(r"\.(responses\(\)\s*\.create)\("), "chat", "openai", "com.openai"),
        (re.compile(r"\.(embeddings\(\)\s*\.create)\("), "embedding", "openai", "com.openai"),
        (re.compile(r"\.(messages\(\)\s*\.create)\("), "chat", "anthropic", "com.anthropic"),
        (re.compile(r"\.(getChatCompletions(?:Stream)?)\("), "chat", "azure", "com.azure.ai.openai"),
        (re.compile(r"\b((?:OpenAi|AzureOpenAi|Anthropic|BedrockAnthropic|GoogleAiGemini|VertexAiGemini|Ollama)"
                    r"(?:Chat|Streaming|Embedding)Model\.builder)\(\)"), "model_config", "unknown", "langchain4j"),
        (re.compile(r"\b(chatClient\.prompt)\("), "chat", "unknown", "springframework.ai"),
    ],
    "csharp": [
        (re.compile(r"\.(CompleteChat(?:Streaming)?(?:Async)?)\("), "chat", "openai", "OpenAI"),
        (re.compile(r"\.(GetChatCompletions(?:Streaming)?(?:Async)?)\("), "chat", "azure", "Azure.AI.OpenAI"),
        (re.compile(r"\.(GenerateEmbeddings?(?:Async)?)\("), "embedding", "openai", "OpenAI"),
        (re.compile(r"\.(Complete(?:Streaming)?Async)\("), "chat", "azure", "Azure.AI.Inference"),
    ],
}
MODEL_PATTERNS = [
    re.compile(r"""\b(?:model|modelId|Model|ModelID|modelName)\s*[:=]\s*['"`]([^'"`]+)['"`]"""),
    re.compile(r"""\.(?:model|modelName|modelId)\(\s*"([^"]+)"\s*\)"""),
    re.compile(r"""\b(openai|anthropic|google|azure|bedrock|mistral|groq)\s*\(\s*['"`]([^'"`]+)['"`]"""),
    re.compile(r"""\b(?:ChatModel|Model|EmbeddingModel)\.([A-Z][A-Z0-9_]{2,})\b"""),
]
MODEL_EXPR = re.compile(r"""\b(?:model|modelId|Model|modelName)\s*[:=]\s*([^,\n}\)]+)""")
STRING = re.compile(r"""(["'`])((?:(?!\1).){3,200}?)\1""")
ROUTE_DEF = re.compile(r"""\.(?:GET|POST|PUT|Handle|HandleFunc|Any|Group|get|post|put|route|all)\s*\(\s*["'`]|"""
                       r"""@(?:Post|Get|Request)Mapping|\[Http(?:Post|Get)|Map(?:Post|Get)\(|router\.|"""
                       r"""Request\.Path|PathString|\.path\s*===?|\.url\s*===?|pathname\s*===?""")
COMMENT = re.compile(r"^\s*(?://|#|\*|/\*|--)")
QUOTED = re.compile(r"""['"]([A-Za-z0-9][\w.\-:/]{1,79})['"]|(?:=|:)\s*([A-Za-z0-9][\w.\-:/]{1,79})\s*$""")
OTHER_TEST = re.compile(r"\.(test|spec)\.[jt]sx?$|_test\.go$|(^|/)src/test/|(^|/)[^/]*\.Tests?/|Tests?\.cs$")


def lang_of(ext: str) -> str | None:
    if ext in JS_EXTS:
        return "javascript"
    if ext in GO_EXTS:
        return "go"
    if ext in JVM_EXTS:
        return "jvm"
    if ext in CS_EXTS:
        return "csharp"
    return None


def _sid(path: str, line: int, kind: str) -> str:
    return hashlib.sha1(f"{path}:{line}:{kind}".encode()).hexdigest()[:12]


def _enum_to_model(name: str) -> str:
    # openai-java ChatModel.GPT_4O -> gpt-4o; Model.CLAUDE_SONNET_4_5 -> claude-sonnet-4-5
    return name.lower().replace("_", "-")


def _find_model(window: str, provider: str) -> tuple[str | None, str]:
    for idx, mp in enumerate(MODEL_PATTERNS):
        m = mp.search(window)
        if not m:
            continue
        groups = [g for g in m.groups() if g]
        if idx == 3:
            return _enum_to_model(groups[-1]), provider
        if idx == 2 and len(groups) == 2 and provider in ("vercel_ai", "unknown"):
            provider = {"google": "gemini"}.get(groups[0], groups[0])
        return groups[-1], provider
    return None, provider


def _line_index(text: str):
    starts = [0]
    for m in re.finditer("\n", text):
        starts.append(m.end())
    return starts


ENDPOINT_HINT = re.compile(r"chat/completions|/messages|/responses|/embeddings|generatecontent", re.I)


def scan_heuristic(rel_path: str, text: str, lang: str) -> list[CallSite]:
    """Run each signature once over the whole file (C-speed) and map hits to lines."""
    import bisect
    sites: list[CallSite] = []
    lines = None
    starts = None
    is_test = bool(TEST_PATH_RE.search(rel_path)) or bool(OTHER_TEST.search(rel_path))
    scope = "example" if EXAMPLE_PATH_RE.search(rel_path) else "app"
    calls = [(p, k, prov) for p, k, prov, need in LANG_CALLS.get(lang, []) if not need or need in text]

    # 1. SDK calls: first matching signature per line wins (signature order = priority)
    per_line: dict[int, tuple] = {}
    for order, (pat, kind, provider) in enumerate(calls):
        for hit in pat.finditer(text):
            if starts is None:
                starts = _line_index(text)
            i = bisect.bisect_right(starts, hit.start()) - 1
            if i in per_line and per_line[i][0] <= order:
                continue
            if "\n" in hit.group(0):
                continue
            per_line[i] = (order, hit, kind, provider)
    if per_line and lines is None:
        lines = text.splitlines()
    for i in sorted(per_line):
        line = lines[i] if i < len(lines) else ""
        if len(line) > 400 or COMMENT.match(line):
            continue
        _, hit, kind, provider = per_line[i]
        window = "\n".join(lines[i: i + 25])
        model, provider = _find_model(window, provider)
        if not model and lang in ("jvm", "csharp", "go"):
            # Builder-style APIs set the model on the request object before the call.
            before = "\n".join(lines[max(0, i - 15): i])
            model, provider = _find_model(before, provider)
        model_expr = None
        if not model:
            mx = MODEL_EXPR.search(window)
            model_expr = mx.group(1).strip()[:80] if mx else None
        if lang == "javascript" and provider == "anthropic" and "anthropic" not in text.lower():
            provider = "openai_compatible"
        params = {}
        for p in ("max_tokens", "maxTokens", "max_completion_tokens", "MaxTokens", "maxOutputTokens",
                  "MaxCompletionTokens", "temperature", "stream", "response_format", "tools"):
            if re.search(rf"\b{p}\s*[:=(]", window):
                params[p] = "<expr>"
        sites.append(CallSite(
            id=_sid(rel_path, i + 1, kind), path=rel_path, line=i + 1, kind=kind,
            provider=provider, sdk_call=re.sub(r"\s+", "", hit.group(1)), language=lang,
            model=model, model_expr=model_expr, model_source="literal" if model else "unresolved",
            params=params, uses_cache_control="cache_control" in window or "CacheControl" in window,
            is_test=is_test, confidence="heuristic", scope=scope,
        ))

    # 2. Raw HTTP: provider endpoint strings (not server route definitions)
    done = set(per_line)
    low = text.lower()
    if not any(t in low for t in ("chat/completions", "/messages", "/responses", "/embeddings",
                                  "generatecontent")):
        sites.sort(key=lambda x: x.line)
        return sites
    if starts is None:
        starts = _line_index(text)
    if lines is None:
        lines = text.splitlines()
    for hint in ENDPOINT_HINT.finditer(text):
        i = bisect.bisect_right(starts, hint.start()) - 1
        if i in done:
            continue
        done.add(i)
        line = lines[i] if i < len(lines) else ""
        if len(line) > 400 or COMMENT.match(line) or ROUTE_DEF.search(line):
            continue
        for sm in STRING.finditer(line):
            ep = http_endpoint(sm.group(2))
            if ep:
                kind, provider, path = ep
                sites.append(CallSite(
                    id=_sid(rel_path, i + 1, kind), path=rel_path, line=i + 1, kind=kind,
                    provider=provider, sdk_call=f"HTTP {path}", language=lang,
                    model_expr="set in request payload", model_source="unresolved",
                    is_test=is_test, confidence="heuristic", scope=scope,
                ))
                break
    sites.sort(key=lambda x: x.line)
    return sites


def scan_js(rel_path: str, text: str) -> list[CallSite]:
    """Backwards-compatible entry point."""
    return scan_heuristic(rel_path, text, "javascript")


_MODEL_PREFIX = (r"(?:gpt-\d|chatgpt-|o[1-9]\b|o[1-9]-|claude-|gemini-|text-embedding-|"
                 r"(?:us\.|eu\.|apac\.|global\.)?anthropic\.claude|(?:meta|amazon|mistral|cohere)\.|"
                 r"(?:openai|anthropic|gemini|azure|bedrock|groq|together_ai|mistral|deepseek)/|"
                 r"mistral-|deepseek-|llama-?\d)")
MODEL_TOKEN = re.compile(
    r"""['"](""" + _MODEL_PREFIX + r"""[\w.\-:/]{0,78})['"]|(?:=|:)[ \t]*(""" + _MODEL_PREFIX
    + r"""[\w.\-:/]{0,78})[ \t]*$""", re.M)


_MODEL_HINTS = ("gpt-", "chatgpt", "o1", "o3", "o4", "o5", "o6", "o7", "o8", "o9", "claude", "gemini",
                "text-embedding", "anthropic.", "meta.", "amazon.", "mistral", "cohere.", "openai/",
                "gemini/", "azure/", "bedrock/", "groq/", "together_ai/", "deepseek", "llama")


def scan_model_refs(rel_path: str, text: str) -> list[tuple[int, str]]:
    """Model-name literals anywhere in the file (quoted, or as an unquoted config value)."""
    import bisect
    refs = []
    starts = None
    low = text.lower()
    if not any(t in low for t in _MODEL_HINTS):
        return refs
    for m in MODEL_TOKEN.finditer(text):
        v = (m.group(1) or m.group(2) or "").strip()
        if not v or not MODEL_NAME_RE.match(v):
            continue
        if starts is None:
            starts = _line_index(text)
        i = bisect.bisect_right(starts, m.start()) - 1
        refs.append((i + 1, v))
    return refs
