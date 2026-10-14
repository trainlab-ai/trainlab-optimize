"""Heuristic (regex) scanning for JS/TS, Go, Java/Kotlin, C# sources and config files.

Lower confidence than the Python AST scanner; sites are marked confidence="heuristic".
"""
from __future__ import annotations

import hashlib
import re

from ..schema import CallSite
from .providers import file_provider_hint, maker_of, provider_from_model_prefix, provider_from_url
from .python_ast import EXAMPLE_PATH_RE, MODEL_NAME_RE, TEST_PATH_RE, http_endpoint

JS_EXTS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
GO_EXTS = {".go"}
JVM_EXTS = {".java", ".kt", ".kts"}
CS_EXTS = {".cs"}
RUBY_EXTS = {".rb"}
PHP_EXTS = {".php"}
RUST_EXTS = {".rs"}
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
        # Vendor SDKs that aren't OpenAI-shaped
        (re.compile(r"\.(chat\.(?:complete|stream|parse))\s*\("), "chat", "mistral", "mistralai"),
        (re.compile(r"\.(chat(?:Stream)?)\s*\(\s*\{"), "chat", "cohere", "cohere-ai"),
        (re.compile(r"\.(embed)\s*\(\s*\{"), "embedding", "cohere", "cohere-ai"),
        (re.compile(r"\b(ollama\.(?:chat|generate|embed))\s*\("), "chat", "ollama", "ollama"),
        (re.compile(r"\.(chatCompletion(?:Stream)?|textGeneration)\s*\(\s*\{"), "chat", "huggingface",
         "@huggingface/inference"),
        (re.compile(r"\b(replicate\.run)\s*\("), "chat", "replicate", "replicate"),
        # LangChain.js model classes
        (re.compile(r"\bnew\s+(Chat(?:OpenAI|Anthropic|GoogleGenerativeAI|VertexAI|Groq|MistralAI|Ollama|Cohere|"
                    r"DeepSeek|XAI|TogetherAI|Fireworks|AlibabaTongyi|BedrockConverse|Bedrock|ZhipuAI|"
                    r"Moonshot|Cerebras|Perplexity|OpenRouter)|AzureChatOpenAI|OpenAIEmbeddings|"
                    r"(?:Cohere|MistralAI|Ollama|Voyage)Embeddings)\s*\("),
         "model_config", "unknown", "langchain"),
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
        (re.compile(r"\b([A-Z]\w*(?:Chat|Language|Embedding|Image|Scoring|Moderation)Model\.builder)\(\)"),
         "model_config", "unknown", "langchain4j"),
        (re.compile(r"\b(chatClient\.prompt)\("), "chat", "unknown", "springframework.ai"),
        # Kotlin: aallam/openai-kotlin
        (re.compile(r"\.(chatCompletions?)\("), "chat", "openai", "com.aallam.openai"),
        (re.compile(r"\.(embeddings)\(\s*(?:request\s*=\s*)?EmbeddingRequest"), "embedding", "openai",
         "com.aallam.openai"),
    ],
    "ruby": [
        # ruby-openai / anthropic (alexrudall), openai-ruby (official), RubyLLM, langchainrb
        (re.compile(r"\.(chat)\(\s*parameters:"), "chat", "openai", None),
        (re.compile(r"\.(embeddings)\(\s*parameters:"), "embedding", "openai", None),
        (re.compile(r"\.(messages)\(\s*parameters:"), "chat", "anthropic", None),
        (re.compile(r"\.(chat\.completions\.create)\("), "chat", "openai", None),
        (re.compile(r"\.(messages\.create)\("), "chat", "anthropic", "anthropic"),
        (re.compile(r"\b(RubyLLM\.(?:chat|embed))\b"), "chat", "unknown", None),
        (re.compile(r"\b(Langchain::LLM::\w+\.new)\b"), "model_config", "unknown", None),
        (re.compile(r"\.(invoke_model(?:_with_response_stream)?|converse(?:_stream)?)\("), "chat", "bedrock",
         "Aws::BedrockRuntime"),
    ],
    "php": [
        # openai-php/client and Laravel facade, Prism, LLPhant
        (re.compile(r"->(chat\(\)\s*->\s*create(?:Streamed)?)\("), "chat", "openai", None),
        (re.compile(r"->(responses\(\)\s*->\s*create(?:Streamed)?)\("), "chat", "openai", None),
        (re.compile(r"->(embeddings\(\)\s*->\s*create)\("), "embedding", "openai", None),
        (re.compile(r"->(messages\(\)\s*->\s*create(?:Streamed)?)\("), "chat", "anthropic", None),
        (re.compile(r"\b(OpenAI::(?:chat|responses)\(\)\s*->\s*create)\("), "chat", "openai", None),
        (re.compile(r"\b(Prism::(?:text|structured)\(\))"), "chat", "unknown", None),
        (re.compile(r"\bnew\s+((?:OpenAI|Anthropic|Mistral|Ollama)Chat)\("), "model_config", "unknown", None),
    ],
    "rust": [
        # async-openai, rig, genai crates
        (re.compile(r"\.(chat\(\)\s*\.create(?:_stream)?)\("), "chat", "openai", "async_openai"),
        (re.compile(r"\.(responses\(\)\s*\.create(?:_stream)?)\("), "chat", "openai", "async_openai"),
        (re.compile(r"\.(embeddings\(\)\s*\.create)\("), "embedding", "openai", "async_openai"),
        (re.compile(r"\.(agent|completion_model|embedding_model)\(\s*(?:openai|anthropic|gemini|\")"),
         "model_config", "unknown", "rig"),
        (re.compile(r"\.(exec_chat(?:_stream)?)\("), "chat", "unknown", "genai"),
        (re.compile(r"\.(converse(?:_stream)?|invoke_model(?:_with_response_stream)?)\(\)"), "chat", "bedrock",
         "aws_sdk_bedrockruntime"),
    ],
    "csharp": [
        (re.compile(r"\.(CompleteChat(?:Streaming)?(?:Async)?)\("), "chat", "openai", "OpenAI"),
        (re.compile(r"\.(GetChatCompletions(?:Streaming)?(?:Async)?)\("), "chat", "azure", "Azure.AI.OpenAI"),
        (re.compile(r"\.(GenerateEmbeddings?(?:Async)?)\("), "embedding", "openai", "OpenAI"),
        (re.compile(r"\.(Complete(?:Streaming)?Async)\("), "chat", "azure", "Azure.AI.Inference"),
        (re.compile(r"\.(Add(?:Azure)?OpenAI(?:ChatCompletion|TextEmbeddingGeneration|EmbeddingGenerator)|"
                    r"AddGoogleAIGeminiChatCompletion|AddBedrockChatCompletionService|AddAnthropicChatCompletion)\("),
         "model_config", "unknown", "SemanticKernel"),
        (re.compile(r"\.(GetChatMessageContents?Async|GetStreamingChatMessageContentsAsync|InvokePromptAsync)\("),
         "chat", "unknown", "SemanticKernel"),
    ],
}
MODEL_PATTERNS = [
    re.compile(r"""\b(?:model|modelId|Model|ModelID|modelName|deploymentName|model_id)\s*[:=]\s*['"`]([^'"`]+)['"`]"""),
    re.compile(r"""['"](?:model|model_name)['"]\s*=>\s*['"]([^'"]+)['"]"""),           # Ruby/PHP hashes
    re.compile(r"""\bModelId\(\s*"([^"]+)"\s*\)"""),                                   # Kotlin
    re.compile(r"""->using\(\s*Provider::\w+\s*,\s*['"]([^'"]+)['"]"""),             # Prism PHP
    re.compile(r'\.(?:agent|completion_model|embedding_model)\(\s*"([^"]+)"'),         # rig
    re.compile(r"""\b(?:openai|anthropic|gemini)::([A-Z][A-Z0-9_]{2,})\b"""),           # rig constants
    re.compile(r"""\.(?:model|modelName|modelId)\(\s*"([^"]+)"\s*\)"""),
    re.compile(r"""\b(openai|anthropic|google|azure|bedrock|mistral|groq|openrouter|deepseek|xai|cohere|"""
               r"""togetherai|fireworks|perplexity|cerebras|deepinfra|alibaba|qwen|ollama|vertex|moonshotai|"""
               r"""zhipu|vercel|gateway)(?:\.(?:chat|languageModel|textEmbeddingModel))?\s*\(\s*['"`]([^'"`]+)['"`]"""),
    re.compile(r"""\b(?:ChatModel|Model|EmbeddingModel)\.([A-Z][A-Z0-9_]{2,})\b"""),
]
MODEL_EXPR = re.compile(r"""\b(?:model|modelId|Model|modelName)\s*[:=]\s*([^,\n}\)]+)""")
STRING = re.compile(r"""(["'`])((?:(?!\1).){3,200}?)\1""")
ROUTE_DEF = re.compile(r"""\.(?:GET|POST|PUT|Handle|HandleFunc|Any|Group|get|post|put|route|all)\s*\(\s*["'`]|"""
                       r"""@(?:Post|Get|Request)Mapping|\[Http(?:Post|Get)|Map(?:Post|Get)\(|\brouter\.|"""
                       r"""Request\.Path|PathString|\.path\s*===?|\.url\s*===?|pathname\s*===?""")
COMMENT = re.compile(r"^\s*(?://|#|\*|/\*|--)")
QUOTED = re.compile(r"""['"]([A-Za-z0-9][\w.\-:/]{1,79})['"]|(?:=|:)\s*([A-Za-z0-9][\w.\-:/]{1,79})\s*$""")
OTHER_TEST = re.compile(r"\.(test|spec|vitest|e2e|stories)\.[cm]?[jt]sx?$|(^|/)(locales?|i18n)/|_test\.go$|(^|/)src/test/|(^|/)[^/]*\.Tests?/|Tests?\.cs$|"
                        r"_spec\.rb$|(^|/)spec/|Test\.php$|(^|/)benches/")


def lang_of(ext: str) -> str | None:
    if ext in JS_EXTS:
        return "javascript"
    if ext in GO_EXTS:
        return "go"
    if ext in JVM_EXTS:
        return "jvm"
    if ext in CS_EXTS:
        return "csharp"
    if ext in RUBY_EXTS:
        return "ruby"
    if ext in PHP_EXTS:
        return "php"
    if ext in RUST_EXTS:
        return "rust"
    return None


def _sid(path: str, line: int, kind: str) -> str:
    return hashlib.sha1(f"{path}:{line}:{kind}".encode()).hexdigest()[:12]


def _enum_to_model(name: str) -> str:
    # openai-java ChatModel.GPT_4O -> gpt-4o; Model.CLAUDE_SONNET_4_5 -> claude-sonnet-4-5
    return name.lower().replace("_", "-")


_ENUM_PATTERNS = {5, 8}          # indexes of MODEL_PATTERNS that capture SCREAMING_CASE constants
_PROVIDER_FN_PATTERN = 7


def _find_model(window: str, provider: str) -> tuple[str | None, str]:
    for idx, mp in enumerate(MODEL_PATTERNS):
        m = mp.search(window)
        if not m:
            continue
        groups = [g for g in m.groups() if g]
        if idx in _ENUM_PATTERNS:
            return _enum_to_model(groups[-1]), provider
        if idx == _PROVIDER_FN_PATTERN and len(groups) == 2 and provider in ("vercel_ai", "unknown"):
            provider = {"google": "gemini", "togetherai": "together", "alibaba": "dashscope",
                        "qwen": "dashscope", "moonshotai": "moonshot", "vertex": "vertex",
                        "vercel": "vercel_ai_gateway", "gateway": "vercel_ai_gateway"}.get(groups[0], groups[0])
        return groups[-1], provider
    return None, provider


def _line_index(text: str):
    starts = [0]
    for m in re.finditer("\n", text):
        starts.append(m.end())
    return starts


ENDPOINT_HINT = re.compile(r"chat/completions|/messages|/responses|/embeddings|generatecontent|"
                           r"services/aigc|services/embeddings|:11434/api|api\.cohere\.(?:com|ai)/v2", re.I)
_LC_CLASS_PROVIDER = {
    "ChatOpenAI": "openai", "AzureChatOpenAI": "azure", "OpenAIEmbeddings": "openai", "ChatAnthropic": "anthropic",
    "ChatGoogleGenerativeAI": "gemini", "ChatVertexAI": "vertex", "ChatGroq": "groq", "ChatMistralAI": "mistral",
    "ChatOllama": "ollama", "ChatCohere": "cohere", "ChatDeepSeek": "deepseek", "ChatXAI": "xai",
    "ChatTogetherAI": "together", "ChatFireworks": "fireworks", "ChatAlibabaTongyi": "dashscope",
    "ChatBedrockConverse": "bedrock", "ChatBedrock": "bedrock", "ChatZhipuAI": "zhipu", "ChatMoonshot": "moonshot",
    "ChatCerebras": "cerebras", "ChatPerplexity": "perplexity", "ChatOpenRouter": "openrouter",
    "CohereEmbeddings": "cohere", "MistralAIEmbeddings": "mistral", "OllamaEmbeddings": "ollama",
    "VoyageEmbeddings": "voyage",
}


_BARE_URL = re.compile(r"https?://[^\s'\"`]+")
_CONST_NAME = re.compile(r"\b[A-Z][A-Z0-9_]*(?:BASE_URL|_URL|_ENDPOINT|_HOST|API_KEY)\b")


def _nearby_provider(lines: list[str], i: int, reach: int = 4) -> str | None:
    """Closest line (within `reach`) naming a provider by a bare base URL or a provider-named constant."""
    from .providers import provider_from_env
    for d in [x for k in range(1, reach + 1) for x in (-k, k)]:
        j = i + d
        if not 0 <= j < len(lines):
            continue
        line = lines[j]
        for um in _BARE_URL.finditer(line):
            if not http_endpoint(um.group(0)):          # skip other endpoints; only base URLs
                p = provider_from_url(um.group(0))
                if p:
                    return p
        for cm in _CONST_NAME.finditer(line):
            p = provider_from_env(cm.group(0))
            if p and p not in ("openai", "anthropic"):
                return p
    return None


def _stem_provider(rel_path: str) -> str | None:
    """providers/deepseek.ts -> deepseek (only for files named after one provider)."""
    from .providers import provider_from_class_name
    stem = rel_path.rsplit("/", 1)[-1].split(".")[0].lower()
    if "compat" in stem or len(stem) > 30:
        return None
    p = provider_from_class_name(stem)
    return p if p not in ("openai", "anthropic", "gemini") or stem in ("openai", "anthropic", "gemini") else None


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
    file_hint = file_provider_hint(text) if (per_line or "completions" in text or "/messages" in text) else None
    for i in sorted(per_line):
        line = lines[i] if i < len(lines) else ""
        if len(line) > 400 or COMMENT.match(line):
            continue
        _, hit, kind, provider = per_line[i]
        window = "\n".join(lines[i: i + 25])
        model, provider = _find_model(window, provider)
        if not model and (lang in ("jvm", "csharp", "go", "rust", "php") or provider == "vercel_ai"):
            # Builder-style APIs set the model on the request object before the call.
            before = "\n".join(lines[max(0, i - 15): i])
            model, provider = _find_model(before, provider)
        model_expr = None
        if not model:
            mx = MODEL_EXPR.search(window)
            model_expr = mx.group(1).strip()[:80] if mx else None
        if lang == "javascript" and provider == "anthropic" and "anthropic" not in text.lower():
            provider = "openai_compatible"
        if kind == "model_config" and provider == "unknown":
            provider = _LC_CLASS_PROVIDER.get(hit.group(1), "unknown")
            if provider == "openai" and file_hint:
                provider = file_hint            # new ChatOpenAI({ configuration: { baseURL: openrouter } })
        elif provider in ("openai", "openai_compatible", "vercel_ai") and file_hint:
            provider = file_hint                # OpenAI SDK pointed at OpenRouter, DashScope, Groq, ...
        if provider in ("unknown", "openai_compatible", "vercel_ai") and model:
            provider = provider_from_model_prefix(model) or (maker_of(model) if provider == "unknown" else None) \
                or provider
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
                                  "generatecontent", "services/aigc", "services/embeddings", ":11434/api",
                                  "/v2/chat")):
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
                if provider == "openai_compatible":
                    # Provider config blocks: { endpoint: MOONSHOT_BASE_URL, chatPath: "v1/chat/completions" }
                    provider = _nearby_provider(lines, i) or provider
                    if provider == "openai_compatible":
                        provider = file_hint or _stem_provider(rel_path) or provider
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


_MODEL_PREFIX = (r"(?:gpt-\d|gpt-oss-|chatgpt-|o[1-9]\b|o[1-9]-|claude-|gemini-|text-embedding-|"
                 r"(?:us\.|eu\.|apac\.|global\.)?anthropic\.claude|(?:meta|amazon|mistral|cohere|qwen)\.|"
                 r"(?:openai|anthropic|gemini|azure|bedrock|groq|together_ai|mistral|mistralai|deepseek|"
                 r"openrouter|dashscope|xai|x-ai|cohere|fireworks_ai|moonshotai|zai|z-ai|qwen|meta-llama)/|"
                 r"mistral-|mixtral-|codestral-|ministral-|deepseek-|llama-?\d|qwen[\d\-]|qwq-|glm-\d|kimi-|"
                 r"moonshot-v\d|grok-\d|command-r|command-a|doubao-|ernie-\d|hunyuan-|minimax-|gemma-?\d|"
                 r"sonar-pro|sonar-reasoning)")
MODEL_TOKEN = re.compile(
    r"""['"](""" + _MODEL_PREFIX + r"""[\w.\-:/]{0,78})['"]|(?:=|:)[ \t]*(""" + _MODEL_PREFIX
    + r"""[\w.\-:/]{0,78})[ \t]*$""", re.M)


_MODEL_HINTS = ("gpt-", "chatgpt", "o1", "o3", "o4", "o5", "o6", "o7", "o8", "o9", "claude", "gemini",
                "text-embedding", "anthropic.", "meta.", "amazon.", "mistral", "cohere.", "openai/",
                "gemini/", "azure/", "bedrock/", "groq/", "together_ai/", "deepseek", "llama", "qwen", "qwq",
                "glm-", "kimi", "moonshot", "grok-", "command-", "doubao", "ernie", "hunyuan", "minimax",
                "gemma", "sonar-", "openrouter/", "dashscope/", "codestral", "mixtral")


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
