"""Python AST scanner: finds LLM call sites, resolves models, and analyses prompts.

Pure stdlib. Never executes the scanned code.
"""
from __future__ import annotations

import ast
import hashlib
import re
import warnings
from dataclasses import dataclass, field

from ..redact import redact
from ..schema import CallSite
from .providers import (SELF_HOSTED, file_provider_hint, maker_of, provider_from_class_name, provider_from_env,
                        provider_from_model_prefix, provider_from_url)


def quiet_parse(source: str, filename: str) -> ast.AST:
    """Parse to an AST without printing SyntaxWarnings (e.g. invalid escapes) from scanned code."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compile(source, filename, "exec",
                       flags=ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT, dont_inherit=True)

# --------------------------------------------------------------------------- signatures

# (suffix of dotted call chain, kind, default provider)
CALL_SUFFIXES: list[tuple[str, str, str]] = [
    # Assistants API: a run is the billed call (thread messages are just stored)
    (".threads.runs.create_and_poll", "chat", "openai"),
    (".threads.runs.create", "chat", "openai"),
    (".threads.runs.stream", "chat", "openai"),
    (".threads.create_and_run", "chat", "openai"),
    (".beta.chat.completions.parse", "chat", "openai"),
    (".chat.completions.create", "chat", "openai"),
    (".chat.completions.parse", "chat", "openai"),
    (".chat.completions.stream", "chat", "openai"),
    (".responses.create", "chat", "openai"),
    (".responses.parse", "chat", "openai"),
    (".responses.stream", "chat", "openai"),
    (".messages.batches.create", "batch", "anthropic"),
    (".beta.messages.create", "chat", "anthropic"),
    (".messages.create", "chat", "anthropic"),
    (".messages.stream", "chat", "anthropic"),
    (".messages.parse", "chat", "anthropic"),
    (".embeddings.create", "embedding", "openai"),
    (".batches.create", "batch", "openai"),
    (".completions.create", "chat", "openai"),
    (".models.generate_content_stream", "chat", "gemini"),
    (".models.generate_content", "chat", "gemini"),
    (".models.embed_content", "embedding", "gemini"),
    (".invoke_model_with_response_stream", "chat", "bedrock"),
    (".invoke_model", "chat", "bedrock"),
    (".converse_stream", "chat", "bedrock"),
    (".converse", "chat", "bedrock"),
]

# OpenAI-shaped method chains that are specific enough to trust without an SDK import
# (suffix, kind, required companion kwargs or None)
INJECTED_SUFFIXES = [
    (".chat.completions.create", "chat", ("messages",)),
    (".chat.completions.parse", "chat", ("messages",)),
    (".responses.create", "chat", ("input", "instructions", "prompt")),
    (".responses.parse", "chat", ("input", "instructions")),
    (".embeddings.create", "embedding", ("input",)),
]

LITELLM_FUNCS = {
    "completion": "chat", "acompletion": "chat", "text_completion": "chat",
    "embedding": "embedding", "aembedding": "embedding", "batch_completion": "chat",
    "responses": "chat", "aresponses": "chat",
}

# Framework constructors that pin a model: class name -> provider
CONFIG_CLASSES = {
    "ChatOpenAI": "openai", "AzureChatOpenAI": "azure", "OpenAIEmbeddings": "openai",
    "AzureOpenAIEmbeddings": "azure", "ChatAnthropic": "anthropic", "ChatBedrock": "bedrock",
    "ChatBedrockConverse": "bedrock", "BedrockEmbeddings": "bedrock",
    "ChatGoogleGenerativeAI": "gemini", "ChatVertexAI": "gemini", "ChatLiteLLM": "litellm",
    "ChatGroq": "groq", "ChatMistralAI": "mistral", "ChatCohere": "cohere", "ChatTogether": "together",
    "ChatFireworks": "fireworks", "ChatDeepSeek": "deepseek", "ChatXAI": "xai", "ChatOllama": "ollama",
    "OllamaLLM": "ollama", "ChatTongyi": "dashscope", "Tongyi": "dashscope", "ChatZhipuAI": "zhipu",
    "ChatHuggingFace": "huggingface", "HuggingFaceEndpoint": "huggingface", "ChatPerplexity": "perplexity",
    "ChatNVIDIA": "nvidia", "ChatCerebras": "cerebras", "ChatSambaNovaCloud": "sambanova",
    "MoonshotChat": "moonshot", "QianfanChatEndpoint": "qianfan", "ChatHunyuan": "tencent",
    "VolcEngineMaasChat": "volcengine", "ChatDatabricks": "databricks", "ChatWatsonx": "watsonx",
    "ChatOpenRouter": "openrouter", "CohereEmbeddings": "cohere", "MistralAIEmbeddings": "mistral",
    "OllamaEmbeddings": "ollama", "DashScopeEmbeddings": "dashscope", "VoyageAIEmbeddings": "voyage",
    "ChatMiniMax": "minimax", "MiniMaxChat": "minimax", "ChatBaichuan": "baichuan", "ChatYi": "01ai",
    # LlamaIndex (only when imported from llama_index)
    "Groq": "groq", "MistralAI": "mistral", "Cohere": "cohere", "Ollama": "ollama", "OpenRouter": "openrouter",
    "DashScope": "dashscope", "TogetherLLM": "together", "Fireworks": "fireworks", "DeepSeek": "deepseek",
    "Perplexity": "perplexity", "HuggingFaceInferenceAPI": "huggingface", "Vertex": "vertex", "NVIDIA": "nvidia",
    "OpenAILike": "openai_compatible", "MistralAIEmbedding": "mistral", "CohereEmbedding": "cohere",
    "OllamaEmbedding": "ollama", "DashScopeEmbedding": "dashscope", "VoyageEmbedding": "voyage",
    # LlamaIndex
    "OpenAI": "openai", "Anthropic": "anthropic", "AzureOpenAI": "azure", "Bedrock": "bedrock",
    "OpenAIEmbedding": "openai", "Gemini": "gemini",
}
CONFIG_FUNCS = {"init_chat_model": "unknown"}

CLIENT_CLASSES = {
    "OpenAI": "openai", "AsyncOpenAI": "openai", "AzureOpenAI": "azure",
    "AsyncAzureOpenAI": "azure", "Anthropic": "anthropic", "AsyncAnthropic": "anthropic",
    "AnthropicBedrock": "bedrock", "AsyncAnthropicBedrock": "bedrock",
    "AnthropicVertex": "gemini", "AsyncAnthropicVertex": "gemini", "Client": None,
}
LLM_IMPORT_ROOTS = {"openai", "anthropic", "litellm", "boto3", "aioboto3", "aiobotocore", "botocore", "google", "langchain_openai",
                    "langchain_anthropic", "langchain_aws", "langchain_google_genai",
                    "langchain_google_vertexai", "langchain", "langchain_community",
                    "llama_index", "pydantic_ai", "agents", "instructor", "mistralai",
                    "groq", "together", "cohere", "dashscope", "fireworks", "cerebras", "zhipuai", "zai",
                    "volcenginesdkarkruntime", "huggingface_hub", "ollama", "replicate", "voyageai", "xai_sdk",
                    "openrouter", "perplexity", "sambanova", "ai21", "vertexai", "aisuite", "qianfan"}

# SDK module -> provider, for client objects built from that SDK (cohere.ClientV2(), Groq(), ...)
SDK_MODULES = {"groq": "groq", "together": "together", "fireworks": "fireworks", "cerebras": "cerebras",
               "zhipuai": "zhipu", "zai": "zhipu", "volcenginesdkarkruntime": "volcengine", "mistralai": "mistral",
               "cohere": "cohere", "huggingface_hub": "huggingface", "ollama": "ollama", "voyageai": "voyage",
               "xai_sdk": "xai", "dashscope": "dashscope", "replicate": "replicate", "openrouter": "openrouter",
               "perplexity": "perplexity", "sambanova": "sambanova", "ai21": "ai21", "qianfan": "qianfan"}
SDK_CLASS_NAMES = {"Groq", "AsyncGroq", "Together", "AsyncTogether", "Fireworks", "AsyncFireworks", "Cerebras",
                   "AsyncCerebras", "ZhipuAI", "ZaiClient", "ZhipuAiClient", "Ark", "AsyncArk", "Mistral",
                   "MistralClient", "MistralAsyncClient", "InferenceClient", "AsyncInferenceClient", "OpenRouter",
                   "Perplexity", "AsyncPerplexity", "SambaNova", "AI21Client", "AsyncAI21Client"}
# Native (non-OpenAI-shaped) methods on a client of that provider: method path after the client -> kind
_IN_PROGRESS = object()
_EXIT = object()
_FACT_TYPES = frozenset({ast.Constant, ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith, ast.Assign, ast.AnnAssign})
_FIELDS_CACHE: dict[type, tuple[str, ...]] = {}
_FACTORY_RE = re.compile(r"_?(get_|create_|make_|build_)?(azure_)?(async_)?openai_client")
_DISTINCT_METHODS = {"chat_completion", "text_generation", "feature_extraction", "contextualized_embed",
                     "chat_stream"}
CLIENT_METHODS = {
    "cohere": {"chat": "chat", "chat_stream": "chat", "embed": "embedding", "generate": "chat"},
    "ollama": {"chat": "chat", "generate": "chat", "embed": "embedding", "embeddings": "embedding"},
    "huggingface": {"chat_completion": "chat", "text_generation": "chat", "feature_extraction": "embedding"},
    "voyage": {"embed": "embedding", "contextualized_embed": "embedding"},
    "mistral": {"chat.complete": "chat", "chat.complete_async": "chat", "chat.stream": "chat",
                "chat.stream_async": "chat", "chat.parse": "chat", "fim.complete": "chat", "chat": "chat",
                "embeddings.create": "embedding", "embeddings": "embedding"},
    "xai": {"chat.create": "chat"},
    "replicate": {"run": "chat", "stream": "chat"},
}
# Module-level functions: chain -> (kind, provider, module that must be imported)
MODULE_FUNCS = {
    "ollama.chat": ("chat", "ollama", "ollama"), "ollama.generate": ("chat", "ollama", "ollama"),
    "ollama.embed": ("embedding", "ollama", "ollama"), "ollama.embeddings": ("embedding", "ollama", "ollama"),
    "replicate.run": ("chat", "replicate", "replicate"), "replicate.stream": ("chat", "replicate", "replicate"),
    "dashscope.Generation.call": ("chat", "dashscope", "dashscope"),
    "Generation.call": ("chat", "dashscope", "dashscope"),
    "dashscope.AioGeneration.call": ("chat", "dashscope", "dashscope"),
    "AioGeneration.call": ("chat", "dashscope", "dashscope"),
    "dashscope.MultiModalConversation.call": ("chat", "dashscope", "dashscope"),
    "MultiModalConversation.call": ("chat", "dashscope", "dashscope"),
    "dashscope.TextEmbedding.call": ("embedding", "dashscope", "dashscope"),
    "TextEmbedding.call": ("embedding", "dashscope", "dashscope"),
}
# Constructors that pin a model for later generate_content() calls (google-generativeai, Vertex AI)
MODEL_CONSTRUCTORS = {"GenerativeModel": {"google": "gemini", "vertexai": "vertex"}}
POSITIONAL_MODEL = ("replicate.run", "replicate.stream", "GenerativeModel")

WRAPPER_MODEL_KWARGS = ("model", "model_name", "name", "llm", "llm_model", "summary_llm",
                        "agent_llm", "embedding", "embedding_model", "fast_llm", "smart_llm",
                        "strategic_llm", "judge_model", "ai_model_id", "model_id", "deployment_name")
LLM_IMPORT_PREFIXES = ("langchain", "llama_index", "openai_", "anthropic_", "litellm_")

MODEL_KWARGS = ("model", "modelId", "model_id", "model_name", "deployment_name", "azure_deployment", "model_version")
PARAM_KWARGS = ("max_tokens", "max_completion_tokens", "max_output_tokens", "temperature",
                "stream", "response_format", "text_format", "tools", "tool_choice", "n",
                "reasoning_effort", "reasoning", "service_tier", "seed", "top_p",
                "prompt_cache_key", "thinking", "config", "generation_config")
PROMPT_KWARGS = ("messages", "system", "instructions", "input", "prompt", "contents")

MODEL_NAME_RE = re.compile(
    r"^(gpt-[0-9][\w.\-]*|gpt-oss-[\w.\-]+|chatgpt-[\w.\-]+|o[1-9](?:-[\w.\-]+)?|claude-[\w.\-:]+|"
    r"gemini-[\w.\-]+|text-embedding-[\w.\-]+|(?:us|eu|apac|global)?\.?anthropic\.claude[\w.\-:]+|"
    r"(?:meta|amazon|mistral|cohere|qwen|deepseek|moonshotai|minimax|openai)\.[\w.\-:]+|"
    r"(?:openai|anthropic|gemini|azure|azure_ai|bedrock|groq|together_ai|mistral|mistralai|deepseek|deepseek-ai|"
    r"openrouter|dashscope|xai|x-ai|cohere|fireworks_ai|perplexity|moonshot|moonshotai|zai|z-ai|ollama|"
    r"ollama_chat|qwen|meta-llama|vertex_ai|huggingface|nvidia_nim|cerebras|deepinfra|volcengine|minimax)"
    r"/[\w.\-:/]+|"
    r"(?:mistral|mixtral|codestral|ministral|magistral|devstral|pixtral)-[\w.\-]+|"
    r"deepseek-(?:chat|reasoner|r1|v\d|coder)[\w.\-]*|llama-?\d[\w.\-:]*|"
    r"qwen[\d\-][\w.\-:]*|qwq-[\w.\-]+|glm-\d[\w.\-]*|kimi-[\w.\-]+|moonshot-v\d[\w.\-]*|"
    r"grok-\d[\w.\-]*|grok-(?:beta|code|vision)[\w.\-]*|command-(?:r|a|light|nightly)[\w.\-]*|"
    r"doubao-[\w.\-]+|ernie-\d[\w.\-]*|hunyuan-[\w.\-]+|minimax-[\w.\-]+|abab\d[\w.\-]*|"
    r"sonar-(?:pro|reasoning|deep)[\w.\-]*|gemma-?\d[\w.\-:]*|phi-?\d[\w.\-:]*|jamba-[\w.\-]+|"
    r"voyage-[\w.\-]+|embed-(?:english|multilingual|v\d)[\w.\-]*|nova-(?:pro|lite|micro|premier)[\w.\-:]*)$",
    re.IGNORECASE,
)

HINT_PATTERNS = [
    ("extraction", re.compile(r"\bextract|\bparse\b|\bpull out|\bfields?\b|\bentities\b", re.I)),
    ("classification", re.compile(r"\bclassif|\bcategori|\blabel\b|\bsentiment|yes or no|true or false|one of the following", re.I)),
    ("summarization", re.compile(r"\bsummar|\btl;?dr", re.I)),
    ("translation", re.compile(r"\btranslat", re.I)),
]

EXAMPLE_PATH_RE = re.compile(
    r"(^|/)(examples?|samples?|demos?|docs?|cookbooks?|notebooks?|benchmarks?|evals?|tutorials?|"
    r"playground|sandbox|scratch|quickstarts?)(/|$)", re.I)
TEST_PATH_RE = re.compile(r"(^|/)(tests?|testing|unittests?|e2e|__tests__|__mocks__|fixtures?|mocks?|"
                          r"[\w.-]+[-_]tests?|tests?[-_][\w.-]+)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")


# --------------------------------------------------------------------------- helpers

def dotted(node: ast.AST) -> str | None:
    """Dotted name of an expression; calls in the chain become '()'.

    client.with_options(timeout=5).chat.completions.create -> client.with_options().chat.completions.create
    """
    parts: list[str] = []
    cur = node
    while True:
        if isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        elif isinstance(cur, ast.Call):
            parts.append("()")
            cur = cur.func
        elif isinstance(cur, ast.Name):
            parts.append(cur.id)
            break
        elif isinstance(cur, ast.Subscript):
            parts.append("[]")
            cur = cur.value
        else:
            return None
    out = ""
    for p in reversed(parts):
        if p == "()":
            out += "()"
        elif p == "[]":
            out += "[]"
        else:
            out += ("." if out else "") + p
    return out


def receiver_root(chain: str) -> str:
    """'self.client.chat.completions.create' -> 'self.client'; 'client.chat...' -> 'client'."""
    head = chain.split("()")[0]
    bits = head.split(".")
    if bits[0] == "self" and len(bits) > 1:
        return "self." + bits[1]
    return bits[0]


def est_tokens(chars: int) -> int:
    return max(0, round(chars / 4))


@dataclass
class Template:
    static: str = ""
    dynamic_slots: int = 0
    first_dynamic_at: int | None = None   # index in static text where first slot appears
    resolved: bool = True

    def add_static(self, s: str) -> None:
        self.static += s

    def add_slot(self) -> None:
        if self.first_dynamic_at is None:
            self.first_dynamic_at = len(self.static)
        self.dynamic_slots += 1
        self.static += "{}"

    def merge(self, other: "Template") -> None:
        if other.first_dynamic_at is not None and self.first_dynamic_at is None:
            self.first_dynamic_at = len(self.static) + other.first_dynamic_at
        self.static += other.static
        self.dynamic_slots += other.dynamic_slots
        self.resolved = self.resolved and other.resolved

    @property
    def static_chars(self) -> int:
        return len(self.static.replace("{}", ""))


@dataclass
class FileFacts:
    imports: set[str] = field(default_factory=set)          # root modules
    from_imports: dict[str, str] = field(default_factory=dict)  # local name -> module.name
    module_aliases: dict[str, str] = field(default_factory=dict)  # local alias -> module
    constants: dict[str, list[ast.AST]] = field(default_factory=dict)  # NAME -> value nodes
    clients: dict[str, tuple[str, str | None]] = field(default_factory=dict)  # var -> (provider, base_url)
    retry_funcs: set[str] = field(default_factory=set)


# --------------------------------------------------------------------------- scanner

class PythonFileScanner:
    def __init__(self, rel_path: str, source: str, external: dict[str, ast.AST] | None = None):
        self.rel_path = rel_path
        self.source = source[1:] if source.startswith("\ufeff") else source   # UTF-8 BOM
        self.external = external or {}      # imported name -> value node from another module
        self.wanted: set[str] = set()       # fully-qualified imported names we could not resolve
        parts = rel_path[:-3].split("/") if rel_path.endswith(".py") else rel_path.split("/")
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        self.module = ".".join(parts)
        self.sites: list[CallSite] = []
        self.model_refs: list[tuple[int, str]] = []
        self.facts = FileFacts()
        self._stack: list[ast.AST] = []
        self._func_nodes: list[ast.AST] = []
        self._locals_cache: dict[int, dict[str, ast.AST]] = {}
        self._const_memo: dict[str, object] = {}
        self.parse_error = False
        self._func_params: list[dict[str, ast.AST]] = []
        self.is_test = bool(TEST_PATH_RE.search(rel_path))
        self.has_llm_import = False

    # ---- entry
    def scan(self) -> list[CallSite]:
        try:
            tree = quiet_parse(self.source.replace("\x00", ""), self.rel_path)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            self.parse_error = True
            return []
        # Single traversal: collect file facts and remember plausible call nodes with their
        # ancestors; resolve those candidates afterwards, once every fact in the file is known.
        candidates = self._traverse(tree)
        # Without an LLM SDK import only wrapper configs (Settings(llm="gpt-4o")) are recorded.
        self.has_llm_import = bool(self.facts.imports & LLM_IMPORT_ROOTS) or any(
            m.startswith(LLM_IMPORT_PREFIXES) for m in self.facts.imports)
        for call, ancestors in candidates:
            self._stack = list(ancestors)
            self._func_nodes = [n for n in ancestors if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            self._func_params = [_params_of(fn) for fn in self._func_nodes]
            self._check_call(call)
        seen_http = {s.line for s in self.sites}
        for const, (kind, provider, path), ancestors in self._http_candidates:
            if const.lineno in seen_http:
                continue
            seen_http.add(const.lineno)
            self._stack = list(ancestors)
            self._func_nodes = [n for n in ancestors if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            self._func_params = [_params_of(fn) for fn in self._func_nodes]
            func = self._enclosing_function()
            loop_kind = self._loop_kind()
            self.sites.append(CallSite(
                id=hashlib.sha1(f"{self.rel_path}:{const.lineno}:{kind}".encode()).hexdigest()[:12],
                path=self.rel_path, line=const.lineno, kind=kind, provider=provider,
                sdk_call=f"HTTP {path}", function=func.name if func else None,
                model_expr="set in request payload", model_source="unresolved",
                in_loop=loop_kind is not None, loop_kind=loop_kind,
                scope="example" if EXAMPLE_PATH_RE.search(self.rel_path) else "app",
                is_test=self.is_test, confidence="heuristic"))
        used = {s.line for s in self.sites}
        self.model_refs = [(ln, v) for ln, v in self._ref_candidates if ln not in used]
        return self.sites

    # ---- single pass: facts + candidate calls
    def _traverse(self, tree: ast.AST) -> list[tuple[ast.Call, tuple]]:
        """Iterative DFS in source order (no recursion limit issues on deep expressions).

        Hot path on big repos: dispatch on the node's type, collect facts only for the few node
        types that carry them, and skip `ctx` (Load/Store markers carry nothing)."""
        self._ref_candidates: list[tuple[int, str]] = []
        self._seen_refs: set = set()
        candidates: list[tuple[ast.Call, tuple]] = []
        self._http_candidates: list[tuple[ast.Constant, tuple, tuple]] = []
        stack: list[ast.AST] = []
        EXIT = _EXIT
        todo: list = [tree]
        pop, push = todo.pop, todo.append
        fact_types, Call, Constant, AST = _FACT_TYPES, ast.Call, ast.Constant, ast.AST
        fields_of = _FIELDS_CACHE
        while todo:
            node = pop()
            if node is EXIT:
                stack.pop()
                continue
            t = type(node)
            if t in fact_types:
                self._fact(node)
            if t is Call:
                if _is_candidate(node) or self._is_alias_candidate(node):
                    candidates.append((node, tuple(stack)))
            elif t is Constant:
                v = node.value
                if type(v) is str and ("completions" in v or "messages" in v or "responses" in v
                                       or "embeddings" in v or "enerateContent" in v or "/services/" in v
                                       or ":11434/api" in v or "/v2/" in v):
                    parent = stack[-1] if stack else None
                    # f"{base_url}/api/v1/chat/completions": a URL built on a variable host is a client
                    # call (a server route is never written that way)
                    dyn = isinstance(parent, ast.JoinedStr) and any(
                        x is node and i > 0 and isinstance(parent.values[i - 1], ast.FormattedValue)
                        for i, x in enumerate(parent.values))
                    ep = http_endpoint(v, dynamic_host=dyn)
                    if ep and not _in_decorator(stack):
                        self._http_candidates.append((node, ep, tuple(stack)))
                continue                       # constants have no children
            stack.append(node)
            push(EXIT)
            fields = fields_of.get(t)
            if fields is None:
                fields = fields_of[t] = tuple(f for f in t._fields if f not in ("ctx", "type_comment"))
            children = []
            for name in fields:
                val = getattr(node, name, None)
                if val is None:
                    continue
                if type(val) is list:
                    for item in val:
                        if isinstance(item, AST):
                            children.append(item)
                elif isinstance(val, AST):
                    children.append(val)
            for c in reversed(children):
                push(c)
        return candidates

    def _is_alias_candidate(self, node: ast.Call) -> bool:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id == "partial" or self.facts.from_imports.get(func.id, "").startswith("litellm."):
                return True
        elif isinstance(func, ast.Attribute) and func.attr == "partial":
            return True
        return False

    def _fact(self, node: ast.AST) -> None:
        f = self.facts
        if True:
            if isinstance(node, ast.Constant):
                v = node.value
                if isinstance(v, str) and 1 < len(v) < 80 and MODEL_NAME_RE.match(v.strip()):
                    key = (node.lineno, v.strip())
                    if key not in self._seen_refs:
                        self._seen_refs.add(key)
                        self._ref_candidates.append(key)
                return
            if isinstance(node, ast.Import):
                for a in node.names:
                    f.imports.add(a.name.split(".")[0])
                    if a.asname:
                        f.module_aliases[a.asname] = a.name
            elif isinstance(node, ast.ImportFrom) and (node.module or node.level):
                module = node.module or ""
                if node.level:   # relative import: resolve against this file's package
                    base = self.module.split(".")
                    base = base[: len(base) - node.level] if len(base) >= node.level else []
                    module = ".".join([*base, module] if module else base)
                if module:
                    f.imports.add(module.split(".")[0])
                for a in node.names:
                    f.from_imports[a.asname or a.name] = f"{module}.{a.name}" if module else a.name
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if isinstance(item.optional_vars, ast.Name):
                        client = self._client_from_call(item.context_expr)
                        if client:
                            f.clients[item.optional_vars.id] = client
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                if value is None:
                    return
                for t in targets:
                    name = t.id if isinstance(t, ast.Name) else (
                        f"self.{t.attr}" if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                        and t.value.id == "self" else None)
                    if not name:
                        continue
                    f.constants.setdefault(name, []).append(value)
                    if isinstance(t, ast.Attribute):
                        f.constants.setdefault(t.attr, []).append(value)
                    client = self._client_from_call(value)
                    if client:
                        f.clients[name] = client

    def _client_from_call(self, value: ast.AST) -> tuple[str, str | None] | None:
        if isinstance(value, ast.Await):
            value = value.value
        if not isinstance(value, ast.Call):
            return None
        name = dotted(value.func) or ""
        last = name.split(".")[-1]
        # Client factories: AIProjectClient.get_openai_client(), get_azure_openai_client(), _openai_client()
        if _FACTORY_RE.fullmatch(last):
            return ("azure" if "azure" in self.facts.imports or "azure" in last else "openai", None)
        base_url = None
        for kw in value.keywords:
            if kw.arg == "base_url" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                base_url = kw.value.value
        origin = self.facts.from_imports.get(last) or self.facts.from_imports.get(name.split(".")[0], "")
        origin = origin or name
        if name.split(".")[0] in self.facts.module_aliases:
            origin = self.facts.module_aliases[name.split(".")[0]] + "." + ".".join(name.split(".")[1:])
        sdk = SDK_MODULES.get(origin.split(".")[0])
        # Vendor SDK clients: Groq(), cohere.ClientV2(), ollama.Client(), voyageai.Client(), Ark(), ...
        if sdk and last[:1].isupper() and (last in SDK_CLASS_NAMES or last.endswith(("Client", "ClientV2"))):
            return (sdk, base_url)
        if last in MODEL_CONSTRUCTORS:
            root = origin.split(".")[0]
            prov = MODEL_CONSTRUCTORS[last].get(root) or (
                "gemini" if "generativeai" in origin else "vertex" if "vertexai" in origin else None)
            if prov:
                return (prov, None)
        if last in CLIENT_CLASSES:
            if last == "Client":
                if "genai" in origin or "genai" in name:
                    return ("gemini", None)
                return None
            provider = CLIENT_CLASSES[last]
            if provider in ("openai", "anthropic", "azure"):
                # OpenAI()/Anthropic() pointed at another provider: OpenRouter, DashScope, DeepSeek, Groq, ...
                ep = self._endpoint_provider(value)
                if ep and ep != provider and not (provider == "azure" and ep == "openai"):
                    return (ep, base_url)
                if provider == "openai" and (base_url or any(kw.arg == "base_url" for kw in value.keywords)):
                    return ("openai_compatible", base_url)
            return (provider, base_url)
        if name.endswith("client") or name.endswith(".client"):
            # boto3.client("bedrock-runtime"), session.client(service_name="bedrock-runtime")
            svc = value.args[0] if value.args else next(
                (kw.value for kw in value.keywords if kw.arg == "service_name"), None)
            if isinstance(svc, ast.Constant) and svc.value in ("bedrock-runtime", "bedrock"):
                return ("bedrock", None)
        low = last.lower()
        if "client" in low or low.startswith(("get_", "make_", "create_", "build_", "init_")):
            for key, prov in (("azure_openai", "azure"), ("azure", "azure"), ("openai", "openai"),
                              ("anthropic", "anthropic"), ("claude", "anthropic"), ("bedrock", "bedrock"),
                              ("gemini", "gemini"), ("genai", "gemini")):
                if key in low:
                    return (prov, None)
        return None

    # ---- call detection
    def _check_call(self, node: ast.Call) -> None:
        chain = dotted(node.func)
        if not chain:
            return
        kind = provider = None
        last = chain.split(".")[-1]

        # from litellm import completion as litellm_completion; partial(litellm_completion, model=...)
        lite = self._litellm_func(chain)
        if lite:
            self._record(node, chain, LITELLM_FUNCS[lite], "litellm",
                         confidence="high" if self.has_llm_import else "heuristic")
            return
        if last == "partial" and node.args:
            target = dotted(node.args[0]) or ""
            lite = self._litellm_func(target) or (
                target.split(".")[-1] if target.startswith("litellm.") and target.split(".")[-1] in LITELLM_FUNCS else None)
            if lite:
                self._record(node, f"partial({target})", LITELLM_FUNCS[lite], "litellm")
                return

        # litellm.completion(...) is unambiguous even when litellm comes through a project wrapper
        if chain.startswith("litellm.") and last in LITELLM_FUNCS and not self.has_llm_import:
            self._record(node, chain, LITELLM_FUNCS[last], "litellm", confidence="heuristic")
            return

        if not self.has_llm_import:
            has_model = any(k.arg == "model" for k in node.keywords)
            has_splat = any(k.arg is None for k in node.keywords)
            root = receiver_root(chain)
            client = self.facts.clients.get(root)
            # Client injected or built by a helper: client.chat.completions.create(model=..., messages=...)
            for suffix, k, needs in INJECTED_SUFFIXES:
                specific = suffix.startswith((".chat.completions", ".responses", ".embeddings"))
                if chain.endswith(suffix) and (
                        (has_model and (not needs or any(kw.arg in needs for kw in node.keywords)))
                        or (specific and has_splat)
                        or (client and any(kw.arg in needs for kw in node.keywords))
                        # OpenAI-shaped clients called positionally: create(messages, "model")
                        or (suffix == ".chat.completions.create" and (len(node.args) >= 1 or "messages" in
                                                                      {k.arg for k in node.keywords}))):
                    self._record(node, chain, k, client[0] if client else (self._file_hint() or "unknown"),
                                 confidence="heuristic")
                    return
            if chain.endswith((".messages.create", ".messages.stream")) and ".beta.threads." not in chain and (
                    (client and has_model and client[0] in ("anthropic", "bedrock")) or self._anthropic_shaped(node)):
                self._record(node, chain, "chat", client[0] if client else "anthropic", confidence="heuristic")
                return
            if chain.endswith((".converse", ".converse_stream")) and self._converse_shaped(node):
                self._record(node, chain, "chat", "bedrock", confidence="heuristic")
                return
            sdk = self._sdk_method(chain)
            if sdk:
                self._record(node, chain, sdk[0], sdk[1], confidence="heuristic")
                return
            kw = self._wrapper_model_kwarg(node)
            if kw:
                self._record(node, chain, "model_config", "unknown", confidence="heuristic", model_kwarg=kw)
            return

        native = self._native_call(node, chain)
        if native:
            self._record(node, chain, native[0], native[1])
            return

        for suffix, k, p in CALL_SUFFIXES:
            if chain.endswith(suffix) and chain != suffix.lstrip("."):
                kind, provider = k, p
                break
        if kind and ".threads.messages." in chain:
            return                       # Assistants: adding a message to a thread isn't a model call

        if kind:
            # Disambiguate with imports / client variables.
            root = receiver_root(chain)
            client = self.facts.clients.get(root)
            if not client and provider == "openai" and self._file_hint():
                # Client built elsewhere in the file/app, which points at one other provider
                client = (self._file_hint(), None)
            if client:
                provider = client[0]
            elif provider == "bedrock" and not self.facts.imports & {"boto3", "aioboto3", "aiobotocore", "botocore"} \
                    and not self._converse_shaped(node):
                return
            elif provider == "gemini" and "google" not in self.facts.imports:
                return
            elif provider == "anthropic" and "anthropic" not in self.facts.imports:
                # .messages.create also exists on Twilio/Slack/Plivo clients: require an Anthropic import,
                # a resolved Anthropic client variable (above), or Anthropic's own call signature.
                if ".beta.threads." in chain or not self._anthropic_shaped(node):
                    return
            elif provider == "openai" and "openai" not in self.facts.imports:
                if chain.endswith(".completions.create") and not chain.endswith(".chat.completions.create"):
                    return  # too generic without an openai import
                provider = "litellm" if "litellm" in self.facts.imports else "openai_compatible"
        elif chain.startswith("litellm.") and last in LITELLM_FUNCS:
            kind, provider = LITELLM_FUNCS[last], "litellm"
        elif chain in LITELLM_FUNCS and self.facts.from_imports.get(chain, "").startswith("litellm."):
            kind, provider = LITELLM_FUNCS[chain], "litellm"
        elif last in CONFIG_FUNCS and (self.facts.from_imports.get(last, "").startswith("langchain")
                                       or "langchain" in chain):
            kind, provider = "model_config", "unknown"
        elif last in CONFIG_CLASSES and self._is_framework_class(last):
            kind, provider = "model_config", CONFIG_CLASSES[last]
        elif last == "Agent" and self.facts.imports & {"agents", "pydantic_ai"}:
            kind, provider = "model_config", "unknown"
        elif self._wrapper_model_kwarg(node):
            # In-house or third-party wrapper: LiteLLMModel(name="gpt-4o"), Settings(llm="...")
            self._record(node, chain, "model_config", "unknown", confidence="heuristic",
                         model_kwarg=self._wrapper_model_kwarg(node))
            return
        else:
            return

        self._record(node, chain, kind, provider)

    def _endpoint_provider(self, call: ast.Call) -> str | None:
        """Provider named by a client's base URL or API-key env var (literals, constants, env defaults)."""
        strings: list[str] = []
        for kw in call.keywords:
            if kw.arg not in ("base_url", "api_base", "azure_endpoint", "endpoint", "api_key", "host", "url"):
                continue
            for n in ast.walk(kw.value):
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    strings.append(n.value)
                elif isinstance(n, ast.Name):
                    for v in self.facts.constants.get(n.id, [])[:3]:
                        strings.extend(m.value for m in ast.walk(v)
                                       if isinstance(m, ast.Constant) and isinstance(m.value, str))
                    strings.append(n.id)
                elif isinstance(n, ast.Attribute):
                    strings.append(n.attr)
        for v in strings:
            p = provider_from_url(v)
            if p:
                return p
        for v in strings:
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{2,}", v) and (v.isupper() or v.endswith(("key", "url"))):
                p = provider_from_env(v)
                if p and p not in ("openai", "anthropic"):
                    return p
        return None

    def _anthropic_shaped(self, node: ast.Call) -> bool:
        """messages.create(model=..., max_tokens=...) is Anthropic's signature; Twilio/Slack/Plivo use to/body/text.
        A **payload splat counts when the file itself talks about Anthropic/Claude."""
        kws = {k.arg for k in node.keywords}
        if "model" in kws and ("max_tokens" in kws or "messages" in kws or "system" in kws):
            return True
        low = self.source.lower()
        return None in kws and ("anthropic" in low or "claude" in low)

    def _converse_shaped(self, node: ast.Call) -> bool:
        kws = {k.arg for k in node.keywords}
        return bool(kws & {"modelId", "inferenceConfig"}) or (None in kws and "bedrock" in self.source.lower())

    def _sdk_method(self, chain: str) -> tuple[str, str] | None:
        """Vendor-specific method paths when the vendor SDK is imported: self.get_client().chat.complete(...)."""
        for provider, methods in CLIENT_METHODS.items():
            if not any(mod for mod, p in SDK_MODULES.items() if p == provider and mod in self.facts.imports):
                continue
            for m, kind in methods.items():
                if ("." in m or m in _DISTINCT_METHODS) and chain.endswith("." + m):
                    return kind, provider
        return None

    def _file_hint(self) -> str | None:
        if not hasattr(self, "_hint_cache"):
            self._hint_cache = file_provider_hint(self.source)
        return self._hint_cache

    def _native_call(self, node: ast.Call, chain: str) -> tuple[str, str] | None:
        """Vendor SDK calls that aren't OpenAI-shaped: cohere co.chat(), ollama.chat(), Generation.call(), ..."""
        root = receiver_root(chain)
        rest = chain[len(root) + 1:] if chain.startswith(root + ".") else ""
        client = self.facts.clients.get(root)
        if client and rest in CLIENT_METHODS.get(client[0], {}):
            return CLIENT_METHODS[client[0]][rest], client[0]
        spec = MODULE_FUNCS.get(chain)
        if not spec and "." in chain:
            # QWenService.dashscope.Generation.call(...): module reached through an attribute
            for k, v in MODULE_FUNCS.items():
                if "." in k and chain.endswith("." + k) and (v[2] in self.facts.imports or v[2] in chain):
                    return v[0], v[1]
        sdk = self._sdk_method(chain)
        if sdk:
            return sdk
        if not spec and "." not in chain:
            # from ollama import chat; chat(model=...)
            origin = self.facts.from_imports.get(chain, "")
            if origin in MODULE_FUNCS:
                kind, provider, _ = MODULE_FUNCS[origin]
                return kind, provider
        if spec:
            kind, provider, module = spec
            origin = (self.facts.module_aliases.get(root) or self.facts.from_imports.get(root, root)).split(".")[0]
            if origin == module or module in self.facts.imports and root == module:
                return kind, provider
        alias = self.facts.module_aliases.get(root)
        if alias and rest and f"{alias.split('.')[0]}.{rest}" in MODULE_FUNCS:
            kind, provider, _ = MODULE_FUNCS[f"{alias.split('.')[0]}.{rest}"]
            return kind, provider
        last = chain.split(".")[-1]
        if last in MODEL_CONSTRUCTORS:
            origin = self.facts.from_imports.get(last, chain)
            if root in self.facts.module_aliases:
                origin = self.facts.module_aliases[root] + "." + last
            if "generativeai" in origin or origin.startswith("google"):
                return "model_config", "gemini"
            if "vertexai" in origin:
                return "model_config", "vertex"
        return None

    def _litellm_func(self, chain: str) -> str | None:
        """Name of the litellm function an imported alias points to (litellm_completion -> completion)."""
        origin = self.facts.from_imports.get(chain, "")
        if origin.startswith("litellm.") and origin.count(".") == 1:
            fn = origin.split(".", 1)[1]
            if fn in LITELLM_FUNCS and chain != fn:
                return fn
        return None

    def _wrapper_model_kwarg(self, node: ast.Call) -> str | None:
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
        # Only constructors/factories of models, clients and settings: RestrictModel(model=...),
        # Settings(llm=...), get_llm(model=...). Not helpers like format_messages(model="o3-mini").
        if not (name[:1].isupper() or _WRAPPER_NAME_RE.search(name)):
            return None
        for kw in node.keywords:
            if kw.arg in WRAPPER_MODEL_KWARGS and isinstance(kw.value, ast.Constant) \
                    and isinstance(kw.value.value, str) and MODEL_NAME_RE.match(kw.value.value.strip()):
                return kw.arg
        return None

    def _is_framework_class(self, last: str) -> bool:
        origin = self.facts.from_imports.get(last, "")
        if origin.startswith(("langchain", "llama_index")):
            return True
        # OpenAI/Anthropic/AzureOpenAI from the vendor SDK are clients, not configs.
        return False

    # ---- record a site
    def _record(self, node: ast.Call, chain: str, kind: str, provider: str,
                confidence: str = "high", model_kwarg: str | None = None) -> None:
        if model_kwarg:
            value = next(kw.value for kw in node.keywords if kw.arg == model_kwarg)
            model, model_expr, model_source = value.value.strip(), None, "literal"
        else:
            model, model_expr, model_source = self._resolve_model(node, kind)
        if model and provider in ("unknown", "openai_compatible"):
            # "openrouter/anthropic/claude-x", "groq:llama-3" (aisuite)
            provider = provider_from_model_prefix(model) or provider
        if kind == "model_config" and provider in ("openai", "unknown", "openai_compatible"):
            # ChatOpenAI(base_url="https://openrouter.ai/api/v1"), OpenAILike(api_base=...)
            ep = self._endpoint_provider(node)
            if ep and ep not in ("openai", "anthropic"):
                provider = ep
        if provider == "unknown" and confidence == "heuristic" and kind == "model_config":
            # In-house wrapper: ChatGroq(model=...), DeepSeekLLM(...) name their provider
            provider = provider_from_class_name(chain.split(".")[-1]) or provider
        if provider in ("unknown", "openai_compatible"):
            provider = self._file_hint() or provider
        if provider == "unknown" and model:
            provider = maker_of(model) or provider

        params = self._params(node)
        prompt = self._prompt(node)
        func = self._enclosing_function()
        loop_kind = self._loop_kind(node)
        in_loop = loop_kind is not None
        has_retry, unbounded = self._retry_info()

        hints: list[str] = []
        text = (prompt["system"] or "") + " " + (prompt["user"] or "")
        for name, pat in HINT_PATTERNS:
            if pat.search(text) or (func and pat.search(func.name.replace("_", " "))):
                hints.append(name)
        if any(k in params for k in ("response_format", "text_format")) or chain.endswith(".parse"):
            hints.append("structured_output")
        if "tools" in params and "tool_choice" in params:
            hints.append("forced_tool")

        site_id = hashlib.sha1(f"{self.rel_path}:{node.lineno}:{kind}".encode()).hexdigest()[:12]
        site = CallSite(
            id=site_id, path=self.rel_path, line=node.lineno, kind=kind, provider=provider,
            sdk_call=chain, function=func.name if func else None, model=model,
            model_expr=model_expr, model_source=model_source, params=params,
            in_loop=in_loop, loop_kind=loop_kind,
            scope="example" if EXAMPLE_PATH_RE.search(self.rel_path) else "app", is_async=self._is_await(node) or bool(func and isinstance(func, ast.AsyncFunctionDef)),
            retry_unbounded=unbounded, has_retry=has_retry,
            prompt_fingerprint=prompt["fingerprint"], system_prompt_tokens=prompt["system_tokens"],
            prompt_static_tokens=prompt["static_tokens"], prompt_dynamic_prefix=prompt["dynamic_prefix"],
            prompt_preview=prompt["preview"], task_hints=sorted(set(hints)),
            uses_cache_control=prompt["cache_control"] or "prompt_cache_key" in params,
            is_test=self.is_test, confidence=confidence,
        )
        self.sites.append(site)

    # ---- model resolution
    def _resolve_model(self, node: ast.Call, kind: str) -> tuple[str | None, str | None, str]:
        expr = None
        for kw in node.keywords:
            if kw.arg in MODEL_KWARGS:
                expr = kw.value
                break
        if expr is None and node.args:
            chain = dotted(node.func) or ""
            last = chain.split(".")[-1]
            if last in LITELLM_FUNCS or last in CONFIG_FUNCS or last == "Agent" or chain.endswith(POSITIONAL_MODEL):
                expr = node.args[0]
        if expr is None:
            # Model may come from the framework object, the batch file, or a default.
            return None, None, "missing"
        return self._resolve_str(expr)

    def _resolve_str(self, expr: ast.AST, depth: int = 0) -> tuple[str | None, str | None, str]:
        src = _unparse(expr) if depth == 0 else None      # only the outermost expression is reported
        if depth > 4:
            return None, src, "unresolved"
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return expr.value, None, "literal"
        if isinstance(expr, ast.Name):
            for scope in self._local_scopes():
                if expr.id in scope:
                    m, _, s = self._resolve_str(scope[expr.id], depth + 1)
                    if m:
                        return m, src, "constant" if s == "literal" else s
            for scope in reversed(self._func_params):
                if expr.id in scope:
                    m, _, s = self._resolve_str(scope[expr.id], depth + 1)
                    if m:
                        return m, src, "param_default"
            values = self.facts.constants.get(expr.id, [])
            # Each file-level name is resolved once (memo + cycle guard): a name assigned hundreds
            # of times in a big file would otherwise branch exponentially.
            memo = self._const_memo.get(expr.id)
            if memo is None:
                self._const_memo[expr.id] = _IN_PROGRESS
                pairs = {(m, s) for m, _, s in (self._resolve_str(v, depth + 1) for v in values[:50]) if m}
                memo = (None, None)
                if len({m for m, _ in pairs}) == 1:
                    m = next(iter(pairs))[0]
                    memo = (m, "env_default" if any(s == "env_default" for _, s in pairs) else "constant")
                self._const_memo[expr.id] = memo
            if memo is not _IN_PROGRESS and memo[0]:
                return memo[0], src, memo[1]
            ext = self._external(expr.id) if not values else None
            if ext is not None:
                m, _, s = self._resolve_str(ext, depth + 1)
                if m:
                    return m, src, "env_default" if s == "env_default" else "constant"
            return None, src, "unresolved"
        if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name) and expr.value.id != "self":
            ext = self._external_attr(expr.value.id, expr.attr)
            if ext is not None:
                m, _, s = self._resolve_str(ext, depth + 1)
                if m:
                    return m, src, "env_default" if s == "env_default" else "constant"
        if isinstance(expr, ast.Attribute):
            key = f"self.{expr.attr}" if isinstance(expr.value, ast.Name) and expr.value.id == "self" else expr.attr
            memo = self._const_memo.get("." + key)
            if memo is None:
                self._const_memo["." + key] = _IN_PROGRESS
                values = self.facts.constants.get(key, []) or self.facts.constants.get(expr.attr, [])
                resolved = {self._resolve_str(v, depth + 1)[0] for v in values[:50]} - {None}
                memo = (resolved.pop() if len(resolved) == 1 else None, None)
                self._const_memo["." + key] = memo
            if memo is not _IN_PROGRESS and memo[0]:
                return memo[0], src, "constant"
            return None, src, "unresolved"
        if isinstance(expr, ast.Call):
            fn = dotted(expr.func) or ""
            if fn in ("os.getenv", "os.environ.get", "getenv", "environ.get") and len(expr.args) >= 2:
                m, _, _ = self._resolve_str(expr.args[1], depth + 1)
                if m:
                    return m, src, "env_default"
            for kw in expr.keywords:
                if fn in ("os.getenv", "os.environ.get") and kw.arg == "default":
                    m, _, _ = self._resolve_str(kw.value, depth + 1)
                    if m:
                        return m, src, "env_default"
        if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.Or):
            for v in reversed(expr.values):
                m, _, s = self._resolve_str(v, depth + 1)
                if m:
                    return m, src, "env_default" if s in ("literal", "constant") else s
        return None, src, "unresolved"

    # ---- params
    def _params(self, node: ast.Call) -> dict:
        out: dict = {}
        for kw in node.keywords:
            if kw.arg is None:
                out["_kwargs_passthrough"] = True   # **kwargs: params may be set by the caller
            elif kw.arg in PARAM_KWARGS:
                v = kw.value
                if isinstance(v, ast.Constant):
                    out[kw.arg] = v.value
                else:
                    out[kw.arg] = "<expr>"
        return out

    # ---- prompts
    def _prompt(self, node: ast.Call) -> dict:
        system = Template()
        user = Template()
        got_system = got_user = False
        cache_control = any(isinstance(n, ast.Constant) and n.value == "cache_control" for n in ast.walk(node))

        for kw in node.keywords:
            if kw.arg not in PROMPT_KWARGS:
                continue
            value = self._deref(kw.value)
            if kw.arg in ("system", "instructions"):
                system.merge(self._template(value))
                got_system = True
                cache_control = cache_control or self._has_cache_control(value)
            elif kw.arg in ("messages", "input", "contents") and isinstance(value, (ast.List, ast.Tuple)):
                cache_control = cache_control or self._has_cache_control(value)
                for elt in value.elts:
                    elt = self._deref(elt)
                    role, content = self._role_content(elt)
                    if content is None:
                        continue
                    t = self._template(self._deref(content))
                    if role in ("system", "developer"):
                        system.merge(t)
                        got_system = True
                    else:
                        user.merge(t)
                        got_user = True
            elif kw.arg in ("input", "prompt", "contents"):
                user.merge(self._template(value))
                got_user = True

        combined = Template()
        combined.merge(system)
        combined.merge(user)
        static_chars = combined.static_chars
        fingerprint = None
        if static_chars >= 20:
            norm = " ".join(combined.static.split()).lower()
            fingerprint = hashlib.sha1(norm.encode()).hexdigest()[:12]
        sys_chars = system.static_chars
        dynamic_prefix = bool(
            got_system and sys_chars >= 400 and system.first_dynamic_at is not None
            and system.first_dynamic_at < 0.5 * len(system.static)
        )
        preview_src = system.static if got_system and system.static.strip() else user.static
        return {
            "system": system.static if got_system else None,
            "user": user.static if got_user else None,
            "system_tokens": est_tokens(sys_chars) if got_system else None,
            "static_tokens": est_tokens(static_chars) if (got_system or got_user) else None,
            "dynamic_prefix": dynamic_prefix,
            "fingerprint": fingerprint,
            "preview": redact(preview_src, 160) if preview_src and preview_src.strip() else None,
            "cache_control": cache_control,
        }

    def _deref(self, node: ast.AST) -> ast.AST:
        """Follow a Name to its local (or single module-level) assignment."""
        seen = 0
        while isinstance(node, ast.Name) and seen < 4:
            seen += 1
            nxt = None
            for scope in self._local_scopes():
                if node.id in scope:
                    nxt = scope[node.id]
                    break
            if nxt is None:
                vals = self.facts.constants.get(node.id, [])
                if len(vals) == 1:
                    nxt = vals[0]
            if nxt is None:
                nxt = self._external(node.id)
            if nxt is None:
                break
            node = nxt
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            # prompts.SYSTEM_PROMPT where `prompts` is an imported module
            ext = self._external_attr(node.value.id, node.attr)
            if ext is not None:
                return self._deref(ext) if isinstance(ext, ast.Name) else ext
        return node

    def _external(self, name: str) -> ast.AST | None:
        """Value of a name imported from another project module (filled in on a second pass)."""
        origin = self.facts.from_imports.get(name)
        if not origin:
            return None
        if origin in self.external:
            return self.external[origin]
        self.wanted.add(origin)
        return None

    def _external_attr(self, alias: str, attr: str) -> ast.AST | None:
        module = self.facts.module_aliases.get(alias) or self.facts.from_imports.get(alias)
        if not module:
            return None
        key = f"{module}.{attr}"
        if key in self.external:
            return self.external[key]
        self.wanted.add(key)
        return None

    def _is_local_name(self, name: str) -> bool:  # noqa: D401
        """True if the name is a function local/param (dynamic at runtime)."""
        return any(name in s for s in self._local_scopes()) or any(name in s for s in self._func_params)

    def _has_cache_control(self, node: ast.AST) -> bool:
        return any(isinstance(n, ast.Constant) and n.value == "cache_control" for n in ast.walk(node))

    def _role_content(self, elt: ast.AST) -> tuple[str | None, ast.AST | None]:
        if isinstance(elt, ast.Dict):
            role = content = None
            for k, v in zip(elt.keys, elt.values):
                if isinstance(k, ast.Constant):
                    if k.value == "role" and isinstance(v, ast.Constant):
                        role = v.value
                    elif k.value in ("content", "text"):
                        content = v
            return role, content
        if isinstance(elt, ast.Call):
            name = (dotted(elt.func) or "").split(".")[-1]
            role = {"SystemMessage": "system", "HumanMessage": "user", "AIMessage": "assistant"}.get(name)
            if role:
                content = elt.args[0] if elt.args else next(
                    (kw.value for kw in elt.keywords if kw.arg == "content"), None)
                return role, content
        if isinstance(elt, ast.Tuple) and len(elt.elts) == 2 and isinstance(elt.elts[0], ast.Constant):
            return str(elt.elts[0].value), elt.elts[1]
        return None, None

    def _template(self, node: ast.AST, depth: int = 0) -> Template:
        t = Template()
        if depth > 5 or node is None:
            t.add_slot()
            t.resolved = False
            return t
        node = self._deref(node)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            t.add_static(node.value)
        elif isinstance(node, ast.JoinedStr):
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    t.add_static(v.value)
                elif isinstance(v, ast.FormattedValue) and isinstance(
                        self._deref(v.value), (ast.Constant, ast.JoinedStr)) and isinstance(v.value, ast.Name) \
                        and not self._is_local_name(v.value.id):
                    # f"...{SYSTEM_PROMPT}..." where SYSTEM_PROMPT is a module constant: inline it
                    t.merge(self._template(v.value, depth + 1))
                else:
                    t.add_slot()
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            t.merge(self._template(node.left, depth + 1))
            t.merge(self._template(node.right, depth + 1))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            t.merge(self._template(node.left, depth + 1))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
            base = self._template(node.func.value, depth + 1)
            base.static = re.sub(r"\{[^{}]*\}", "{}", base.static)
            if base.first_dynamic_at is None and "{}" in base.static:
                base.first_dynamic_at = base.static.index("{}")
            t.merge(base)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join":
            t.add_slot()
        elif isinstance(node, (ast.List, ast.Tuple)):
            # content blocks: [{"type": "text", "text": ...}]
            for elt in node.elts:
                elt = self._deref(elt)
                _, content = self._role_content(elt)
                if content is not None:
                    t.merge(self._template(content, depth + 1))
                elif isinstance(elt, (ast.Constant, ast.JoinedStr)):
                    t.merge(self._template(elt, depth + 1))
        else:
            t.add_slot()
            t.resolved = False
        return t

    # ---- context
    def _enclosing_function(self):
        for n in reversed(self._stack):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return n
        return None

    def _loop_kind(self, call: ast.AST | None = None) -> str | None:
        """Innermost loop around the call within the same function: for | comprehension | while | agent.

        "agent" is a for loop whose body feeds each answer into the next call (an agent or tool loop,
        `for _ in range(max_turns): ... messages.append(...)`): those calls can't be batched."""
        for n in reversed(self._stack):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                return None
            if isinstance(n, (ast.For, ast.AsyncFor)):
                return "agent" if call is not None and _feeds_back(n, call) else "for"
            if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                return "comprehension"
            if isinstance(n, ast.While):
                return "while"
        return None

    def _is_await(self, node: ast.Call) -> bool:
        return bool(self._stack) and isinstance(self._stack[-1], ast.Await)

    def _retry_info(self) -> tuple[bool, bool]:
        func = self._enclosing_function()
        if not func:
            return False, False
        has = unbounded = False
        for dec in func.decorator_list:
            name = dotted(dec.func if isinstance(dec, ast.Call) else dec) or ""
            last = name.split(".")[-1]
            if last == "retry" or name.endswith("tenacity.retry"):
                has = True
                if not isinstance(dec, ast.Call):
                    unbounded = True  # bare @retry retries forever
                elif not any(kw.arg == "stop" for kw in dec.keywords):
                    unbounded = True
            elif last in ("on_exception", "on_predicate") and "backoff" in name:
                has = True
                if isinstance(dec, ast.Call) and not any(kw.arg in ("max_tries", "max_time") for kw in dec.keywords):
                    unbounded = True
        return has, unbounded

    # ---- lazily computed function locals (only for functions that contain a detected call)
    def _local_scopes(self):
        """Innermost-first local assignment maps for the enclosing functions."""
        for fn in reversed(self._func_nodes):
            key = id(fn)
            local = self._locals_cache.get(key)
            if local is None:
                local = {}
                for sub in ast.walk(fn):
                    if isinstance(sub, ast.Assign) and len(sub.targets) == 1 \
                            and isinstance(sub.targets[0], ast.Name):
                        local[sub.targets[0].id] = sub.value
                self._locals_cache[key] = local
            yield local


_HTTP_PATH_RE = re.compile(
    r"(/v\d+(?:beta\d*|alpha)?)?(/chat/completions|/messages|/responses|/embeddings|"
    r":generatecontent|:streamgeneratecontent|/services/aigc/text-generation/generation|"
    r"/services/aigc/multimodal-generation/generation|/services/embeddings/text-embedding/text-embedding|"
    r"/api/chat|/api/generate|/v2/chat|/v2/embed)(?=$|[?#\"'/])")
_PROVIDER_HOSTS = (("api.anthropic.com", "anthropic"), ("api.openai.com", "openai"),
                   ("openai.azure.com", "azure"), ("generativelanguage.googleapis.com", "gemini"),
                   ("aiplatform.googleapis.com", "gemini"), ("bedrock-runtime", "bedrock"),
                   ("api.groq.com", "openai_compatible"), ("api.together.xyz", "openai_compatible"),
                   ("api.mistral.ai", "mistral"), ("api.deepseek.com", "deepseek"),
                   ("openrouter.ai", "openai_compatible"), ("api.x.ai", "openai_compatible"))


def http_endpoint(value: str, dynamic_host: bool = False) -> tuple[str, str, str] | None:
    """Recognise a provider REST endpoint in a URL/path string: (kind, provider, path) or None.

    Server routes that merely *serve* an OpenAI-compatible API ("/api/chat/completions") are
    excluded, as are prose strings (docstrings) that mention an endpoint.
    """
    v = value.strip()
    if not v or len(v) > 200 or "\n" in v or " " in v:
        return None
    low = v.lower()
    host = provider_from_url(low) if ("://" in low or low.startswith(("//", "api.")) or
                                      re.match(r"^[a-z0-9.-]+\.[a-z]{2,}(:\d+)?/", low)) else None
    m = _HTTP_PATH_RE.search(low)
    if not m:
        return None
    version, path = m.group(1), m.group(2)
    before = low[:m.start()]
    if before.endswith("/api") and not host and "://" not in before and not dynamic_host:
        return None                     # own server route, e.g. /api/chat/completions (relative path)
    if path in ("/messages", "/responses") and not (version or host):
        return None                     # too generic without /v1 or a provider host
    if path in ("/api/chat", "/api/generate", "/v2/chat", "/v2/embed") and not host:
        return None                     # Ollama / Cohere paths only count with their host
    if path == "/embeddings" and not (version or host or low.endswith("/embeddings") and len(low) < 40):
        return None
    kind = "embedding" if path in ("/embeddings", "/v2/embed") or "text-embedding" in path else "chat"
    provider = host or ("anthropic" if path == "/messages" else
                        "gemini" if "generatecontent" in path else
                        "dashscope" if "/services/" in path else "openai_compatible")
    return kind, provider, (version or "") + path


def _in_decorator(stack: list) -> bool:
    for i, anc in enumerate(stack[:-1]):
        if isinstance(anc, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nxt = stack[i + 1]
            if any(nxt is d for d in anc.decorator_list):
                return True
    return False


_WRAPPER_NAME_RE = re.compile(r"llm|model|chat|client|agent|config|setting|completion|embed", re.I)

_CANDIDATE_NAMES: set[str] = set()


def _init_candidate_names() -> None:
    _CANDIDATE_NAMES.update(suf.rsplit(".", 1)[-1] for suf, _, _ in CALL_SUFFIXES)
    _CANDIDATE_NAMES.update(LITELLM_FUNCS)
    _CANDIDATE_NAMES.update(CONFIG_CLASSES)
    _CANDIDATE_NAMES.update(CONFIG_FUNCS)
    _CANDIDATE_NAMES.add("Agent")
    for methods in CLIENT_METHODS.values():
        _CANDIDATE_NAMES.update(m.rsplit(".", 1)[-1] for m in methods)
    _CANDIDATE_NAMES.update(c.rsplit(".", 1)[-1] for c in MODULE_FUNCS)
    _CANDIDATE_NAMES.update(MODEL_CONSTRUCTORS)


def _is_candidate(node: ast.Call) -> bool:
    """Cheap filter: could this call be an LLM call or a model configuration?"""
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
    if name in _CANDIDATE_NAMES:
        return True
    for kw in node.keywords:
        if kw.arg in WRAPPER_MODEL_KWARGS and isinstance(kw.value, ast.Constant) \
                and isinstance(kw.value.value, str):
            return True
    return False


def _params_of(fn) -> dict[str, ast.AST]:
    params: dict[str, ast.AST] = {}
    args = fn.args
    positional = args.posonlyargs + args.args
    for a, d in zip(positional[len(positional) - len(args.defaults):], args.defaults):
        params[a.arg] = d
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        if d is not None:
            params[a.arg] = d
    return params


def module_constants(source: str) -> dict[str, ast.AST]:
    """Top-level NAME = <str-ish> assignments of a module (for cross-file prompt/model resolution).

    Names inside the values that refer to other constants of the same module are inlined, so
    SYSTEM = BASE + RULES resolves when BASE and RULES are defined in that module.
    """
    try:
        tree = quiet_parse(source[1:] if source.startswith("\ufeff") else source, "<module>")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return {}
    consts: dict[str, ast.AST] = {}
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
            node, ast.AnnAssign) and node.value is not None else []
        for t in targets:
            if isinstance(t, ast.Name) and isinstance(node.value, (
                    ast.Constant, ast.JoinedStr, ast.BinOp, ast.Call, ast.Name)):
                consts[t.id] = node.value

    class _Inline(ast.NodeTransformer):
        def __init__(self, depth=0):
            self.depth = depth

        def visit_Name(self, n):
            v = consts.get(n.id)
            if v is not None and self.depth < 3 and not isinstance(v, ast.Call):
                return _Inline(self.depth + 1).visit(ast.fix_missing_locations(ast.parse(
                    ast.unparse(v), mode="eval").body))
            return n

    out = {}
    for name, v in consts.items():
        if isinstance(v, ast.Constant) and not isinstance(v.value, str):
            continue
        try:
            out[name] = _Inline().visit(ast.parse(ast.unparse(v), mode="eval").body)
        except Exception:  # pragma: no cover
            out[name] = v
    return out


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)[:120]
    except Exception:  # pragma: no cover
        return "<expr>"


_init_candidate_names()


_MUTATORS = ("append", "extend", "insert", "update", "add", "appendleft")


def _feeds_back(loop: ast.For | ast.AsyncFor, call: ast.AST) -> bool:
    """True when the loop body grows a value the call reads (messages.append(...), history += ...),
    i.e. each call depends on earlier iterations. A name rebuilt by plain assignment in the body is
    per-item data, not a feedback channel."""
    targets = {t.id for t in ast.walk(loop.target) if isinstance(t, ast.Name)}
    used = {n.id for n in ast.walk(call) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - targets
    if not used:
        return False
    body = [x for stmt in loop.body for x in ast.walk(stmt)]
    rebuilt = {t.id for x in body if isinstance(x, (ast.Assign, ast.AnnAssign))
               for tgt in (x.targets if isinstance(x, ast.Assign) else [x.target])
               for t in ast.walk(tgt) if isinstance(t, ast.Name)}
    for x in body:
        if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr in _MUTATORS \
                and isinstance(x.func.value, ast.Name) and x.func.value.id in used - rebuilt:
            return True
        if isinstance(x, ast.AugAssign) and isinstance(x.target, ast.Name) and x.target.id in used - rebuilt:
            return True
    return False
