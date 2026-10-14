"""Python AST scanner: finds LLM call sites, resolves models, and analyses prompts.

Pure stdlib. Never executes the scanned code.
"""
from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field

from ..redact import redact
from ..schema import CallSite

# --------------------------------------------------------------------------- signatures

# (suffix of dotted call chain, kind, default provider)
CALL_SUFFIXES: list[tuple[str, str, str]] = [
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
    "ChatGroq": "openai_compatible", "ChatMistralAI": "mistral",
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
                    "groq", "together", "cohere"}

WRAPPER_MODEL_KWARGS = ("model", "model_name", "name", "llm", "llm_model", "summary_llm",
                        "agent_llm", "embedding", "embedding_model", "fast_llm", "smart_llm",
                        "strategic_llm", "judge_model")
MODEL_KWARGS = ("model", "modelId", "model_id", "model_name", "deployment_name", "azure_deployment")
PARAM_KWARGS = ("max_tokens", "max_completion_tokens", "max_output_tokens", "temperature",
                "stream", "response_format", "text_format", "tools", "tool_choice", "n",
                "reasoning_effort", "reasoning", "service_tier", "seed", "top_p",
                "prompt_cache_key", "thinking", "config", "generation_config")
PROMPT_KWARGS = ("messages", "system", "instructions", "input", "prompt", "contents")

MODEL_NAME_RE = re.compile(
    r"^(gpt-[0-9][\w.\-]*|chatgpt-[\w.\-]+|o[1-9](?:-[\w.\-]+)?|claude-[\w.\-:]+|"
    r"gemini-[\w.\-]+|text-embedding-[\w.\-]+|(?:us|eu|apac|global)?\.?anthropic\.claude[\w.\-:]+|"
    r"(?:meta|amazon|mistral|cohere)\.[\w.\-:]+|(?:openai|anthropic|gemini|azure|bedrock|groq|"
    r"together_ai|mistral|deepseek)/[\w.\-:/]+|mistral-(?:large|medium|small|tiny)[\w.\-]*|"
    r"deepseek-(?:chat|reasoner|r1|v3)[\w.\-]*|llama-?\d[\w.\-]*)$",
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
TEST_PATH_RE = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")


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
    constants: dict[str, list[ast.AST]] = field(default_factory=dict)  # NAME -> value nodes
    clients: dict[str, tuple[str, str | None]] = field(default_factory=dict)  # var -> (provider, base_url)
    retry_funcs: set[str] = field(default_factory=set)


# --------------------------------------------------------------------------- scanner

class PythonFileScanner:
    def __init__(self, rel_path: str, source: str):
        self.rel_path = rel_path
        self.source = source
        self.sites: list[CallSite] = []
        self.model_refs: list[tuple[int, str]] = []
        self.facts = FileFacts()
        self._stack: list[ast.AST] = []
        self._func_nodes: list[ast.AST] = []
        self._locals_cache: dict[int, dict[str, ast.AST]] = {}
        self.parse_error = False
        self._func_params: list[dict[str, ast.AST]] = []
        self.is_test = bool(TEST_PATH_RE.search(rel_path))
        self.has_llm_import = False

    # ---- entry
    def scan(self) -> list[CallSite]:
        try:
            tree = ast.parse(self.source)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            self.parse_error = True
            return []
        # Single traversal: collect file facts and remember plausible call nodes with their
        # ancestors; resolve those candidates afterwards, once every fact in the file is known.
        candidates = self._traverse(tree)
        # Without an LLM SDK import only wrapper configs (Settings(llm="gpt-4o")) are recorded.
        self.has_llm_import = bool(self.facts.imports & LLM_IMPORT_ROOTS)
        for call, ancestors in candidates:
            self._stack = list(ancestors)
            self._func_nodes = [n for n in ancestors if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            self._func_params = [_params_of(fn) for fn in self._func_nodes]
            self._check_call(call)
        used = {s.line for s in self.sites}
        self.model_refs = [(ln, v) for ln, v in self._ref_candidates if ln not in used]
        return self.sites

    # ---- single pass: facts + candidate calls
    def _traverse(self, tree: ast.AST) -> list[tuple[ast.Call, tuple]]:
        """Iterative DFS in source order (no recursion limit issues on deep expressions)."""
        self._ref_candidates: list[tuple[int, str]] = []
        self._seen_refs: set = set()
        candidates: list[tuple[ast.Call, tuple]] = []
        stack: list[ast.AST] = []
        todo: list[tuple[ast.AST, bool]] = [(tree, False)]
        iter_children = ast.iter_child_nodes
        while todo:
            node, exiting = todo.pop()
            if exiting:
                stack.pop()
                continue
            self._fact(node)
            if isinstance(node, ast.Call) and _is_candidate(node):
                candidates.append((node, tuple(stack)))
            stack.append(node)
            todo.append((node, True))
            children = list(iter_children(node))
            for c in reversed(children):
                todo.append((c, False))
        return candidates

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
            elif isinstance(node, ast.ImportFrom) and node.module:
                f.imports.add(node.module.split(".")[0])
                for a in node.names:
                    f.from_imports[a.asname or a.name] = f"{node.module}.{a.name}"
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
        if not isinstance(value, ast.Call):
            return None
        name = dotted(value.func) or ""
        last = name.split(".")[-1]
        base_url = None
        for kw in value.keywords:
            if kw.arg == "base_url" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                base_url = kw.value.value
        if last in CLIENT_CLASSES:
            origin = self.facts.from_imports.get(last, name)
            if last == "Client":
                if "genai" in origin or "genai" in name:
                    return ("gemini", None)
                return None
            provider = CLIENT_CLASSES[last]
            if provider in ("openai",) and (base_url or any(kw.arg == "base_url" for kw in value.keywords)):
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

        # litellm.completion(...) is unambiguous even when litellm comes through a project wrapper
        if chain.startswith("litellm.") and last in LITELLM_FUNCS and not self.has_llm_import:
            self._record(node, chain, LITELLM_FUNCS[last], "litellm", confidence="heuristic")
            return

        if not self.has_llm_import:
            has_model = any(k.arg == "model" for k in node.keywords)
            root = receiver_root(chain)
            client = self.facts.clients.get(root)
            # Client injected or built by a helper: client.chat.completions.create(model=..., messages=...)
            for suffix, k, needs in INJECTED_SUFFIXES:
                if chain.endswith(suffix) and has_model and (
                        not needs or any(kw.arg in needs for kw in node.keywords)):
                    self._record(node, chain, k, client[0] if client else "unknown", confidence="heuristic")
                    return
            if client and chain.endswith((".messages.create", ".messages.stream")) and has_model \
                    and client[0] in ("anthropic", "bedrock"):
                self._record(node, chain, "chat", client[0], confidence="heuristic")
                return
            kw = self._wrapper_model_kwarg(node)
            if kw:
                self._record(node, chain, "model_config", "unknown", confidence="heuristic", model_kwarg=kw)
            return

        for suffix, k, p in CALL_SUFFIXES:
            if chain.endswith(suffix) and chain != suffix.lstrip("."):
                kind, provider = k, p
                break

        if kind:
            # Disambiguate with imports / client variables.
            root = receiver_root(chain)
            client = self.facts.clients.get(root)
            if client:
                provider = client[0]
            elif provider == "bedrock" and not self.facts.imports & {"boto3", "aioboto3", "aiobotocore", "botocore"}:
                return
            elif provider == "gemini" and "google" not in self.facts.imports:
                return
            elif provider == "anthropic" and "anthropic" not in self.facts.imports:
                # .messages.create also exists on Twilio/Slack clients: require an Anthropic import
                # (or a resolved Anthropic client variable, handled above).
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

    def _wrapper_model_kwarg(self, node: ast.Call) -> str | None:
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
        if model and provider in ("unknown", "litellm", "openai_compatible") and "/" in model:
            prefix = model.split("/")[0].lower()
            if prefix in ("openai", "anthropic", "azure", "bedrock", "gemini", "vertex_ai", "groq",
                          "mistral", "together_ai", "deepseek", "openrouter"):
                provider = provider if provider == "litellm" else prefix
        if provider == "unknown" and model:
            low = model.lower()
            provider = ("anthropic" if low.startswith("claude") else
                        "openai" if low.startswith(("gpt", "o1", "o3", "o4", "text-embedding")) else
                        "gemini" if low.startswith("gemini") else provider)

        params = self._params(node)
        prompt = self._prompt(node)
        func = self._enclosing_function()
        loop_kind = self._loop_kind()
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
            if last in LITELLM_FUNCS or last in CONFIG_FUNCS or last == "Agent":
                expr = node.args[0]
        if expr is None:
            # Model may come from the framework object, the batch file, or a default.
            return None, None, "missing"
        return self._resolve_str(expr)

    def _resolve_str(self, expr: ast.AST, depth: int = 0) -> tuple[str | None, str | None, str]:
        src = _unparse(expr)
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
            pairs = {(m, s) for m, _, s in (self._resolve_str(v, depth + 1) for v in values) if m}
            if len({m for m, _ in pairs}) == 1:
                m = next(iter(pairs))[0]
                env = any(s == "env_default" for _, s in pairs)
                return m, src, "env_default" if env else "constant"
            return None, src, "unresolved"
        if isinstance(expr, ast.Attribute):
            key = f"self.{expr.attr}" if isinstance(expr.value, ast.Name) and expr.value.id == "self" else expr.attr
            values = self.facts.constants.get(key, []) or self.facts.constants.get(expr.attr, [])
            resolved = {self._resolve_str(v, depth + 1)[0] for v in values} - {None}
            if len(resolved) == 1:
                return resolved.pop(), src, "constant"
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
                break
            node = nxt
        return node

    def _is_local_name(self, name: str) -> bool:
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

    def _loop_kind(self) -> str | None:
        """Innermost loop around the call within the same function: for | comprehension | while."""
        for n in reversed(self._stack):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                return None
            if isinstance(n, (ast.For, ast.AsyncFor)):
                return "for"
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


_CANDIDATE_NAMES: set[str] = set()


def _init_candidate_names() -> None:
    _CANDIDATE_NAMES.update(suf.rsplit(".", 1)[-1] for suf, _, _ in CALL_SUFFIXES)
    _CANDIDATE_NAMES.update(LITELLM_FUNCS)
    _CANDIDATE_NAMES.update(CONFIG_CLASSES)
    _CANDIDATE_NAMES.update(CONFIG_FUNCS)
    _CANDIDATE_NAMES.add("Agent")


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


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)[:120]
    except Exception:  # pragma: no cover
        return "<expr>"


_init_candidate_names()
