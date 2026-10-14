"""Where an LLM call goes: provider ids, API hosts, API-key env vars and model-family makers.

One table for every scanner (Python AST, other languages, raw HTTP), so supporting a new
provider is a one-line change here.
"""
from __future__ import annotations

import re

# (host substring, provider id). Order matters: more specific hosts first.
HOSTS: list[tuple[str, str]] = [
    ("openai.azure.com", "azure"), ("cognitiveservices.azure.com", "azure"),
    ("services.ai.azure.com", "azure"), ("models.inference.ai.azure.com", "azure"),
    ("models.github.ai", "azure"),
    ("api.openai.com", "openai"),
    ("api.anthropic.com", "anthropic"),
    ("generativelanguage.googleapis.com", "gemini"), ("aiplatform.googleapis.com", "vertex"),
    ("bedrock-runtime", "bedrock"), ("bedrock-mantle", "bedrock"),
    ("openrouter.ai", "openrouter"),
    ("dashscope", "dashscope"), ("aliyuncs.com", "dashscope"),
    ("api.deepseek.com", "deepseek"),
    ("api.groq.com", "groq"),
    ("api.together.xyz", "together"), ("api.together.ai", "together"),
    ("api.fireworks.ai", "fireworks"),
    ("api.mistral.ai", "mistral"),
    ("api.cohere.com", "cohere"), ("api.cohere.ai", "cohere"),
    ("api.x.ai", "xai"),
    ("api.perplexity.ai", "perplexity"),
    ("api.moonshot.cn", "moonshot"), ("api.moonshot.ai", "moonshot"),
    ("open.bigmodel.cn", "zhipu"), ("api.z.ai", "zhipu"),
    ("api.minimax", "minimax"), ("minimaxi.com", "minimax"),
    ("volces.com", "volcengine"), ("volcengineapi.com", "volcengine"),
    ("qianfan.baidubce.com", "qianfan"), ("aip.baidubce.com", "qianfan"),
    ("hunyuan.tencentcloudapi.com", "tencent"), ("api.hunyuan.cloud.tencent.com", "tencent"),
    ("api.lingyiwanwu.com", "01ai"), ("api.baichuan-ai.com", "baichuan"),
    ("spark-api-open.xf-yun.com", "iflytek"),
    ("api.siliconflow", "siliconflow"),
    ("api.deepinfra.com", "deepinfra"),
    ("api.novita.ai", "novita"),
    ("api.hyperbolic.xyz", "hyperbolic"),
    ("api.studio.nebius", "nebius"), ("api.tokenfactory.nebius", "nebius"),
    ("integrate.api.nvidia.com", "nvidia"),
    ("api.cerebras.ai", "cerebras"),
    ("api.sambanova.ai", "sambanova"),
    ("router.huggingface.co", "huggingface"), ("api-inference.huggingface.co", "huggingface"),
    ("api.replicate.com", "replicate"),
    ("api.cloudflare.com", "cloudflare"), ("gateway.ai.cloudflare.com", "cloudflare"),
    ("api.voyageai.com", "voyage"),
    ("api.ai21.com", "ai21"),
    ("api.lambdalabs.com", "lambda"), ("api.lambda.ai", "lambda"),
    ("api.friendli.ai", "friendli"),
    ("inference.baseten.co", "baseten"),
    ("api.anyscale.com", "anyscale"),
    ("ai-gateway.vercel.sh", "vercel_ai_gateway"),
    ("api.llama.com", "meta"), ("inference.do-ai.run", "digitalocean"), ("api.venice.ai", "venice"),
    ("api.githubcopilot.com", "github_copilot"), ("api.reka.ai", "reka"), ("api.302.ai", "302ai"),
    ("api.inceptionlabs.ai", "inception"), ("api.writer.com", "writer"), ("api.upstage.ai", "upstage"),
    ("api.stepfun.com", "stepfun"), ("api.baichuan-ai.com", "baichuan"),
    ("api.portkey.ai", "portkey"),
    ("cloud.databricks.com", "databricks"), ("azuredatabricks.net", "databricks"),
    ("ml.cloud.ibm.com", "watsonx"),
    ("api-inference.modelscope", "modelscope"), ("modelscope.cn/v1", "modelscope"),
    (":11434", "ollama"),
    ("localhost:1234", "lmstudio"),
]

# API-key / base-URL environment variables that reveal the provider
ENV_HINTS: list[tuple[str, str]] = [
    ("AZURE_OPENAI", "azure"), ("AZURE_AI", "azure"), ("OPENROUTER", "openrouter"),
    ("DASHSCOPE", "dashscope"), ("ALIBABA", "dashscope"), ("QWEN", "dashscope"),
    ("DEEPSEEK", "deepseek"), ("GROQ", "groq"), ("TOGETHER", "together"),
    ("FIREWORKS", "fireworks"), ("MISTRAL", "mistral"), ("COHERE", "cohere"), ("CO_API_KEY", "cohere"),
    ("XAI_", "xai"), ("GROK", "xai"), ("PERPLEXITY", "perplexity"), ("PPLX", "perplexity"),
    ("MOONSHOT", "moonshot"), ("KIMI", "moonshot"), ("ZHIPU", "zhipu"), ("ZAI_API", "zhipu"),
    ("GLM_", "zhipu"), ("MINIMAX", "minimax"), ("ARK_API", "volcengine"), ("VOLC", "volcengine"),
    ("DOUBAO", "volcengine"), ("QIANFAN", "qianfan"), ("ERNIE", "qianfan"), ("HUNYUAN", "tencent"),
    ("SILICONFLOW", "siliconflow"), ("IFLYTEK", "iflytek"), ("SPARK", "iflytek"), ("TENCENT", "tencent"),
    ("BAIDU", "qianfan"), ("BYTEDANCE", "volcengine"), ("CHATGLM", "zhipu"), ("STEPFUN", "stepfun"),
    ("VENICE", "venice"), ("DIGITALOCEAN", "digitalocean"), ("BASETEN", "baseten"), ("AI302", "302ai"), ("DEEPINFRA", "deepinfra"), ("NOVITA", "novita"),
    ("HYPERBOLIC", "hyperbolic"), ("NEBIUS", "nebius"), ("NVIDIA", "nvidia"), ("NGC_API", "nvidia"),
    ("CEREBRAS", "cerebras"), ("SAMBANOVA", "sambanova"), ("HF_TOKEN", "huggingface"),
    ("HUGGINGFACE", "huggingface"), ("REPLICATE", "replicate"), ("VOYAGE", "voyage"),
    ("AI21", "ai21"), ("GEMINI", "gemini"), ("GOOGLE_API_KEY", "gemini"), ("OLLAMA", "ollama"),
    ("ANTHROPIC", "anthropic"), ("OPENAI", "openai"),
]

# Self-hosted runtimes: no per-token bill, so no price and no model-switch saving
SELF_HOSTED = {"ollama", "lmstudio", "vllm", "self_hosted"}

# Model-family prefix -> maker (used when the call itself doesn't say where it goes)
_FAMILY = [
    (r"^(claude)", "anthropic"), (r"^(gpt|o[1-9]\b|o[1-9]-|chatgpt|text-embedding-3|text-embedding-ada)", "openai"),
    (r"^gemini", "gemini"), (r"^(qwen|qwq|qvq)", "dashscope"), (r"^deepseek", "deepseek"),
    (r"^grok", "xai"), (r"^(mistral|mixtral|codestral|ministral|pixtral|magistral|devstral)", "mistral"),
    (r"^(command|embed-(english|multilingual|v4))", "cohere"), (r"^(glm|chatglm)", "zhipu"),
    (r"^(kimi|moonshot)", "moonshot"), (r"^minimax", "minimax"), (r"^doubao", "volcengine"),
    (r"^ernie", "qianfan"), (r"^hunyuan", "tencent"), (r"^sonar", "perplexity"), (r"^voyage", "voyage"),
    (r"^jamba", "ai21"),
]

# "vendor/model" and "provider:model" prefixes as written in model strings
MODEL_PREFIXES = {
    "openai": "openai", "anthropic": "anthropic", "azure": "azure", "azure_ai": "azure",
    "bedrock": "bedrock", "gemini": "gemini", "google": "gemini", "vertex_ai": "vertex", "vertex": "vertex",
    "groq": "groq", "mistral": "mistral", "together_ai": "together",
    "together": "together", "fireworks_ai": "fireworks", "fireworks": "fireworks", "deepseek": "deepseek",
    "openrouter": "openrouter", "xai": "xai", "cohere": "cohere", "dashscope": "dashscope",
    "moonshot": "moonshot", "zhipu": "zhipu",
    "zai": "zhipu", "minimax": "minimax", "volcengine": "volcengine", "perplexity": "perplexity",
    "ollama": "ollama", "ollama_chat": "ollama", "huggingface": "huggingface", "nvidia_nim": "nvidia",
    "cerebras": "cerebras", "sambanova": "sambanova", "deepinfra": "deepinfra", "replicate": "replicate",
}


def provider_from_url(text: str | None) -> str | None:
    if not text:
        return None
    low = text.lower()
    for host, provider in HOSTS:
        if host in low:
            return provider
    return None


def provider_from_env(name: str | None) -> str | None:
    if not name:
        return None
    up = name.upper()
    for hint, provider in ENV_HINTS:
        if hint in up:
            return provider
    return None


def maker_of(model: str | None) -> str | None:
    if not model:
        return None
    low = model.lower().rsplit("/", 1)[-1]
    for pat, provider in _FAMILY:
        if re.match(pat, low):
            return provider
    return None


def provider_from_model_prefix(model: str | None) -> str | None:
    """'openrouter/anthropic/claude-x' -> openrouter; 'groq:llama-3' (aisuite) -> groq."""
    if not model:
        return None
    m = re.match(r"^([a-z_]+)[/:]", model.strip().lower())
    return MODEL_PREFIXES.get(m.group(1)) if m else None


# SDK packages / provider factories that pin a file to one provider (Vercel AI SDK providers etc.)
SDK_PACKAGES: list[tuple[str, str]] = [
    ("@openrouter/ai-sdk-provider", "openrouter"), ("createopenrouter", "openrouter"),
    ("@ai-sdk/mistral", "mistral"), ("@ai-sdk/groq", "groq"), ("@ai-sdk/deepseek", "deepseek"),
    ("@ai-sdk/xai", "xai"), ("@ai-sdk/togetherai", "together"), ("@ai-sdk/fireworks", "fireworks"),
    ("@ai-sdk/cohere", "cohere"), ("@ai-sdk/amazon-bedrock", "bedrock"), ("@ai-sdk/azure", "azure"),
    ("@ai-sdk/cerebras", "cerebras"), ("@ai-sdk/perplexity", "perplexity"), ("@ai-sdk/deepinfra", "deepinfra"),
    ("@ai-sdk/google-vertex", "vertex"), ("@ai-sdk/google", "gemini"), ("qwen-ai-provider", "dashscope"),
    ("@ai-sdk/alibaba", "dashscope"), ("ollama-ai-provider", "ollama"), ("@ai-sdk/gateway", "vercel_ai_gateway"),
    ("@ai-sdk/anthropic", "anthropic"), ("@ai-sdk/openai\"", "openai"), ("@ai-sdk/openai'", "openai"),
]

_ENV_NAME = re.compile(r"\b([A-Z][A-Z0-9_]*(?:API_KEY|_KEY|_TOKEN|BASE_URL|_URL|_ENDPOINT|_HOST))\b")
_SKIP_OPENAI_LIKE = {"openai", "anthropic"}
_OPENAI_SIGNALS = ("api.openai.com", "openai_api_key", "api.anthropic.com", "anthropic_api_key")


def file_provider_hint(text: str) -> str | None:
    """The single non-OpenAI provider a file points at through hosts or key env vars, if unambiguous.

    Used for OpenAI-shaped calls (the OpenAI SDK pointed at OpenRouter, DashScope, Groq, ...)
    when the client's base URL isn't a literal on the call itself.
    """
    found = set()
    low = text.lower()
    for host, provider in HOSTS:
        if host in low and host != "ollama":        # bare "ollama" is too weak; the :11434 hosts count
            found.add(provider)
    for name in set(_ENV_NAME.findall(text)):
        p = provider_from_env(name)
        if p:
            found.add(p)
    for pkg, provider in SDK_PACKAGES:
        if pkg in low:
            found.add(provider)
    if any(sig in low for sig in _OPENAI_SIGNALS):
        found.add("openai")
    if found & _SKIP_OPENAI_LIKE:
        return None          # the file also talks to OpenAI/Anthropic directly: can't tell which call goes where
    return found.pop() if len(found) == 1 else None


_CLASS_TOKENS = [("openrouter", "openrouter"), ("azure", "azure"), ("bedrock", "bedrock"), ("vertex", "vertex"),
                 ("groq", "groq"), ("deepseek", "deepseek"), ("ollama", "ollama"), ("mistral", "mistral"),
                 ("cohere", "cohere"), ("together", "together"), ("fireworks", "fireworks"), ("cerebras", "cerebras"),
                 ("moonshot", "moonshot"), ("kimi", "moonshot"), ("zhipu", "zhipu"), ("glm", "zhipu"),
                 ("qwen", "dashscope"), ("tongyi", "dashscope"), ("dashscope", "dashscope"), ("minimax", "minimax"),
                 ("doubao", "volcengine"), ("xai", "xai"), ("grok", "xai"), ("perplexity", "perplexity"),
                 ("huggingface", "huggingface"), ("nvidia", "nvidia"), ("sambanova", "sambanova"),
                 ("anthropic", "anthropic"), ("claude", "anthropic"), ("gemini", "gemini"), ("google", "gemini"),
                 ("openai", "openai"), ("gpt", "openai")]


def provider_from_class_name(name: str | None) -> str | None:
    """ChatGroq -> groq, DeepSeekLLM -> deepseek: in-house wrappers usually name their provider."""
    if not name:
        return None
    low = name.lower()
    for token, provider in _CLASS_TOKENS:
        if token in low:
            return provider
    return None
