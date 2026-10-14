"""Candidate pool for the 100-repo benchmark, by corner-case category.

`bench/sample.py` draws a stratified random sample from this pool with a fixed seed, so the
run is reproducible; repos that can't be cloned are replaced from the same category.
Repositories used in earlier benchmark rounds (scripts/eval_repos.py) are excluded.

`llm` is False for negative controls: projects with no LLM API usage. Some are deliberate
traps whose APIs look like LLM calls (Twilio/Slack/Stripe `messages.create`, `chat.*`).
"""

POOL: dict[str, dict] = {
    "python-agents-rag": {
        "quota": 14, "llm": True,
        "repos": ["geekan/MetaGPT", "OpenBMB/ChatDev", "assafelovic/gpt-researcher", "stanford-oval/storm",
                  "deepset-ai/haystack", "langchain-ai/langgraph", "pydantic/pydantic-ai", "567-labs/instructor",
                  "simonw/llm", "Shubhamsaboo/awesome-llm-apps", "QuivrHQ/quivr", "PrefectHQ/marvin",
                  "guidance-ai/guidance", "stanfordnlp/dspy", "letta-ai/letta", "agno-agi/agno",
                  "huggingface/smolagents", "openai/openai-agents-python", "openai/swarm", "vanna-ai/vanna",
                  "yoheinakajima/babyagi", "Pythagora-io/gpt-pilot", "OpenInterpreter/open-interpreter",
                  "gptme/gptme", "SWE-agent/SWE-agent", "unclecode/crawl4ai", "ScrapeGraphAI/Scrapegraph-ai",
                  "microsoft/TaskWeaver", "langroid/langroid", "griptape-ai/griptape", "explodinggradients/ragas",
                  "confident-ai/deepeval", "zilliztech/GPTCache", "zylon-ai/private-gpt", "SciPhi-AI/R2R",
                  "Cinnamon/kotaemon", "AgentOps-AI/agentops", "TransformerOptimus/SuperAGI", "h2oai/h2ogpt"],
    },
    "js-ts-apps": {
        "quota": 12, "llm": True,
        "repos": ["vercel/ai-chatbot", "Nutlope/llamacoder", "Nutlope/aicommits", "di-sukharev/opencommit",
                  "transitive-bullshit/agentic", "run-llama/LlamaIndexTS", "mastra-ai/mastra", "FlowiseAI/Flowise",
                  "anse-app/anse", "ChatGPTBox-dev/chatGPTBox", "CopilotKit/CopilotKit", "assistant-ui/assistant-ui",
                  "e2b-dev/fragments", "openai/openai-realtime-console", "huggingface/chat-ui",
                  "stackblitz-labs/bolt.diy", "cline/cline", "RooCodeInc/Roo-Code", "Mintplex-Labs/anything-llm",
                  "promptfoo/promptfoo", "langfuse/langfuse", "BuilderIO/micro-agent", "reworkd/AgentGPT"],
    },
    "go": {
        "quota": 7, "llm": True,
        "repos": ["sashabaranov/go-openai", "charmbracelet/mods", "appleboy/CodeGPT", "henomis/lingoose",
                  "k8sgpt-ai/k8sgpt", "danielmiessler/fabric", "cloudwego/eino", "envoyproxy/ai-gateway",
                  "openai/openai-go", "anthropics/anthropic-sdk-go", "mark3labs/mcphost"],
    },
    "jvm": {
        "quota": 6, "llm": True,
        "repos": ["openai/openai-java", "alibaba/spring-ai-alibaba", "JetBrains/koog", "tjake/Jlama",
                  "Grt1228/chatgpt-java", "aallam/openai-kotlin", "anthropics/anthropic-sdk-java",
                  "langchain4j/langchain4j-spring", "spring-projects/spring-ai"],
    },
    "dotnet": {
        "quota": 5, "llm": True,
        "repos": ["microsoft/kernel-memory", "dotnet/ai-samples", "openai/openai-dotnet", "tryAGI/LangChain",
                  "betalgo/openai", "Azure-Samples/azure-search-openai-demo-csharp", "dotnet/eShop"],
    },
    "rust": {
        "quota": 5, "llm": True,
        "repos": ["64bit/async-openai", "jeremychone/rust-genai", "sigoden/aichat", "TabbyML/tabby",
                  "Abraxas-365/langchain-rust", "rustformers/llm"],
    },
    "ruby-php": {
        "quota": 6, "llm": True,
        "repos": ["alexrudall/ruby-openai", "crmne/ruby_llm", "discourse/discourse-ai", "openai-php/client",
                  "openai-php/laravel", "prism-php/prism", "orhanerday/open-ai", "symfony/ai"],
    },
    "notebooks": {
        "quota": 7, "llm": True,
        "repos": ["google-gemini/cookbook", "microsoft/generative-ai-for-beginners", "mlabonne/llm-course",
                  "NirDiamant/GenAI_Agents", "NirDiamant/RAG_Techniques", "pinecone-io/examples",
                  "huggingface/cookbook", "aws-samples/amazon-bedrock-workshop", "Azure-Samples/openai",
                  "anthropics/courses"],
    },
    "chinese-providers": {
        "quota": 9, "llm": True,
        "repos": ["chatchat-space/Langchain-Chatchat", "labring/FastGPT", "netease-youdao/QAnything",
                  "1Panel-dev/MaxKB", "modelscope/ms-agent", "binary-husky/gpt_academic",
                  "zhayujie/chatgpt-on-wechat", "AstrBotDevs/AstrBot", "lss233/kirara-ai", "xtekky/gpt4free",
                  "THUDM/ChatGLM3", "MoonshotAI/Kimi-K2", "deepseek-ai/DeepSeek-V3"],
    },
    "local-self-hosted": {
        "quota": 5, "llm": True,
        "repos": ["lm-sys/FastChat", "oobabooga/text-generation-webui", "nomic-ai/gpt4all", "janhq/jan",
                  "mudler/LocalAI", "ollama/ollama", "vllm-project/vllm"],
    },
    "gateways": {
        "quota": 5, "llm": True,
        "repos": ["BerriAI/litellm", "Portkey-AI/gateway", "Helicone/helicone", "songquanpeng/one-api",
                  "QuantumNous/new-api", "maximhq/bifrost"],
    },
    "mcp-prompts-config": {
        "quota": 5, "llm": True,
        "repos": ["modelcontextprotocol/servers", "f/awesome-chatgpt-prompts", "dair-ai/Prompt-Engineering-Guide",
                  "x1xhlol/system-prompts-and-models-of-ai-tools", "github/awesome-copilot",
                  "modelcontextprotocol/python-sdk", "punkpeye/awesome-mcp-servers"],
    },
    "negative-controls": {
        "quota": 10, "llm": False,
        "repos": ["pallets/flask", "psf/requests", "encode/httpx", "expressjs/express", "gin-gonic/gin",
                  "spring-projects/spring-petclinic", "sinatra/sinatra", "laravel/laravel", "tokio-rs/axum",
                  "BurntSushi/ripgrep", "twbs/bootstrap", "jquery/jquery", "fastapi/full-stack-fastapi-template"],
    },
    "negative-traps": {
        "quota": 4, "llm": False,
        "repos": ["twilio/twilio-python", "slackapi/python-slack-sdk", "stripe/stripe-python",
                  "sendgrid/sendgrid-python", "twilio/twilio-node", "slackapi/bolt-js"],
    },
}

assert sum(c["quota"] for c in POOL.values()) == 100
