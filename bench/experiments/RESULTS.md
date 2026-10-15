# `trainlab experiment` on 100 repositories

Every repository of the frozen 100-repo suite (`bench/repos.lock.json`) gets one or two experiments: a
call site `trainlab experiment plan` proposes (auto), the repo's top call site with its billed model named
by hand (explicit), or, for repos with no LLM code, a bare CSV dataset run through the CLI with no scan
(dataset). Models answer from a simulated zoo that speaks the OpenAI, Azure, Anthropic, Gemini and Bedrock
formats; each candidate has a designed accuracy, so the right verdict is known. The zoo injects 429/500
errors and logs every request. No real API key is involved.

**120 experiments on 100 repositories, 240 candidate models, 75 s.**

| Check | Pass | Fail | Not applicable |
|---|---|---|---|
| X01 Run completes | 120 | 0 | 0 |
| X02 Cost = zoo token log x catalog price | 120 | 0 | 0 |
| X03 Report shows the result | 96 | 0 | 24 |
| X04 No PASS for a model below the bar | 120 | 0 | 0 |
| X05 No FAIL for a model clearly above the bar | 120 | 0 | 0 |
| X06 Every sample's pass/fail = truth | 120 | 0 | 0 |
| X07 Data only to configured endpoints, no key on disk | 120 | 0 | 0 |
| X08 Injected 429/500 all recovered | 120 | 0 | 0 |
| X09 Free text without a judge never validates | 2 | 0 | 118 |
| X10 Saving % = measured cost per call | 46 | 0 | 74 |
| X11 Resume never pays twice | 120 | 0 | 0 |
| X12 Budget never exceeded | 120 | 0 | 0 |
| X13 CLI refuses runs that can't give a verdict, before spending | 27 | 0 | 0 |

## Verdicts against the truth

The bar is the threshold the run used (95% by default, 90% for free text, lowered to the current model's
own accuracy minus 2 points when expected answers show it is below that).

| Candidate's true pass rate | PASS | INCONCLUSIVE | FAIL | NEEDS-JUDGE |
|---|---|---|---|---|
| ≥ bar + 3 pts | 55 | 19 | 0 | 5 |
| within 3 pts | 4 | 47 | 2 | 0 |
| ≤ bar − 3 pts | 0 | 11 | 97 | 0 |

PASS given to a model whose true rate is below the bar, by any margin: **0**. Candidates stopped early once clearly failing: 81 (median 35 samples).

## Coverage

- Experiment kinds: explicit 60, auto 46, dataset 14
- Tasks: extraction 53, classification 33, generation 34
- Reference: baseline called 50, expected answers 49, production outputs 21
- Free-text runs with a judge: 32, without: 2
- Through the CLI: 27; budget-cut runs: 10; resumed runs: 13

## Per experiment

| Repository | Kind | Task | Reference | Current model | Candidates (verdict, pass/scored) | Checks |
|---|---|---|---|---|---|---|
| 1Panel-dev/MaxKB | explicit | extraction | fresh | `gpt-4o-mini` | `gpt-5-nano` INCONCLUSIVE 97/100 | all pass |
| 64bit/async-openai | auto | classification | gold | `gpt-5` | `gpt-4.1-mini` FAIL 25/34; `gpt-5-mini` FAIL 13/33; `gpt-4o-mini` PASS 100/100 | all pass |
| 64bit/async-openai | auto | generation | fresh | `gpt-5` | `gpt-4.1-mini` FAIL 39/100; `gpt-5-mini` PASS 99/100; `gpt-4o-mini` PASS 98/100 | all pass |
| Abraxas-365/langchain-rust | explicit | extraction | fresh | `gpt-4o-mini` | `gpt-5-nano` INCONCLUSIVE 95/100 | all pass |
| Azure-Samples/openai | explicit | extraction | reference | `gpt-4.1-nano` | `gpt-5-nano` INCONCLUSIVE 93/100 | all pass |
| BerriAI/litellm | auto | classification | gold | `gpt-3.5-turbo` | `gpt-4.1-mini` INCONCLUSIVE 96/100; `gpt-5-mini` INCONCLUSIVE 37/41; `gpt-4o-mini` FAIL 27/33 | all pass |
| BerriAI/litellm | auto | generation | fresh | `gpt-3.5-turbo` | `gpt-4.1-mini` PASS 100/100; `gpt-5-mini` PASS 99/100; `gpt-4o-mini` PASS 97/100 | all pass |
| BuilderIO/micro-agent | explicit | classification | gold | `gpt-4.1-nano` | `gpt-5-nano` INCONCLUSIVE 98/100 | all pass |
| ChatGPTBox-dev/chatGPTBox | explicit | extraction | gold | `gpt-4o-mini` | `gpt-5-nano` INCONCLUSIVE 95/100 | all pass |
| CopilotKit/CopilotKit | auto | extraction | fresh | `openai/gpt-5.4` | `gpt-4.1-mini` FAIL 27/34; `gpt-5-mini` FAIL 24/33; `gpt-4o-mini` FAIL 11/34 | all pass |
| CopilotKit/CopilotKit | auto | extraction | fresh | `openai/gpt-5.4` | `gpt-4.1-mini` FAIL 33/38; `gpt-5-mini` FAIL 13/34; `gpt-4o-mini` PASS 100/100 | all pass |
| FlowiseAI/Flowise | explicit | extraction | gold | `gpt-4.1-nano` | `gpt-5-nano` FAIL 37/42 | all pass |
| Grt1228/chatgpt-java | explicit | extraction | gold | `gpt-4o-mini` | `gpt-5-nano` INCONCLUSIVE 0/0 | all pass |
| Helicone/helicone | auto | extraction | fresh | `claude-sonnet-4` | `claude-haiku-4-5` INCONCLUSIVE 0/0; `claude-haiku-5-5` INCONCLUSIVE 0/0 | all pass |
| Helicone/helicone | auto | generation | fresh | `claude-sonnet-4` | `claude-haiku-4-5` PASS 98/100; `claude-haiku-5-5` PASS 95/100 | all pass |
| JetBrains/koog | explicit | extraction | fresh | `anthropic.claude-haiku-4-5-20251001-v1:0` | `gpt-5-mini` FAIL 27/33; `gpt-4.1-mini` FAIL 8/34; `gpt-4o-mini` INCONCLUSIVE 98/100 | all pass |
| Mintplex-Labs/anything-llm | explicit | generation | gold | `gpt-4.1-nano` | `gpt-5-nano` INCONCLUSIVE 92/100 | all pass |
| NirDiamant/RAG_Techniques | auto | extraction | fresh | `gpt-4o-mini` | `command-r7b-12-2024` FAIL 34/42 | all pass |
| NirDiamant/RAG_Techniques | auto | extraction | fresh | `gpt-4o-mini` | `gpt-5-nano` PASS 99/100 | all pass |
| Nutlope/llamacoder | explicit | classification | fresh | `openai/gpt-4o-mini` | `gpt-5-nano` FAIL 15/41 | all pass |
| Portkey-AI/gateway | auto | extraction | fresh | `gpt-4o` | `gpt-4.1-mini` PASS 99/100; `gpt-5-mini` PASS 99/100; `gpt-4o-mini` INCONCLUSIVE 92/100 | all pass |
| Portkey-AI/gateway | auto | extraction | fresh | `gpt-4o` | `gpt-4.1-mini` INCONCLUSIVE 95/100; `gpt-5-mini` FAIL 60/67; `gpt-4o-mini` FAIL 28/34 | all pass |
| QuantumNous/new-api | auto | generation | gold | `gpt-4.1-mini` | `gpt-5-mini` INCONCLUSIVE 94/100; `gpt-4o-mini` FAIL 78/100; `gpt-5-nano` FAIL 27/100 | all pass |
| SWE-agent/SWE-agent | explicit | extraction | fresh | `gpt-4o-mini` | `gpt-5-nano` INCONCLUSIVE 97/100 | all pass |
| ScrapeGraphAI/Scrapegraph-ai | explicit | classification | gold | `openai/gpt-4o-mini` | `gpt-5-nano` FAIL 16/42 | all pass |
| THUDM/ChatGLM3 | explicit | extraction | fresh | `openai/gpt-4.1-nano` | `gpt-5-nano` PASS 100/100 | all pass |
| TabbyML/tabby | explicit | classification | fresh | `openai/gpt-4o-mini` | `gpt-5-nano` INCONCLUSIVE 98/100 | all pass |
| agno-agi/agno | auto | extraction | fresh | `mistral-large-latest` | `mistral/mistral-small-latest` INCONCLUSIVE 93/100; `mistral/ministral-8b-latest` FAIL 70/79; `mistral/ministral-3b-latest` FAIL 24/33 | all pass |
| agno-agi/agno | auto | extraction | gold | `mistral-large-latest` | `mistral/mistral-small-latest` PASS 99/100; `mistral/ministral-8b-latest` INCONCLUSIVE 98/100; `mistral/ministral-3b-latest` FAIL 29/34 | all pass |
| alexrudall/ruby-openai | explicit | classification | reference | `gpt-4o-mini` | `gpt-5-nano` FAIL 14/42 | all pass |
| anthropics/anthropic-sdk-java | auto | classification | reference | `global.anthropic.claude-opus-5-5` | `anthropic.claude-haiku-4-5-20251001-v1:0` PASS 99/100; `amazon.nova-lite-v1:0` INCONCLUSIVE 95/100; `amazon.nova-micro-v1:0` INCONCLUSIVE 94/100 | all pass |
| anthropics/anthropic-sdk-java | auto | extraction | gold | `claude-sonnet-4-5` | `claude-haiku-4-5` INCONCLUSIVE 42/46; `claude-haiku-5-5` FAIL 8/34 | all pass |
| appleboy/CodeGPT | explicit | extraction | fresh | `openai/gpt-4.1-nano` | `gpt-5-nano` INCONCLUSIVE 69/75 | all pass |
| assafelovic/gpt-researcher | explicit | classification | reference | `gpt-4` | `gpt-5-mini` FAIL 28/34; `gpt-4.1-mini` FAIL 9/35; `gpt-4o-mini` PASS 100/100 | all pass |
| aws-samples/amazon-bedrock-workshop | auto | generation | fresh | `openai.gpt-oss-120b-1:0` | `amazon.nova-lite-v1:0` FAIL 30/100; `amazon.nova-micro-v1:0` PASS 98/100 | all pass |
| aws-samples/amazon-bedrock-workshop | auto | generation | fresh | `global.anthropic.claude-sonnet-4-6` | `anthropic.claude-haiku-4-5-20251001-v1:0` INCONCLUSIVE 92/100; `amazon.nova-lite-v1:0` FAIL 83/100; `amazon.nova-micro-v1:0` FAIL 37/100 | all pass |
| betalgo/openai | explicit | generation | gold | `gpt-4o-mini` | `gpt-5-nano` FAIL 79/100 | all pass |
| binary-husky/gpt_academic | explicit | extraction | gold | `gpt-4.1-nano` | `gpt-5-nano` INCONCLUSIVE 98/100 | all pass |
| charmbracelet/mods | explicit | generation | fresh | `claude-haiku-4-5` | `claude-haiku-5-5` INCONCLUSIVE 92/100 | all pass |
| cloudwego/eino | explicit | generation | reference | `gpt-4.1-nano` | `gpt-5-nano` INCONCLUSIVE 3/4 | all pass |
| confident-ai/deepeval | auto | extraction | gold | `gpt-3.5-turbo-instruct` | `gpt-4.1-mini` PASS 100/100; `gpt-5-mini` INCONCLUSIVE 97/100; `gpt-4o-mini` INCONCLUSIVE 98/100 | all pass |
| confident-ai/deepeval | auto | classification | reference | `claude-3-5-sonnet-20241022` | `claude-haiku-4-5` INCONCLUSIVE 96/100; `claude-haiku-5-5` FAIL 40/46 | all pass |
| crmne/ruby_llm | auto | generation | fresh | `claude-opus-5-5` | `claude-haiku-4-5` PASS 100/100; `claude-haiku-5-5` PASS 98/100 | all pass |
| crmne/ruby_llm | auto | extraction | reference | `gpt-5.6` | `gpt-4.1-mini` INCONCLUSIVE 95/100; `gpt-5-mini` INCONCLUSIVE 92/100; `gpt-4o-mini` FAIL 28/33 | all pass |
| dair-ai/Prompt-Engineering-Guide | auto | extraction | fresh | `gpt-3.5-turbo-1106` | `gpt-4.1-mini` FAIL 23/34; `gpt-5-mini` FAIL 11/33; `gpt-4o-mini` PASS 99/100 | all pass |
| dair-ai/Prompt-Engineering-Guide | auto | classification | fresh | `mistral-small` | `mistral/ministral-3b-latest` FAIL 8/42 | all pass |
| danielmiessler/fabric | explicit | classification | gold | `openai/gpt-4o-mini` | `gpt-5-nano` INCONCLUSIVE 0/0 | all pass |
| deepseek-ai/DeepSeek-V3 | explicit | classification | gold | `gpt-4o-mini` | `gpt-5-nano` PASS 99/100 | all pass |
| dotnet/ai-samples | explicit | classification | fresh | `openai/gpt-4.1-nano` | `gpt-5-nano` PASS 99/100 | all pass |
| dotnet/eShop | explicit | generation | fresh | `gpt-4o-mini` | `gpt-5-nano` INCONCLUSIVE 93/100 | all pass |
| encode/httpx | dataset | classification | gold | `claude-sonnet-4-5` | `claude-haiku-4-5` FAIL 36/41; `azure/gpt-4o-mini` FAIL 25/33; `gemini/gemini-2.5-flash-lite` FAIL 12/33; `mistral/mistral-small-latest` PASS 100/100 | all pass |
| envoyproxy/ai-gateway | explicit | extraction | reference | `gpt-4.1-nano` | `gpt-5-nano` PASS 100/100 | all pass |
| f/awesome-chatgpt-prompts | explicit | generation | reference | `gpt-4o-mini` | `gpt-5-nano` INCONCLUSIVE 88/100 | all pass |
| fastapi/full-stack-fastapi-template | dataset | classification | gold | `claude-sonnet-4-5` | `claude-haiku-4-5` FAIL 29/35; `azure/gpt-4o-mini` FAIL 25/33; `gemini/gemini-2.5-flash-lite` FAIL 8/32; `mistral/mistral-small-latest` PASS 100/100 | all pass |
| geekan/MetaGPT | explicit | generation | gold | `gpt-4.1-nano` | `gpt-5-nano` PASS 97/100 | all pass |
| gin-gonic/gin | dataset | classification | gold | `claude-sonnet-4-5` | `claude-haiku-4-5` FAIL 30/35; `azure/gpt-4o-mini` FAIL 91/100; `gemini/gemini-2.5-flash-lite` FAIL 28/32; `mistral/mistral-small-latest` FAIL 11/33 | all pass |
| github/awesome-copilot | explicit | classification | reference | `gpt-5` | `gpt-5-mini` INCONCLUSIVE 20/20; `gpt-4.1-mini` INCONCLUSIVE 21/21; `gpt-4o-mini` INCONCLUSIVE 19/20 | all pass |
| google-gemini/cookbook | auto | generation | fresh | `gemini-3.7-flash` | `gemini/gemini-flash-lite-latest` INCONCLUSIVE 89/100; `gemini/gemini-3.5-flash-lite` FAIL 78/100 | all pass |
| google-gemini/cookbook | auto | generation | gold | `gemini-3.7-flash` | `gemini/gemini-flash-lite-latest` FAIL 19/100; `gemini/gemini-3.5-flash-lite` PASS 95/100 | all pass |
| griptape-ai/griptape | explicit | extraction | fresh | `claude-3-7-sonnet-latest` | `claude-haiku-5-5` FAIL 12/37; `claude-haiku-4-5` PASS 100/100 | all pass |
| h2oai/h2ogpt | explicit | generation | fresh | `openai/gpt-4.1-nano` | `gpt-5-nano` INCONCLUSIVE 93/100 | all pass |
| henomis/lingoose | explicit | extraction | fresh | `gpt-4o-mini` | `gpt-5-nano` INCONCLUSIVE 0/0 | all pass |
| huggingface/cookbook | auto | generation | gold | `gpt-4o-mini` | `gpt-5-nano` INCONCLUSIVE 90/100 | all pass |
| janhq/jan | explicit | extraction | gold | `gpt-4o-mini` | `gpt-5-nano` FAIL 33/41 | all pass |
| jquery/jquery | dataset | generation | gold | `claude-sonnet-4-5` | `claude-haiku-4-5` FAIL 26/100; `azure/gpt-4o-mini` PASS 100/100; `gemini/gemini-2.5-flash-lite` PASS 96/100; `mistral/mistral-small-latest` PASS 96/100 | all pass |
| labring/FastGPT | auto | extraction | gold | `gpt-4o-mini` | `gpt-5-nano` FAIL 33/42 | all pass |
| langchain4j/langchain4j-spring | explicit | extraction | fresh | `gpt-4.1-nano` | `gpt-5-nano` FAIL 34/42 | all pass |
| laravel/laravel | dataset | classification | gold | `claude-sonnet-4-5` | `claude-haiku-4-5` PASS 99/100; `azure/gpt-4o-mini` INCONCLUSIVE 97/100; `gemini/gemini-2.5-flash-lite` INCONCLUSIVE 36/40; `mistral/mistral-small-latest` FAIL 27/32 | all pass |
| lm-sys/FastChat | explicit | extraction | reference | `gpt-4-turbo-2024-04-09` | `gpt-5-mini` FAIL 10/33; `gpt-4.1-mini` PASS 99/100; `gpt-4o-mini` PASS 99/100 | all pass |
| lss233/kirara-ai | explicit | extraction | gold | `gpt-4o-mini` | `gpt-5-nano` INCONCLUSIVE 0/0 | all pass |
| mark3labs/mcphost | explicit | classification | fresh | `openai/gpt-4.1-nano` | `gpt-5-nano` FAIL 74/83 | all pass |
| mastra-ai/mastra | explicit | classification | gold | `gpt-4.1-nano` | `gpt-5-nano` FAIL 31/39 | all pass |
| maximhq/bifrost | auto | generation | reference | `claude-3-sonnet-20240229` | `claude-haiku-4-5` FAIL 26/100; `claude-haiku-5-5` PASS 98/100 | all pass |
| maximhq/bifrost | auto | generation | fresh | `claude-3-sonnet-20240229` | `claude-haiku-4-5` NEEDS-JUDGE 0/0; `claude-haiku-5-5` NEEDS-JUDGE 0/0 | all pass |
| microsoft/TaskWeaver | explicit | extraction | fresh | `gpt-4.1-nano` | `gpt-5-nano` INCONCLUSIVE 97/100 | all pass |
| mlabonne/llm-course | explicit | generation | fresh | `openai/gpt-4.1-nano` | `gpt-5-nano` PASS 98/100 | all pass |
| modelcontextprotocol/python-sdk | explicit | classification | reference | `openai/gpt-3.5-turbo` | `gpt-5-mini` FAIL 26/34; `gpt-4.1-mini` FAIL 23/33; `gpt-4o-mini` FAIL 7/33 | all pass |
| modelcontextprotocol/servers | explicit | extraction | fresh | `gpt-4o-mini` | `gpt-5-nano` PASS 100/100 | all pass |
| modelscope/ms-agent | explicit | generation | reference | `Qwen/Qwen3-235B-A22B-Instruct-2507` | `dashscope/qwen-turbo` INCONCLUSIVE 91/100; `dashscope/qwen3.8-flash` FAIL 78/100; `dashscope/qwen-plus` FAIL 28/100 | all pass |
| mudler/LocalAI | explicit | classification | fresh | `openai/gpt-4.1-nano` | `gpt-5-nano` PASS 100/100 | all pass |
| ollama/ollama | explicit | extraction | gold | `gpt-4o-mini` | `gpt-5-nano` PASS 100/100 | all pass |
| oobabooga/text-generation-webui | explicit | classification | reference | `claude-haiku-4-5` | `claude-haiku-5-5` INCONCLUSIVE 97/100 | all pass |
| openai-php/client | explicit | extraction | gold | `openai/gpt-4.1-nano` | `gpt-5-nano` FAIL 75/83 | all pass |
| openai-php/laravel | explicit | classification | fresh | `gpt-4.1-nano` | `gpt-5-nano` FAIL 34/41 | all pass |
| openai/openai-dotnet | auto | generation | reference | `gpt-5` | `gpt-4.1-mini` PASS 96/100; `gpt-5-mini` INCONCLUSIVE 88/100; `gpt-4o-mini` FAIL 79/100 | all pass |
| openai/openai-dotnet | auto | classification | gold | `gpt-5` | `gpt-4.1-mini` FAIL 28/33; `gpt-5-mini` FAIL 29/36; `gpt-4o-mini` FAIL 13/35 | all pass |
| openai/swarm | auto | extraction | gold | `gpt-4-turbo-preview` | `gpt-4.1-mini` FAIL 10/35; `gpt-5-mini` PASS 100/100; `gpt-4o-mini` PASS 100/100 | all pass |
| orhanerday/open-ai | explicit | classification | fresh | `gpt-4o-mini` | `gpt-5-nano` FAIL 11/41 | all pass |
| pallets/flask | dataset | generation | gold | `claude-sonnet-4-5` | `claude-haiku-4-5` PASS 99/100; `azure/gpt-4o-mini` PASS 98/100; `gemini/gemini-2.5-flash-lite` INCONCLUSIVE 92/100; `mistral/mistral-small-latest` INCONCLUSIVE 88/100 | all pass |
| pinecone-io/examples | auto | extraction | fresh | `gpt-4o-mini` | `gpt-5-nano` INCONCLUSIVE 0/0 | all pass |
| pinecone-io/examples | auto | extraction | gold | `gpt-4o-mini-2024-07-18` | `gpt-5-nano` INCONCLUSIVE 98/100 | all pass |
| promptfoo/promptfoo | auto | classification | reference | `gpt-4o-2024-08-06` | `gpt-4.1-mini` FAIL 12/33; `gpt-5-mini` PASS 100/100; `gpt-4o-mini` INCONCLUSIVE 98/100 | all pass |
| promptfoo/promptfoo | auto | generation | gold | `gpt-4o` | `gpt-4.1-mini` FAIL 77/100; `gpt-5-mini` FAIL 26/100; `gpt-4o-mini` PASS 97/100 | all pass |
| psf/requests | dataset | extraction | gold | `claude-sonnet-4-5` | `claude-haiku-4-5` PASS 100/100; `azure/gpt-4o-mini` INCONCLUSIVE 98/100; `gemini/gemini-2.5-flash-lite` INCONCLUSIVE 98/100; `mistral/mistral-small-latest` FAIL 26/32 | all pass |
| pydantic/pydantic-ai | auto | extraction | fresh | `us.amazon.nova-2-lite-v1:0` | `amazon.nova-lite-v1:0` INCONCLUSIVE 96/100; `amazon.nova-micro-v1:0` INCONCLUSIVE 96/100 | all pass |
| reworkd/AgentGPT | auto | generation | fresh | `gpt-3.5-turbo` | `gpt-4.1-mini` INCONCLUSIVE 93/100; `gpt-5-mini` FAIL 80/100; `gpt-4o-mini` FAIL 31/100 | all pass |
| run-llama/LlamaIndexTS | auto | generation | gold | `gpt-3.5-turbo` | `gpt-4.1-mini` NEEDS-JUDGE 0/0; `gpt-5-mini` NEEDS-JUDGE 0/0; `gpt-4o-mini` NEEDS-JUDGE 0/0 | all pass |
| run-llama/LlamaIndexTS | auto | generation | fresh | `gpt-4o` | `gpt-4.1-mini` FAIL 85/100; `gpt-5-mini` FAIL 28/100; `gpt-4o-mini` PASS 99/100 | all pass |
| rustformers/llm | explicit | extraction | fresh | `gpt-4o-mini` | `gpt-5-nano` FAIL 59/68 | all pass |
| sigoden/aichat | explicit | generation | reference | `gpt-4.1-nano` | `gpt-5-nano` INCONCLUSIVE 3/3 | all pass |
| slackapi/bolt-js | dataset | generation | gold | `claude-sonnet-4-5` | `claude-haiku-4-5` INCONCLUSIVE 91/100; `azure/gpt-4o-mini` INCONCLUSIVE 92/100; `gemini/gemini-2.5-flash-lite` FAIL 74/100; `mistral/mistral-small-latest` FAIL 23/100 | all pass |
| spring-projects/spring-ai | explicit | extraction | fresh | `anthropic.claude-haiku-4-5-20251001-v1:0` | `gpt-5-mini` PASS 99/100; `gpt-4.1-mini` INCONCLUSIVE 98/100; `gpt-4o-mini` INCONCLUSIVE 95/100 | all pass |
| spring-projects/spring-petclinic | dataset | classification | gold | `claude-sonnet-4-5` | `claude-haiku-4-5` FAIL 32/37; `azure/gpt-4o-mini` FAIL 22/32; `gemini/gemini-2.5-flash-lite` FAIL 14/35; `mistral/mistral-small-latest` PASS 100/100 | all pass |
| stackblitz-labs/bolt.diy | explicit | generation | fresh | `gpt-4o-mini` | `gpt-5-nano` PASS 99/100 | all pass |
| stanford-oval/storm | explicit | classification | reference | `gpt-4o-2024-05-13` | `gpt-5-mini` INCONCLUSIVE 96/100; `gpt-4.1-mini` FAIL 30/34; `gpt-4o-mini` FAIL 27/34 | all pass |
| stripe/stripe-python | dataset | extraction | gold | `claude-sonnet-4-5` | `claude-haiku-4-5` FAIL 5/32; `azure/gpt-4o-mini` PASS 100/100; `gemini/gemini-2.5-flash-lite` INCONCLUSIVE 98/100; `mistral/mistral-small-latest` INCONCLUSIVE 94/100 | all pass |
| symfony/ai | explicit | extraction | gold | `openai/gpt-4o-mini` | `gpt-5-nano` INCONCLUSIVE 94/100 | all pass |
| tjake/Jlama | explicit | extraction | fresh | `openai/gpt-4o-mini` | `gpt-5-nano` FAIL 36/42 | all pass |
| tokio-rs/axum | dataset | generation | gold | `claude-sonnet-4-5` | `claude-haiku-4-5` FAIL 32/100; `azure/gpt-4o-mini` PASS 100/100; `gemini/gemini-2.5-flash-lite` PASS 97/100; `mistral/mistral-small-latest` INCONCLUSIVE 90/100 | all pass |
| tryAGI/LangChain | explicit | generation | gold | `openai/gpt-4.1-nano` | `gpt-5-nano` INCONCLUSIVE 87/100 | all pass |
| twbs/bootstrap | dataset | classification | gold | `claude-sonnet-4-5` | `claude-haiku-4-5` FAIL 24/31; `azure/gpt-4o-mini` FAIL 12/34; `gemini/gemini-2.5-flash-lite` PASS 100/100; `mistral/mistral-small-latest` PASS 99/100 | all pass |
| twilio/twilio-node | dataset | extraction | gold | `claude-sonnet-4-5` | `claude-haiku-4-5` INCONCLUSIVE 97/100; `azure/gpt-4o-mini` FAIL 28/32; `gemini/gemini-2.5-flash-lite` FAIL 22/31; `mistral/mistral-small-latest` FAIL 10/31 | all pass |
| twilio/twilio-python | dataset | extraction | gold | `claude-sonnet-4-5` | `claude-haiku-4-5` PASS 100/100; `azure/gpt-4o-mini` INCONCLUSIVE 98/100; `gemini/gemini-2.5-flash-lite` FAIL 37/42; `mistral/mistral-small-latest` FAIL 79/87 | all pass |
| vanna-ai/vanna | auto | extraction | gold | `gemini-2.5-pro` | `gemini/gemini-flash-lite-latest` FAIL 28/35; `gemini/gemini-3.5-flash-lite` FAIL 13/37 | all pass |
| vanna-ai/vanna | auto | extraction | fresh | `gemini-2.5-pro` | `gemini/gemini-flash-lite-latest` FAIL 14/35; `gemini/gemini-3.5-flash-lite` INCONCLUSIVE 98/100 | all pass |
| vercel/ai-chatbot | explicit | extraction | reference | `openai/gpt-4o-mini` | `gpt-5-nano` PASS 100/100 | all pass |
| xtekky/gpt4free | auto | extraction | reference | `gpt-4o-mini` | `gpt-5-nano` INCONCLUSIVE 17/17 | all pass |
| xtekky/gpt4free | auto | classification | gold | `gpt-4o-mini` | `gpt-5-nano` INCONCLUSIVE 95/100 | all pass |
| zhayujie/chatgpt-on-wechat | explicit | extraction | fresh | `gpt-4.1-nano` | `gpt-5-nano` INCONCLUSIVE 93/100 | all pass |
| zylon-ai/private-gpt | explicit | classification | gold | `openai/gpt-4o-mini` | `gpt-5-nano` INCONCLUSIVE 39/43 | all pass |

Reproduce: `python bench/experiments/prepare.py WORK && python bench/experiments/run.py WORK && python bench/experiments/report.py`.
