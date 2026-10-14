from tl import BASE
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", base_url=BASE)
except Exception:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", transport="rest", client_options={"api_endpoint": BASE})
llm.invoke("hi")  # @call
for _ in llm.stream("hi"):  # @call
    pass
