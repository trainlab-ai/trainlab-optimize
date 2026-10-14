from tl import BASE
import cohere

c2 = cohere.ClientV2(api_key="c", base_url=BASE)
c2.chat(model="command-r-plus", messages=[{"role": "user", "content": "hi"}])  # @call
for _ in c2.chat_stream(model="command-r", messages=[{"role": "user", "content": "hi"}]):  # @call
    pass
c2.embed(model="embed-english-v3.0", texts=["a"], input_type="search_document", embedding_types=["float"])  # @call
c1 = cohere.Client(api_key="c", base_url=BASE)
c1.chat(model="command-r", message="hi")  # @call
