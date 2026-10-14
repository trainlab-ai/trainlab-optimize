from tl import BASE
import google.generativeai as genai

genai.configure(api_key="g", transport="rest", client_options={"api_endpoint": BASE})
m = genai.GenerativeModel("gemini-1.5-flash")
m.generate_content("hi")  # @call
for _ in m.generate_content("hi", stream=True):  # @call
    pass
