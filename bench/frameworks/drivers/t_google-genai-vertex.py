"""Vertex AI through the current Google Gen AI SDK (the legacy vertexai SDK is deprecated)."""
from tl import BASE
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials

c = genai.Client(vertexai=True, project="p", location="us-central1", credentials=Credentials(token="fake"),
                 http_options=types.HttpOptions(base_url=BASE))
c.models.generate_content(model="gemini-2.5-flash", contents="hi")  # @call
