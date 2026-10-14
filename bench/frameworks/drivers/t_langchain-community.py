"""Older apps: langchain_community integrations that call HTTP APIs with requests directly."""
from tl import BASE
from langchain_community.llms import Ollama

Ollama(model="llama3.2", base_url=BASE).invoke("hi")  # @call
