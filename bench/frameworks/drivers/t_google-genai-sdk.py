import asyncio

from tl import BASE
from google import genai
from google.genai import types

c = genai.Client(api_key="g", http_options=types.HttpOptions(base_url=BASE))
c.models.generate_content(model="gemini-2.5-flash", contents="hi")  # @call
for _ in c.models.generate_content_stream(model="gemini-2.5-pro", contents="hi"):  # @call
    pass
chat = c.chats.create(model="gemini-2.5-flash")
chat.send_message("hi")  # @call


async def main():
    await c.aio.models.generate_content(model="gemini-2.5-flash", contents="hi")  # @call
asyncio.run(main())
