from tl import BASE
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator

AnthropicChatGenerator(model="claude-sonnet-4-5").run([ChatMessage.from_user("hi")])  # @call
