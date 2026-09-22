import sys
from pathlib import Path

import httpx2
import openai
import pytest
from openai.types.chat import ChatCompletion

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def completion(content=None, tool_calls=None) -> ChatCompletion:
    """Build a Chat Completions response like a provider would return."""
    return ChatCompletion.model_validate({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls" if tool_calls else "stop",
            "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
        }],
    })


def tool_call(call_id, name, arguments, **extra) -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}, **extra}


def status_error(cls, status):
    request = httpx2.Request("POST", "https://example.test/v1/chat/completions")
    return cls("error", response=httpx2.Response(status, request=request), body=None)


class FakeClient:
    """Stands in for openai.OpenAI: returns scripted responses in order and
    records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def rate_limit_error():
    return status_error(openai.RateLimitError, 429)
