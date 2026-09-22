import json

import pytest

import agent
from conftest import FakeClient, completion, tool_call
from tools import MemoryStore, make_tracker_tools


@pytest.fixture
def tracker():
    return make_tracker_tools(MemoryStore())


def test_tool_call_round_trip(tracker):
    client = FakeClient([
        completion(tool_calls=[tool_call(
            "call_1", "add_application", json.dumps({"company": "Acme", "role": "Analyst"}),
            extra_content={"google": {"thought_signature": "sig"}},
        )]),
        completion("Added it to your tracker."),
    ])
    messages = [{"role": "user", "content": "add Analyst at Acme"}]

    reply = agent.run_turn(client, messages, tools=list(tracker.values()), system="SYS")

    assert reply == "Added it to your tracker."
    assert tracker["list_applications"].call({}) == "0. Analyst at Acme [applied]"
    # History: user, assistant tool call, tool result, final assistant.
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[1]["tool_calls"][0]["function"]["name"] == "add_application"
    # Provider-specific fields (Gemini thought signatures) are sent back.
    assert messages[1]["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "sig"}}
    assert messages[2]["tool_call_id"] == "call_1"
    assert messages[2]["content"].startswith("Added application #0")
    # Every request carries the system prompt first and the tool schemas.
    first = client.requests[0]
    assert first["messages"][0] == {"role": "system", "content": "SYS"}
    assert first["model"] == agent.LLM_MODEL
    assert {t["function"]["name"] for t in first["tools"]} == set(tracker)
    assert client.requests[1]["messages"][1:] == messages[:3]


def test_bad_tool_calls_are_reported_to_the_model(tracker):
    client = FakeClient([
        completion(tool_calls=[
            tool_call("a", "no_such_tool", "{}"),
            tool_call("b", "add_application", "{not json"),
            tool_call("c", "add_application", "[1]"),
        ]),
        completion("Sorry about that."),
    ])
    messages = [{"role": "user", "content": "hi"}]
    agent.run_turn(client, messages, tools=list(tracker.values()), system="SYS")
    results = [m["content"] for m in messages if m["role"] == "tool"]
    assert results == [
        "Error: there is no tool named 'no_such_tool'.",
        "Error: the arguments for add_application weren't valid JSON.",
        "Error: the arguments for add_application must be a JSON object.",
    ]


def test_tool_iteration_cap(tracker, monkeypatch):
    monkeypatch.setattr(agent, "MAX_TOOL_ITERATIONS", 3)
    looping = completion(tool_calls=[tool_call("x", "list_notes", "{}")])
    client = FakeClient([looping] * 3)
    reply = agent.run_turn(client, [{"role": "user", "content": "hi"}],
                           tools=list(tracker.values()), system="SYS")
    assert len(client.requests) == 3
    assert "too many tool calls" in reply


def test_rate_limit_propagates(tracker, rate_limit_error):
    import openai
    client = FakeClient([rate_limit_error])
    with pytest.raises(openai.RateLimitError):
        agent.run_turn(client, [{"role": "user", "content": "hi"}],
                       tools=list(tracker.values()), system="SYS")


def test_make_client_reads_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    client = agent.make_client()
    assert client.api_key == "test-key"
    assert str(client.base_url).rstrip("/") == agent.LLM_BASE_URL.rstrip("/")




def test_is_auth_error():
    import openai
    from conftest import status_error
    assert agent.is_auth_error(status_error(openai.AuthenticationError, 401))
    gemini_bad_key = openai.BadRequestError(
        "Error code: 400 - Please pass a valid API key",
        response=status_error(openai.BadRequestError, 400).response, body=None)
    assert agent.is_auth_error(gemini_bad_key)
    assert not agent.is_auth_error(status_error(openai.BadRequestError, 400))
    assert not agent.is_auth_error(status_error(openai.RateLimitError, 429))
