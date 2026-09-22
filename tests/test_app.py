import openai
import pytest
from streamlit.testing.v1 import AppTest

import agent
from conftest import ROOT, status_error

APP = str(ROOT / "app.py")


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.delenv("ACCESS_CODE", raising=False)
    monkeypatch.delenv("MAX_MESSAGES_PER_SESSION", raising=False)


def fake_turn(reply="Added it to your tracker.", error=None):
    def run_turn(client, messages, tools, system):
        if error:
            raise error
        add = next(t for t in tools if t.name == "add_application")
        add.call({"company": "Acme", "role": "Backend Engineer"})
        messages.append({"role": "assistant", "content": reply})
        return reply
    return run_turn


def chat_texts(at):
    return [m.markdown[0].value for m in at.chat_message]


def test_chat_updates_tracker(monkeypatch):
    monkeypatch.setattr(agent, "run_turn", fake_turn())
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    at.chat_input[0].set_value("add Backend Engineer at Acme").run()
    assert not at.exception
    assert chat_texts(at)[-2:] == ["add Backend Engineer at Acme", "Added it to your tracker."]
    assert len(at.sidebar.dataframe[0].value) == 1


def test_privacy_notice_mentions_provider():
    at = AppTest.from_file(APP, default_timeout=30).run()
    captions = " ".join(c.value for c in at.sidebar.caption)
    assert "Google's Gemini API" in captions
    assert "sensitive personal information" in captions


def test_rate_limit_shows_quota_message(monkeypatch):
    monkeypatch.setattr(agent, "run_turn", fake_turn(error=status_error(openai.RateLimitError, 429)))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("hello").run()
    assert not at.exception
    assert chat_texts(at)[-1] == "The free AI quota is used up, please try again later."


def test_session_message_limit(monkeypatch):
    monkeypatch.setenv("MAX_MESSAGES_PER_SESSION", "1")
    monkeypatch.setattr(agent, "run_turn", fake_turn())
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("hi").run()
    assert at.chat_input[0].disabled
    assert "message limit" in at.info[0].value


def test_access_code_gate(monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "letmein")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert len(at.chat_input) == 0
    at.text_input[0].set_value("wrong").run()
    assert at.error[0].value == "That access code isn't right."
    at.text_input[0].set_value("letmein").run()
    assert len(at.chat_input) == 1


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert "LLM_API_KEY isn't set" in at.error[0].value
    assert len(at.chat_input) == 0


def test_bad_api_key_message(monkeypatch):
    monkeypatch.setattr(agent, "run_turn", fake_turn(error=status_error(openai.AuthenticationError, 401)))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("hello").run()
    assert "rejected its API key" in chat_texts(at)[-1]


def test_busy_provider_message(monkeypatch):
    monkeypatch.setattr(agent, "run_turn", fake_turn(error=status_error(openai.InternalServerError, 503)))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("hello").run()
    assert chat_texts(at)[-1] == "The AI service is busy right now, please try again in a minute."
