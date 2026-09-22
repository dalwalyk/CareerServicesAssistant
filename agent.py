#!/usr/bin/env python3
"""A standalone career services chatbot agent.

Runs a chat loop backed by any OpenAI-compatible chat model (Google Gemini's
free tier by default; Groq or a local Ollama model work too), with tools for
resume/job-description review, a job application tracker, interview and
company research notes, and an offer-comparison calculator.

Configure it with LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL (see README).

Usage:
    python agent.py
"""

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

import openai

from tools import (
    add_application,
    add_note,
    calculate,
    get_current_datetime,
    list_applications,
    list_directory,
    list_notes,
    read_document,
    remove_application,
    search_notes,
    update_application,
)

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.8-flash")
# Caps model calls per turn so a runaway tool loop can't burn through quota.
MAX_TOOL_ITERATIONS = 15

QUOTA_MESSAGE = "The free AI quota is used up, please try again later."
AUTH_MESSAGE = "The AI provider rejected the API key. Check LLM_API_KEY."
BUSY_MESSAGE = "The AI service is busy right now, please try again in a minute."

SYSTEM_PROMPT = """\
You are a career services assistant running locally for one user. You help
with:
- Reviewing resumes, cover letters, and job descriptions the user points you
  to (PDF, DOCX, CSV, or text files) — feedback, tailoring, gap analysis.
- Tracking job applications: company, role, status, and notes.
- Interview prep and company research, saved as notes for later reference.
- Comparing offers (base pay, hourly equivalents, raises) with the calculator.

Use the tools rather than guessing: read a file before reviewing it, check
the application tracker before claiming what's in it, and use the calculator
for any arithmetic. You can't browse the web, so for anything that needs
current information (recent company news, typical salary ranges), say that
your knowledge may be out of date and suggest where the user can check.

Be concise, concrete, and honest — if a resume has a real gap or a cover
letter is generic, say so plainly rather than being falsely encouraging. If a
file path doesn't exist or a tool errors, tell the user plainly what went
wrong rather than making up an answer.
"""

TOOLS = [
    read_document,
    list_directory,
    add_application,
    list_applications,
    update_application,
    remove_application,
    add_note,
    list_notes,
    search_notes,
    calculate,
    get_current_datetime,
]


def make_client() -> openai.OpenAI:
    """Create a client for the configured provider. Reads LLM_API_KEY from the
    environment; never hardcode it."""
    return openai.OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=LLM_BASE_URL)


def is_auth_error(exc: openai.APIError) -> bool:
    """True if the provider rejected the API key. Gemini reports a bad key as
    400 "Please pass a valid API key" rather than 401."""
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return True
    return isinstance(exc, openai.BadRequestError) and "api key" in str(exc).lower()


def is_busy_error(exc: openai.APIError) -> bool:
    """True if the provider is temporarily overloaded or too slow to answer,
    e.g. Gemini's 503 "This model is currently experiencing high demand"."""
    if isinstance(exc, openai.APITimeoutError):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code in (502, 503, 504)


def _assistant_entry(message) -> dict:
    """Convert a response message into a history entry to send back."""
    if not message.tool_calls:
        return {"role": "assistant", "content": message.content or ""}
    return {
        "role": "assistant",
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
                # Keep provider-specific fields (e.g. Gemini's thought
                # signatures), which some models require to be sent back.
                **(call.model_extra or {}),
            }
            for call in message.tool_calls
        ],
    }


def _run_tool_call(tools_by_name: dict, call) -> str:
    tool = tools_by_name.get(call.function.name)
    if tool is None:
        return f"Error: there is no tool named '{call.function.name}'."
    try:
        arguments = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        return f"Error: the arguments for {call.function.name} weren't valid JSON."
    if not isinstance(arguments, dict):
        return f"Error: the arguments for {call.function.name} must be a JSON object."
    return tool.call(arguments)


def run_turn(
    client: openai.OpenAI,
    messages: list,
    tools: list = TOOLS,
    system: str = SYSTEM_PROMPT,
) -> str:
    """Run one user turn to completion and return the reply text. Executes any
    tool calls and appends every assistant and tool message to `messages` so
    later turns keep full context. Raises openai.APIError on API failures.
    """
    tools_by_name = {t.name: t for t in tools}
    schemas = [t.schema for t in tools]
    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": system}, *messages],
            tools=schemas,
        )
        message = response.choices[0].message
        messages.append(_assistant_entry(message))
        if not message.tool_calls:
            return message.content or ""
        for call in message.tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": _run_tool_call(tools_by_name, call),
            })
    return "(I stopped after too many tool calls without finishing. Please try rephrasing.)"


def main() -> None:
    if not os.environ.get("LLM_API_KEY"):
        print(
            "No LLM_API_KEY found.\n"
            "Copy .env.example to .env and add your key, or export it in your shell."
        )
        sys.exit(1)

    client = make_client()
    messages: list = []

    print(f"Career services agent ready ({LLM_MODEL}). Type 'exit' or 'quit' to stop.")
    print(
        "Try: 'read ~/Documents/resume.pdf and review it against this job "
        "description: ...', or 'add an application for Backend Engineer at Acme'.\n"
    )

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        turn_start = len(messages)
        messages.append({"role": "user", "content": user_input})

        try:
            reply = run_turn(client, messages)
        except openai.APIError as exc:
            # Drop the failed turn so the history stays valid for the next try.
            del messages[turn_start:]
            if isinstance(exc, openai.RateLimitError):
                print(f"({QUOTA_MESSAGE})")
            elif is_auth_error(exc):
                print(f"({AUTH_MESSAGE})")
            elif is_busy_error(exc):
                print(f"({BUSY_MESSAGE})")
            elif isinstance(exc, openai.APIConnectionError):
                print(f"(network error reaching {LLM_BASE_URL} - check your connection)")
            else:
                print(f"(API error: {exc})")
            continue

        print(f"agent> {reply}")


if __name__ == "__main__":
    main()
