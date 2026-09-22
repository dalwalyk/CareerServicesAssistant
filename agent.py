#!/usr/bin/env python3
"""A standalone career services chatbot agent.

Runs a chat loop backed by Claude, with tools for resume/job-description
review, a job application tracker, interview and company research notes, an
offer-comparison calculator, and web search for company/role research.

Usage:
    python agent.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

import anthropic

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

MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5")
MAX_TOKENS = 16000
MAX_PAUSE_RESTARTS = 5
# Caps model calls per runner so a runaway tool loop can't run up the bill.
MAX_TOOL_ITERATIONS = 15

# If Claude Opus 5 declines a request, the API retries it on a fallback model
# within the same call instead of returning a refusal.
FALLBACK_OPTIONS = (
    {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    if MODEL == "claude-opus-5"
    else {}
)

SYSTEM_PROMPT = """\
You are a career services assistant running locally for one user. You help
with:
- Reviewing resumes, cover letters, and job descriptions the user points you
  to (PDF, DOCX, CSV, or text files) — feedback, tailoring, gap analysis.
- Tracking job applications: company, role, status, and notes.
- Interview prep and company research, saved as notes for later reference.
- Comparing offers (base pay, hourly equivalents, raises) with the calculator.
- Looking up current information about a company or role with web search.

Use the tools rather than guessing: read a file before reviewing it, check
the application tracker before claiming what's in it, use the calculator for
any arithmetic, and use web search for anything that requires current
information (recent company news, typical salary ranges, role expectations)
rather than relying on training data alone.

Be concise, concrete, and honest — if a resume has a real gap or a cover
letter is generic, say so plainly rather than being falsely encouraging. If a
file path doesn't exist or a tool errors, tell the user plainly what went
wrong rather than making up an answer.
"""

WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 5}

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
    WEB_SEARCH_TOOL,
]


def run_turn(
    client: anthropic.Anthropic,
    messages: list,
    tools: list = TOOLS,
    system: str = SYSTEM_PROMPT,
) -> "anthropic.types.beta.BetaMessage | None":
    """Run one user turn to completion, handling any tool calls and mirroring
    every intermediate message back into `messages` so later turns keep full
    context. Restarts the runner if a long tool sequence pauses mid-turn.
    """
    restarts = 0
    last = None
    while True:
        runner = client.beta.messages.tool_runner(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            max_iterations=MAX_TOOL_ITERATIONS,
            # Cache the conversation prefix so each follow-up message (and any
            # attached resume) isn't billed at the full input rate again.
            cache_control={"type": "ephemeral"},
            system=system,
            tools=tools,
            messages=messages,
            **FALLBACK_OPTIONS,
        )
        last = None
        for message in runner:
            last = message
            messages.append({"role": "assistant", "content": message.content})
            tool_response = runner.generate_tool_call_response()
            if tool_response is not None:
                messages.append(tool_response)

        if last is not None and last.stop_reason == "pause_turn":
            restarts += 1
            if restarts > MAX_PAUSE_RESTARTS:
                print("(agent paused repeatedly and gave up on this turn)")
                return last
            continue

        return last


def main() -> None:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print(
            "No ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) found.\n"
            "Copy .env.example to .env and add your key, or export it in your shell."
        )
        sys.exit(1)

    client = anthropic.Anthropic()
    messages: list = []

    print("Career services agent ready. Type 'exit' or 'quit' to stop.")
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

        messages.append({"role": "user", "content": user_input})

        try:
            response = run_turn(client, messages)
        except anthropic.APIStatusError as exc:
            print(f"(API error: {exc.message})")
            continue
        except anthropic.APIConnectionError:
            print("(network error reaching the Claude API - check your connection)")
            continue

        if response is None:
            continue
        if response.stop_reason == "refusal":
            print("agent> (Claude declined to answer that request.)")
            continue

        for block in response.content:
            if block.type == "text":
                print(f"agent> {block.text}")


if __name__ == "__main__":
    main()
