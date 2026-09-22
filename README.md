# Career Services Agent

A standalone command-line chatbot with tool use, built to help with a job
search: reviewing resumes and cover letters, tracking applications, prepping
for interviews, and comparing offers. Runs entirely on your own machine.

## What it can do

- **Resume / cover letter / job description review** — point it at a
  `.pdf`, `.docx`, `.csv`, `.txt`, or `.md` file and ask it to review,
  tailor, or compare documents against each other.
- **Job application tracker** — company, role, status (saved / applied /
  phone screen / interviewing / offer / rejected / withdrawn), and notes,
  saved locally in `data/`.
- **Interview & company research notes** — save and search notes on
  recruiters, interview questions, and company research.
- **Offer comparison calculator** — quick arithmetic for converting salary
  to hourly, computing a raise percentage, etc.
- **Web search** — looks up current information (company news, typical
  salary ranges, role expectations) instead of relying on stale knowledge.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Get an Anthropic API key from https://console.anthropic.com/ and put it
   in a `.env` file:

   ```bash
   cp .env.example .env
   # then edit .env and paste your key in
   ```

   (If you already run `ant auth login`, you can skip this — the agent picks
   up that credential automatically.)

3. Run it:

   ```bash
   python agent.py
   ```

## Example session

```
you> read ~/Documents/resume.pdf and review it against this job description:
     [paste the JD]
agent> [reads the resume, then gives concrete, specific feedback]

you> add an application for Backend Engineer at Acme Corp, referred by Jane
agent> Added application #0: Backend Engineer at Acme Corp (applied).

you> what's a $95,000 salary as an hourly rate?
agent> $45.67/hour (assuming a 2,080-hour work year).

you> what's Acme Corp's engineering culture like?
agent> [searches the web, then summarizes what it finds]
```

Type `exit` or `quit` to stop. Conversation history is kept in memory for the
session but not saved to disk; applications and notes persist in `data/`
between runs.

## Files

| File | Purpose |
|---|---|
| `agent.py` | Chat loop and the tool-calling agent |
| `tools.py` | Tool implementations (documents, application tracker, notes, calculator) |
| `data/` | Local JSON storage for applications and notes (created on first run) |

## Notes

- Uses `claude-opus-5` by default; set `CLAUDE_MODEL` in `.env` to use a
  different model (e.g. `claude-sonnet-5` for lower cost).
- Document parsing reads the file directly from disk — nothing is uploaded
  anywhere except to the Claude API as part of the conversation.
- This is a personal tool, not a hosted service: it doesn't apply to jobs or
  contact anyone on your behalf — it reviews, tracks, and researches.
- Your `data/` (applications and notes) and `.env` file are gitignored.
