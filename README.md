# Career Services Agent

A chatbot with tool use, built to help with a job search: reviewing resumes
and cover letters, tracking applications, prepping for interviews, and
comparing offers. It runs on any AI provider with an OpenAI-compatible Chat
Completions API. By default that's **Google Gemini's free tier**; switching
to Groq or a local Ollama model only takes a change of settings.

It comes in two forms:

- **Web app** (`app.py`) — a public website anyone can use from a link, no
  install or API key needed on their side. See [Web app](#web-app).
- **Command-line app** (`agent.py`) — runs on your own machine and reads
  files straight from your disk.

## What it can do

- **Resume / cover letter / job description review** — attach (web) or
  point it at (command line) a `.pdf`, `.docx`, `.csv`, `.txt`, or `.md`
  file and ask it to review, tailor, or compare documents against each other.
- **Job application tracker** — company, role, status (saved / applied /
  phone screen / interviewing / offer / rejected / withdrawn), and notes.
- **Interview & company research notes** — save and search notes on
  recruiters, interview questions, and company research.
- **Offer comparison calculator** — quick arithmetic for converting salary
  to hourly, computing a raise percentage, etc.

It has no live web search, so its knowledge of companies and salary ranges
can be out of date. It says so when that matters.

## Configuration

Three settings choose the AI provider. Set them as environment variables, in
a `.env` file (copy `.env.example`), or in Streamlit Secrets:

| Setting | Required | Default |
|---|---|---|
| `LLM_API_KEY` | Yes | — |
| `LLM_BASE_URL` | No | `https://generativelanguage.googleapis.com/v1beta/openai/` (Gemini) |
| `LLM_MODEL` | No | `gemini-3.8-flash` |

Never put your key in the code or commit it. `.env` and
`.streamlit/secrets.toml` are gitignored.

### Get a free Gemini API key

1. Go to Google AI Studio at https://aistudio.google.com/apikey and sign in
   with a Google account.
2. Click **Create API key** and copy it.
3. Use it as `LLM_API_KEY`. That's all you need, since the other two
   settings default to Gemini.

The free tier has per-minute and per-day request limits (shown in AI Studio).
When they run out, the app tells users "The free AI quota is used up, please
try again later." **On the free tier, Google may use prompts and uploaded
documents to improve its products.** Keep that in mind, and see
[Privacy](#privacy).

### Switch to Groq

Create a free key at https://console.groq.com/keys, then set:

```
LLM_API_KEY=<your Groq key>
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
```

### Switch to a local Ollama model

Install Ollama (https://ollama.com), pull a model that supports tool
calling (for example `ollama pull qwen3`), then set:

```
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3
```

Ollama ignores the key, but the app requires a non-empty value. This runs
entirely on your own machine, so nothing is sent to a third party. It only
works where the app runs, though, so it isn't usable on Streamlit Community
Cloud.

The model you choose must support tool (function) calling. Otherwise it
can't use the tracker, notes, or calculator.

## Web app

Visitors chat in the browser and attach their resume, cover letter, or job
description with the upload box. Each visitor gets their own private tracker
and notes. These last as long as the browser tab and are never written to the
server's disk. A **Download backup** button saves them to a file, and visitors
can load that file on a later visit to pick up where they left off.

### Run it locally

```bash
pip install -r requirements.txt
cp .env.example .env   # then paste your key into .env
streamlit run app.py
```

### Put it online (Streamlit Community Cloud, free)

1. Push this repository to GitHub (it can be public or private).
2. Sign in at https://share.streamlit.io with your GitHub account and choose
   **Create app** → pick this repository, branch `main`, main file `app.py`.
3. Under **Advanced settings → Secrets**, add your key (and, optionally, the
   other settings):

   ```toml
   LLM_API_KEY = "your-gemini-key"
   # Optional — only to use something other than Gemini's defaults:
   # LLM_BASE_URL = "https://api.groq.com/openai/v1"
   # LLM_MODEL = "llama-3.3-70b-versatile"
   ```

4. Click **Deploy**. You'll get a public `*.streamlit.app` link to share.

To change secrets later, open the app's **Settings → Secrets** on
share.streamlit.io. The app restarts with the new values.

Any host that can run a Python web process works too, such as Render,
Railway, Fly.io, or a VM. Install `requirements.txt`, set the settings as
environment variables, and run
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

### Limits — read before sharing the link

Every visitor's messages use **your** key's quota. On a free tier that
quota is small, and on a paid key it costs money. To keep usage under
control:

- `DAILY_MESSAGE_LIMIT` (default `500`): total messages per day across all
  visitors. After that, the site asks people to come back tomorrow. The
  count resets at midnight server time and whenever the app restarts.
- `MAX_MESSAGES_PER_SESSION` (default `30`): messages per browser session.
  A visitor can get around it by refreshing the page, so it's a speed bump,
  not a wall.
- `ACCESS_CODE` (optional): if set, visitors must enter this code before
  chatting. Use it to share only with a class, a club, or a cohort.
- Each message allows at most 15 tool calls, so a confused model can't loop
  forever.

If you move to a paid key, set a spending cap in your provider's console.
That's the only hard limit.

### Privacy

Messages and attached files go to the configured AI provider (Google Gemini
by default) to generate replies. The app shows visitors a notice saying so
and asks them not to upload sensitive personal information. The app itself
doesn't store chats, files, trackers, or notes.

## Command-line app

### Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Put your key in a `.env` file (see [Configuration](#configuration)):

   ```bash
   cp .env.example .env
   # then edit .env and paste your key in
   ```

3. Run it:

   ```bash
   python agent.py
   ```

### Example session

```
you> read ~/Documents/resume.pdf and review it against this job description:
     [paste the JD]
agent> [reads the resume, then gives concrete, specific feedback]

you> add an application for Backend Engineer at Acme Corp, referred by Jane
agent> Added application #0: Backend Engineer at Acme Corp (applied).

you> what's a $95,000 salary as an hourly rate?
agent> $45.67/hour (assuming a 2,080-hour work year).
```

Type `exit` or `quit` to stop. Conversation history is kept in memory for the
session but not saved to disk; applications and notes persist in `data/`
between runs.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use a fake AI client, so they need no API key or network.

## Files

| File | Purpose |
|---|---|
| `app.py` | Web app (Streamlit): chat page, uploads, per-visitor tracker, usage limits |
| `agent.py` | Command-line chat loop, plus the tool-calling loop shared with the web app |
| `tools.py` | Tool implementations (documents, application tracker, notes, calculator) and the `@tool` decorator |
| `tests/` | Tests for the tools, the agent loop, and the web app |
| `.streamlit/config.toml` | Web app settings (5 MB upload limit) |
| `data/` | The command-line app's JSON storage for applications and notes (created on first run) |

## Notes

- It doesn't apply to jobs or contact anyone on a user's behalf. It reviews,
  tracks, and advises.
- Your `data/` (applications and notes), `.env`, and
  `.streamlit/secrets.toml` files are gitignored.
