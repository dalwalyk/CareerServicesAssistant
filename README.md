# Career Services Agent

A chatbot with tool use, built to help with a job search: reviewing resumes
and cover letters, tracking applications, prepping for interviews, and
comparing offers. It comes in two forms:

- **Web app** (`app.py`) — a public website anyone can use from a link, no
  install or API key needed on their side. See [Web app](#web-app).
- **Command-line app** (`agent.py`) — runs on your own machine and reads
  files straight from your disk.

## What it can do

- **Resume / cover letter / job description review** — attach (web) or
  point it at (command line) a `.pdf`, `.docx`, `.csv`, `.txt`, or `.md`
  file and ask it to review,
  tailor, or compare documents against each other.
- **Job application tracker** — company, role, status (saved / applied /
  phone screen / interviewing / offer / rejected / withdrawn), and notes.
- **Interview & company research notes** — save and search notes on
  recruiters, interview questions, and company research.
- **Offer comparison calculator** — quick arithmetic for converting salary
  to hourly, computing a raise percentage, etc.
- **Web search** — looks up current information (company news, typical
  salary ranges, role expectations) instead of relying on stale knowledge.

## Web app

Visitors chat in the browser and attach their resume, cover letter, or job
description with the upload box. Each visitor gets their own private tracker
and notes. These last as long as the browser tab and are never written to the
server's disk. A **Download backup** button saves them to a file, and visitors
can load that file on a later visit to pick up where they left off.

### Run it locally

```bash
pip install -r requirements.txt
cp .env.example .env   # then paste your API key into .env
streamlit run app.py
```

### Put it online (Streamlit Community Cloud, free)

1. Push this repository to GitHub (it can be public or private).
2. Sign in at https://share.streamlit.io with your GitHub account and choose
   **Create app** → pick this repository, branch `main`, main file `app.py`.
3. Under **Advanced settings → Secrets**, add your key:

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```

4. Click **Deploy**. You'll get a public `*.streamlit.app` link to share.

Any host that can run a Python web process works too, such as Render,
Railway, Fly.io, or a VM. Install `requirements.txt`, set
`ANTHROPIC_API_KEY` as an environment variable, and run
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

### Costs and limits — read before sharing the link

Every visitor's messages are billed to **your** Anthropic API key. To keep
that under control:

- **Set a monthly spend limit** in the Anthropic Console
  (https://console.anthropic.com → Settings → Limits). This is the only hard
  cap. The limits below are best-effort.
- `DAILY_MESSAGE_LIMIT` (default `500`): total messages per day across all
  visitors. After that, the site asks people to come back tomorrow. The
  count resets at midnight server time and whenever the app restarts.
- `MAX_MESSAGES_PER_SESSION` (default `30`): messages per browser session.
  A visitor can get around it by refreshing the page, so it's a speed bump,
  not a wall.
- `ACCESS_CODE` (optional): if set, visitors must enter this code before
  chatting. Use it to share only with a class, a club, or a cohort.
- `CLAUDE_MODEL`: `claude-opus-5` by default. `claude-sonnet-5` costs less
  per message.

Set any of these as environment variables or in the Streamlit Secrets box,
next to `ANTHROPIC_API_KEY`.

### Privacy

Messages and attached files go to Anthropic's Claude API to generate replies.
The app itself doesn't store chats, files, trackers, or notes. Tell your
users this, and let them decide what to upload.

## Command-line app

### Setup

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

### Example session

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
| `app.py` | Web app (Streamlit): chat page, uploads, per-visitor tracker, usage limits |
| `agent.py` | Command-line chat loop, plus the tool-calling turn shared with the web app |
| `tools.py` | Tool implementations (documents, application tracker, notes, calculator) |
| `.streamlit/config.toml` | Web app settings (5 MB upload limit) |
| `data/` | The command-line app's JSON storage for applications and notes (created on first run) |

## Notes

- Uses `claude-opus-5` by default; set `CLAUDE_MODEL` in `.env` to use a
  different model (e.g. `claude-sonnet-5` for lower cost).
- Documents are only sent to the Claude API as part of the conversation,
  never anywhere else.
- It doesn't apply to jobs or contact anyone on a user's behalf. It reviews,
  tracks, and researches.
- Your `data/` (applications and notes), `.env`, and
  `.streamlit/secrets.toml` files are gitignored.
