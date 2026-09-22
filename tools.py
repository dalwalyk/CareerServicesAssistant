"""Tool implementations for the career services agent.

Each function is a plain Python function decorated with @beta_tool so the
Anthropic tool runner can turn it into a callable tool automatically from its
signature and docstring.

The application tracker and notes tools read and write through a store, so the
same tools can back the local CLI (JSON files in data/) and the web app (one
private in-memory store per visitor). Build a set bound to a store with
make_tracker_tools(); the module-level versions use the CLI's file store.
"""

import ast
import csv
import io
import json
import operator
from datetime import datetime
from pathlib import Path

from anthropic import beta_tool

DATA_DIR = Path(__file__).parent / "data"

MAX_DOCUMENT_CHARS = 100_000


class FileStore:
    """Keeps each collection as a JSON file in a directory (used by the CLI)."""

    def __init__(self, directory: Path):
        self.directory = directory

    def load(self, name: str) -> list:
        path = self.directory / f"{name}.json"
        if not path.exists():
            return []
        return json.loads(path.read_text())

    def save(self, name: str, data: list) -> None:
        self.directory.mkdir(exist_ok=True)
        (self.directory / f"{name}.json").write_text(json.dumps(data, indent=2))


class MemoryStore:
    """Keeps collections in a dict (used by the web app, one per visitor)."""

    def __init__(self, data: dict | None = None):
        self.data = data if data is not None else {}

    def load(self, name: str) -> list:
        return [dict(item) for item in self.data.get(name, [])]

    def save(self, name: str, data: list) -> None:
        self.data[name] = data


# ---------------------------------------------------------------------------
# Document parsing
# ---------------------------------------------------------------------------

def _read_pdf(source) -> str:
    from pypdf import PdfReader

    reader = PdfReader(source)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _read_docx(source) -> str:
    import docx

    doc = docx.Document(source)
    return "\n".join(p.text for p in doc.paragraphs)


def _csv_to_text(f) -> str:
    return "\n".join(", ".join(row) for row in csv.reader(f))


def _read_csv(path: Path) -> str:
    with path.open(newline="") as f:
        return _csv_to_text(f)


def _truncate(text: str) -> str:
    if len(text) <= MAX_DOCUMENT_CHARS:
        return text
    return (
        f"{text[:MAX_DOCUMENT_CHARS]}\n\n"
        f"[Truncated: document is {len(text)} characters, "
        f"only the first {MAX_DOCUMENT_CHARS} were returned. "
        "Ask to read a specific section or page range if you need more.]"
    )


UPLOAD_TYPES = ("pdf", "docx", "csv", "txt", "md")


def extract_text(filename: str, data: bytes) -> str:
    """Extract the text of an uploaded document (web app). Raises ValueError
    for unsupported or unreadable files."""
    suffix = Path(filename).suffix.lower().lstrip(".")
    try:
        if suffix == "pdf":
            text = _read_pdf(io.BytesIO(data))
        elif suffix == "docx":
            text = _read_docx(io.BytesIO(data))
        elif suffix == "csv":
            text = _csv_to_text(io.StringIO(data.decode("utf-8", errors="replace"), newline=""))
        elif suffix in ("txt", "md"):
            text = data.decode("utf-8", errors="replace")
        else:
            raise ValueError(f"unsupported file type '.{suffix}'")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"could not read '{filename}': {exc}") from exc
    if not text.strip():
        raise ValueError(f"'{filename}' contains no extractable text (is it a scanned image?)")
    return _truncate(text)


@beta_tool
def read_document(file_path: str) -> str:
    """Read and extract the text content of a document so it can be summarized,
    reviewed, or compared against a job description — a resume, cover letter,
    job posting, or offer letter.

    Supports .pdf, .docx, .txt, .md, and .csv files.

    Args:
        file_path: Path to the document, absolute or relative to where the
            agent is running.
    """
    path = Path(file_path).expanduser()
    if not path.exists():
        return f"Error: no file found at '{file_path}'."
    if not path.is_file():
        return f"Error: '{file_path}' is a directory, not a file."

    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            text = _read_pdf(str(path))
        elif suffix == ".docx":
            text = _read_docx(str(path))
        elif suffix == ".csv":
            text = _read_csv(path)
        elif suffix in (".txt", ".md", ".json", ".py", ".log"):
            text = path.read_text(errors="replace")
        else:
            return (
                f"Error: unsupported file type '{suffix}'. "
                "Supported types: .pdf, .docx, .csv, .txt, .md."
            )
    except Exception as exc:
        return f"Error reading '{file_path}': {exc}"

    if not text.strip():
        return f"'{path.name}' was read but contains no extractable text."

    return _truncate(text)


@beta_tool
def list_directory(directory_path: str = ".") -> str:
    """List files and subfolders in a directory, so the user can find a
    document to parse.

    Args:
        directory_path: Directory to list. Defaults to the current directory.
    """
    path = Path(directory_path).expanduser()
    if not path.exists():
        return f"Error: no directory found at '{directory_path}'."
    if not path.is_dir():
        return f"Error: '{directory_path}' is a file, not a directory."

    entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    if not entries:
        return f"'{directory_path}' is empty."
    lines = []
    for entry in entries:
        kind = "dir" if entry.is_dir() else "file"
        lines.append(f"[{kind}] {entry.name}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Career services: job application tracker and notes
# ---------------------------------------------------------------------------

APPLICATION_STATUSES = (
    "saved",
    "applied",
    "phone screen",
    "interviewing",
    "offer",
    "rejected",
    "withdrawn",
)


def make_tracker_tools(store) -> dict:
    """Build the application tracker and notes tools bound to `store` (a
    FileStore or MemoryStore). Returns them by name."""

    @beta_tool
    def add_application(company: str, role: str, status: str = "applied", notes: str = "") -> str:
        """Add a job application to the tracker.

        Args:
            company: The company name.
            role: The job title applied for.
            status: One of saved, applied, phone screen, interviewing, offer,
                rejected, withdrawn. Defaults to "applied".
            notes: Any extra context (recruiter name, referral, link, salary range).
        """
        status = status.lower().strip()
        if status not in APPLICATION_STATUSES:
            return f"Error: status must be one of {', '.join(APPLICATION_STATUSES)}."
        applications = store.load("applications")
        applications.append({
            "company": company,
            "role": role,
            "status": status,
            "notes": notes,
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
        })
        store.save("applications", applications)
        return f"Added application #{len(applications) - 1}: {role} at {company} ({status})."

    @beta_tool
    def list_applications(status: str = "") -> str:
        """List tracked job applications, optionally filtered by status.

        Args:
            status: If given, only show applications with this status (one of
                saved, applied, phone screen, interviewing, offer, rejected,
                withdrawn). Leave blank to show all.
        """
        applications = list(enumerate(store.load("applications")))
        if status:
            status = status.lower().strip()
            applications = [(i, a) for i, a in applications if a["status"] == status]
        if not applications:
            return "No applications tracked yet." if not status else f"No applications with status '{status}'."
        lines = []
        for i, a in applications:
            note = f" — {a['notes']}" if a["notes"] else ""
            lines.append(f"{i}. {a['role']} at {a['company']} [{a['status']}]{note}")
        return "\n".join(lines)

    @beta_tool
    def update_application(index: int, status: str = "", notes: str = "") -> str:
        """Update a tracked application's status and/or notes.

        Args:
            index: The index of the application, as shown by list_applications.
            status: New status (one of saved, applied, phone screen, interviewing,
                offer, rejected, withdrawn). Leave blank to keep unchanged.
            notes: New notes text, replacing the existing notes. Leave blank to
                keep unchanged.
        """
        applications = store.load("applications")
        if index < 0 or index >= len(applications):
            return f"Error: no application at index {index}."
        if status:
            status = status.lower().strip()
            if status not in APPLICATION_STATUSES:
                return f"Error: status must be one of {', '.join(APPLICATION_STATUSES)}."
            applications[index]["status"] = status
        if notes:
            applications[index]["notes"] = notes
        applications[index]["updated"] = datetime.now().isoformat()
        store.save("applications", applications)
        a = applications[index]
        return f"Updated application #{index}: {a['role']} at {a['company']} [{a['status']}]."

    @beta_tool
    def remove_application(index: int) -> str:
        """Delete a tracked application entirely.

        Args:
            index: The index of the application, as shown by list_applications.
        """
        applications = store.load("applications")
        if index < 0 or index >= len(applications):
            return f"Error: no application at index {index}."
        removed = applications.pop(index)
        store.save("applications", applications)
        return f"Removed application: {removed['role']} at {removed['company']}."

    @beta_tool
    def add_note(title: str, content: str) -> str:
        """Save a note for later reference — interview prep, company research,
        a recruiter's contact info, salary research, etc.

        Args:
            title: Short title for the note.
            content: The note's body text.
        """
        notes = store.load("notes")
        notes.append({"title": title, "content": content, "created": datetime.now().isoformat()})
        store.save("notes", notes)
        return f"Saved note '{title}'."

    @beta_tool
    def list_notes() -> str:
        """List the titles and creation dates of all saved notes."""
        notes = store.load("notes")
        if not notes:
            return "No notes saved yet."
        lines = [f"{i}. {n['title']} ({n['created']})" for i, n in enumerate(notes)]
        return "\n".join(lines)

    @beta_tool
    def search_notes(query: str) -> str:
        """Search saved notes by title or content.

        Args:
            query: Text to search for, case-insensitive.
        """
        notes = store.load("notes")
        query_lower = query.lower()
        matches = [
            (i, n) for i, n in enumerate(notes)
            if query_lower in n["title"].lower() or query_lower in n["content"].lower()
        ]
        if not matches:
            return f"No notes match '{query}'."
        lines = [f"{i}. {n['title']}: {n['content']}" for i, n in matches]
        return "\n".join(lines)

    return {
        "add_application": add_application,
        "list_applications": list_applications,
        "update_application": update_application,
        "remove_application": remove_application,
        "add_note": add_note,
        "list_notes": list_notes,
        "search_notes": search_notes,
    }


# The CLI's tracker and notes tools, stored as JSON files in data/.
_cli_tracker_tools = make_tracker_tools(FileStore(DATA_DIR))
add_application = _cli_tracker_tools["add_application"]
list_applications = _cli_tracker_tools["list_applications"]
update_application = _cli_tracker_tools["update_application"]
remove_application = _cli_tracker_tools["remove_application"]
add_note = _cli_tracker_tools["add_note"]
list_notes = _cli_tracker_tools["list_notes"]
search_notes = _cli_tracker_tools["search_notes"]


# ---------------------------------------------------------------------------
# Career services: salary/offer calculator and date/time
# ---------------------------------------------------------------------------

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError("expression contains unsupported syntax")


@beta_tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression — useful for comparing offers, e.g.
    converting an annual salary to hourly ("95000 / 2080"), or computing a
    percentage raise ("110000 * 1.08").

    Only numbers and + - * / // % ** and parentheses are supported.

    Args:
        expression: The arithmetic expression to evaluate.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
    except Exception as exc:
        return f"Error: could not evaluate '{expression}': {exc}"
    return str(result)


@beta_tool
def get_current_datetime() -> str:
    """Get the current local date and time."""
    return datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")
