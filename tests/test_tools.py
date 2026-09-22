import io

import docx
import pytest
from reportlab.pdfgen import canvas

import tools
from tools import FileStore, MemoryStore, extract_text, make_tracker_tools


def test_tracker_is_private_per_store():
    a, b = make_tracker_tools(MemoryStore()), make_tracker_tools(MemoryStore())
    assert "Added application #0" in a["add_application"].call({"company": "Acme", "role": "Analyst"})
    assert "[interviewing]" in a["update_application"].call({"index": 0, "status": "interviewing"})
    assert a["list_applications"].call({}) == "0. Analyst at Acme [interviewing]"
    assert a["list_applications"].call({"status": "offer"}) == "No applications with status 'offer'."
    assert b["list_applications"].call({}) == "No applications tracked yet."
    assert a["remove_application"].call({"index": 5}) == "Error: no application at index 5."


def test_notes():
    t = make_tracker_tools(MemoryStore())
    t["add_note"].call({"title": "Acme prep", "content": "STAR stories"})
    assert t["search_notes"].call({"query": "star"}) == "0. Acme prep: STAR stories"
    assert t["search_notes"].call({"query": "zzz"}) == "No notes match 'zzz'."


def test_file_store_round_trip(tmp_path):
    t = make_tracker_tools(FileStore(tmp_path / "data"))
    t["add_application"].call({"company": "X", "role": "Y"})
    assert '"company": "X"' in (tmp_path / "data" / "applications.json").read_text()
    assert t["list_applications"].call({}) == "0. Y at X [applied]"


def test_schema_is_openai_function_format():
    schema = make_tracker_tools(MemoryStore())["update_application"].schema
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "update_application"
    assert fn["description"] == "Update a tracked application's status and/or notes."
    params = fn["parameters"]
    assert params["type"] == "object"
    assert params["required"] == ["index"]
    assert params["properties"]["index"]["type"] == "integer"
    assert params["properties"]["status"]["type"] == "string"
    # Multi-line argument descriptions are joined.
    assert params["properties"]["notes"]["description"] == (
        "New notes text, replacing the existing notes. Leave blank to keep unchanged.")


def test_schema_omits_parameters_for_no_arg_tools():
    assert "parameters" not in tools.get_current_datetime.schema["function"]


def test_call_coerces_numbers_and_reports_errors():
    t = make_tracker_tools(MemoryStore())
    t["add_application"].call({"company": "Acme", "role": "Analyst"})
    assert "Updated application #0" in t["update_application"].call({"index": "0", "status": "offer"})
    assert t["update_application"].call({"index": 0, "bogus": 1}).startswith("Error:")
    assert t["update_application"].call({"index": "zero"}).startswith("Error running update_application")
    assert t["add_application"].call({"company": "Acme"}).startswith("Error running add_application")


def test_calculate():
    assert tools.calculate.call({"expression": "95000 / 2080"}).startswith("45.67")
    assert tools.calculate.call({"expression": "__import__('os')"}).startswith("Error")


def test_extract_text_from_uploads():
    pdf = io.BytesIO()
    c = canvas.Canvas(pdf)
    c.drawString(72, 720, "Jane Doe - Data Analyst")
    c.save()
    assert "Jane Doe - Data Analyst" in extract_text("r.pdf", pdf.getvalue())

    d = docx.Document()
    d.add_paragraph("Cover letter body")
    word = io.BytesIO()
    d.save(word)
    assert extract_text("c.docx", word.getvalue()) == "Cover letter body"

    assert extract_text("t.csv", b"a,b\n1,2\n") == "a, b\n1, 2"
    assert extract_text("big.txt", b"x" * 100_050).endswith("if you need more.]")


@pytest.mark.parametrize("name,data", [("x.exe", b"hi"), ("e.txt", b"   "), ("bad.pdf", b"not a pdf")])
def test_extract_text_rejects_bad_files(name, data):
    with pytest.raises(ValueError):
        extract_text(name, data)
