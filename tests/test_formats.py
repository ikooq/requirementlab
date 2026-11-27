"""Загрузка PDF и DOCX с настоящими таблицами: то, что выдают Word и экспорт в PDF."""

from __future__ import annotations

from pathlib import Path

import pytest

from reqlab import loader, rules
from reqlab.grounding import Locator
from reqlab.loader import load

HTML = (
    "<h1>Уведомление о расходах</h1>"
    "<p><b>Ф-1.</b> Держатель карты задаёт порог от 1 000 до 1 000 000 ₸.</p>"
    "<table>"
    "<tr><td>Parameter</td><td>Format</td><td>Description</td></tr>"
    "<tr><td>amount</td><td>integer</td><td>Порог в тенге, целое число без дробной части</td></tr>"
    "</table>"
    "<table><tr><td>Поле</td><td>Значение</td></tr><tr><td>Swagger</td><td>—</td></tr></table>"
    "<p>Блок согласующих</p>"
    "<table><tr><td>Подразделение</td><td>ФИО</td></tr><tr><td>&nbsp;</td><td>&nbsp;</td></tr></table>"
)
CSS = "* {font-family: sans-serif; font-size: 10pt} td {border: 1px solid black; padding: 2px}"


def _make_pdf(path: Path) -> None:
    pymupdf = pytest.importorskip("pymupdf")
    story = pymupdf.Story(html=HTML, user_css=CSS)
    writer = pymupdf.DocumentWriter(str(path))
    more = True
    while more:
        dev = writer.begin_page(pymupdf.paper_rect("a4"))
        more, _ = story.place(pymupdf.paper_rect("a4") + (40, 40, -40, -40))
        story.draw(dev)
        writer.end_page()
    writer.close()


def test_pdf_tables_become_pipe_rows(tmp_path):
    p = tmp_path / "spec.pdf"
    _make_pdf(p)
    doc = load(p)
    text = doc.text
    # строка таблицы целиком, многословная ячейка не разорвана
    assert "| amount | integer | Порог в тенге, целое число без дробной части |" in text
    assert "| Swagger | — |" in text
    assert all(page == 1 for page in doc.pages.values())
    hits = {h.rule for h in rules.run(doc)}
    assert {"placeholder", "empty-row"} <= hits  # прочерк и пустая таблица согласующих
    m = Locator(doc).find("Держатель карты задаёт порог от 1 000 до 1 000 000 ₸")
    assert m and m.exact


def test_pdf_falls_back_to_plain_text(tmp_path, monkeypatch):
    p = tmp_path / "spec.pdf"
    _make_pdf(p)

    def broken(_):
        raise RuntimeError("не удалось найти таблицы")

    monkeypatch.setattr(loader, "_read_pdf_tables", broken)
    doc = load(p)
    assert "Держатель карты задаёт порог" in doc.text


def test_docx_headings_and_tables(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Уведомление о расходах", level=1)
    d.add_paragraph("Ф-1. Держатель карты задаёт порог.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Подразделение", "ФИО"
    p = tmp_path / "spec.docx"
    d.save(p)

    doc = load(p)
    assert doc.title == "Уведомление о расходах"
    assert doc.lines[0] == "# Уведомление о расходах"
    assert "| Подразделение | ФИО |" in doc.lines
    assert "empty-row" in {h.rule for h in rules.run(doc)}
