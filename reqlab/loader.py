"""Чтение документа требований: .md, .txt, .pdf, .docx."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED = (".md", ".markdown", ".txt", ".pdf", ".docx")


class LoadError(Exception):
    pass


@dataclass
class Document:
    path: Path
    name: str
    title: str
    lines: list[str]
    sha256: str
    fmt: str
    # номер строки (1-based) -> номер страницы PDF; пусто для текстовых форматов
    pages: dict[int, int] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def numbered(self) -> str:
        """Текст с номерами строк — так модель ссылается на конкретное место."""
        return "\n".join(f"{i:>4}| {line}" for i, line in enumerate(self.lines, 1))

    def location(self, start: int | None, end: int | None = None) -> str:
        if not start:
            return "место не определено"
        end = end or start
        loc = f"строка {start}" if start == end else f"строки {start}–{end}"
        page = self.pages.get(start)
        return f"{loc}, стр. PDF {page}" if page else loc


def load(path: str | Path) -> Document:
    p = Path(path)
    if not p.exists():
        raise LoadError(f"Файл не найден: {p}")
    ext = p.suffix.lower()
    if ext not in SUPPORTED:
        raise LoadError(f"Формат {ext or '(без расширения)'} не поддерживается. "
                        f"Поддерживаются: {', '.join(SUPPORTED)}")
    raw = p.read_bytes()
    pages: dict[int, int] = {}
    if ext == ".pdf":
        lines, pages = _read_pdf(p)
    elif ext == ".docx":
        lines = _read_docx(p)
    else:
        lines = raw.decode("utf-8-sig", errors="replace").splitlines()
    lines = [ln.rstrip() for ln in lines]
    while lines and not lines[-1].strip():
        lines.pop()
    if not any(ln.strip() for ln in lines):
        raise LoadError("В документе нет текста (возможно, это скан без текстового слоя).")
    return Document(path=p, name=p.name, title=_title(lines, p), lines=lines,
                    sha256=hashlib.sha256(raw).hexdigest(), fmt=ext.lstrip("."), pages=pages)


def _title(lines: list[str], p: Path) -> str:
    for ln in lines:
        m = re.match(r"^\s*#\s+(.+)", ln)
        if m:
            return m.group(1).strip()
    for ln in lines:
        if ln.strip():
            return ln.strip()[:120]
    return p.stem


def _read_pdf(p: Path) -> tuple[list[str], dict[int, int]]:
    """PDF: через PyMuPDF с восстановлением таблиц, иначе — простым текстом через pypdf."""
    try:
        import pymupdf  # noqa: F401
    except ImportError:
        return _read_pdf_plain(p)
    try:
        return _read_pdf_tables(p)
    except Exception:  # noqa: BLE001 — нестандартный PDF: не падаем, читаем проще
        return _read_pdf_plain(p)


def _read_pdf_tables(p: Path) -> tuple[list[str], dict[int, int]]:
    """Текст страниц по порядку сверху вниз; таблицы — строками «| a | b |».

    У pypdf ячейки таблицы превращаются в текст без границ, многострочная ячейка
    рвётся на отдельные строки, а пустые ячейки исчезают совсем — пропадают
    незаполненные таблицы (например, блок согласующих). PyMuPDF находит таблицы
    по линиям сетки и отдаёт ячейки целиком, включая пустые.
    """
    import contextlib
    import io

    import pymupdf

    lines: list[str] = []
    pages: dict[int, int] = {}
    with pymupdf.open(str(p)) as pdf:
        for page_no, page in enumerate(pdf, 1):
            items: list[tuple[float, float, list[str]]] = []
            boxes = []
            with contextlib.redirect_stdout(io.StringIO()):  # PyMuPDF печатает советы в stdout
                tables = page.find_tables().tables
            for tab in tables:
                box = pymupdf.Rect(tab.bbox)
                boxes.append(box)
                rows = ["| " + " | ".join(" ".join(str(c or "").split()) for c in row) + " |"
                        for row in tab.extract()]
                items.append((box.y0, box.x0, rows + [""]))
            for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
                centre = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
                if any(centre in b for b in boxes):
                    continue
                items.append((y0, x0, text.splitlines()))
            for _, _, chunk in sorted(items, key=lambda it: (round(it[0], 1), it[1])):
                for ln in chunk:
                    lines.append(ln)
                    pages[len(lines)] = page_no
    return lines, pages


def _read_pdf_plain(p: Path) -> tuple[list[str], dict[int, int]]:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise LoadError("Для PDF установите зависимости: pip install -r requirements.txt") from e
    reader = PdfReader(str(p))
    lines: list[str] = []
    pages: dict[int, int] = {}
    for page_no, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        for ln in text.splitlines():
            lines.append(ln)
            pages[len(lines)] = page_no
    return lines, pages


def _read_docx(p: Path) -> list[str]:
    try:
        import docx  # python-docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as e:  # pragma: no cover
        raise LoadError("Для DOCX установите зависимости: pip install -r requirements.txt") from e
    d = docx.Document(str(p))
    lines: list[str] = []
    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, d)
            text = para.text.strip()
            style = (para.style.name if para.style is not None else "") or ""
            m = re.search(r"(\d)", style) if style.lower().startswith(("heading", "заголовок")) else None
            if text and m:
                text = "#" * int(m.group(1)) + " " + text
            lines.append(text)
        elif tag == "tbl":
            table = Table(child, d)
            for row in table.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    return lines
