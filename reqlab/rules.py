"""Детерминированные проверки без LLM.

Они дешёвые и воспроизводимые: ловят заглушки, незаполненные служебные поля
и размытые слова. Их результаты идут в отчёт отдельным блоком и подсказками
передаются модели.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .loader import Document

VAGUE = [
    r"быстр\w*", r"удобн\w*", r"своевременн\w*", r"оптимальн\w*", r"эффективн\w*",
    r"по возможности", r"при необходимости", r"как правило", r"и т\.\s?[дп]\.?",
    r"и прочее", r"примерно", r"около", r"достаточн\w*", r"интуитивн\w*", r"гибк\w*",
    r"современн\w*", r"надёжн\w*", r"надежн\w*", r"корректн\w*", r"понятн\w*",
    r"user-friendly", r"asap", r"т\.\s?п\.",
]
_VAGUE_RE = re.compile(r"(?<![\w-])(" + "|".join(VAGUE) + r")(?![\w-])", re.I)
_PLACEHOLDER_RE = re.compile(
    r"\bTBD\b|\bTODO\b|\bуточнить\b|\bзаполняются\b|\bне заполнен\w*|:\s*—\s*\.?\s*(\||$)|\|\s*—\s*\|",
    re.I)
_CHECKBOX_RE = re.compile(r"Да\s*/\s*Нет")
_EMPTY_ROW_RE = re.compile(r"^\s*\|(\s*\|)+\s*$")
_VERSION_RE = re.compile(r"(верси[яи]|version|ред\.)\s*[:№]?\s*v?\d", re.I)
_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}")


@dataclass
class RuleHit:
    rule: str
    criterion: str
    severity: str
    line_start: int
    line_end: int
    quote: str
    problem: str
    recommendation: str
    related_line: tuple[int, str] | None = None


_NAME_COLS = ("parameter", "параметр", "поле", "field", "name")
_TYPE_COLS = ("format", "формат", "type", "тип")


def _cells(line: str) -> list[str]:
    return [c.strip().strip("`").strip() for c in line.strip().strip("|").split("|")]


def _type_mismatches(doc: Document) -> list[RuleHit]:
    """Один и тот же параметр в разных таблицах (запрос, ответ) объявлен с разным типом."""
    seen: dict[str, tuple[str, int]] = {}
    hits: list[RuleHit] = []
    name_i = type_i = None
    fence = ""
    for i, line in enumerate(doc.lines, 1):
        s = line.strip()
        marker = re.match(r"^`{3,}|^~{3,}", s)
        if marker:
            if not fence:
                fence = marker.group()
            elif s.startswith(fence) and not s.strip(fence[0]):
                fence = ""
            name_i = type_i = None
            continue
        if fence or not s.startswith("|"):
            name_i = type_i = None
            continue
        if _SEPARATOR_RE.match(s):
            continue
        cells = _cells(s)
        low = [c.lower() for c in cells]
        ni = next((k for k, c in enumerate(low) if c in _NAME_COLS), None)
        ti = next((k for k, c in enumerate(low) if c in _TYPE_COLS), None)
        if ni is not None and ti is not None:  # строка заголовка новой таблицы
            name_i, type_i = ni, ti
            continue
        if name_i is None or max(name_i, type_i) >= len(cells):
            continue
        name, typ = cells[name_i], cells[type_i].lower()
        if not name or not typ:
            continue
        if name in seen and seen[name][0] != typ:
            first_typ, first_line = seen[name]
            hits.append(RuleHit(
                "type-mismatch", "S-CON", "major", i, i, s,
                f"Параметр `{name}` объявлен с разными типами: «{first_typ}» в строке "
                f"{first_line} и «{typ}» в строке {i}.",
                f"Согласовать представление поля `{name}` или описать преобразование.",
                related_line=(first_line, doc.lines[first_line - 1].strip())))
        seen.setdefault(name, (typ, i))
    return hits


_SCREEN = re.compile(r"на (?:этом )?экране\b", re.I)
_BUTTON = re.compile(r'кнопк[аиу]\s+[«"]([^»"]+)[»"]', re.I)
_CLICK = re.compile(r'нажима(?:ет|ют|ть)\s+(?:кнопку\s+)?[«"]([^»"]+)[»"]', re.I)


def _acceptance_buttons(doc: Document) -> list[RuleHit]:
    """Compare explicit button names only when the document describes one screen."""
    screens = []
    actions = []
    acceptance_level = 0
    fence = ""
    for number, line in enumerate(doc.lines, 1):
        text = line.strip()
        marker = re.match(r"^`{3,}|^~{3,}", text)
        if marker:
            if not fence:
                fence = marker.group()
            elif text.startswith(fence) and not text.strip(fence[0]):
                fence = ""
            continue
        if fence:
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)", text)
        if heading:
            level = len(heading[1])
            if level <= acceptance_level:
                acceptance_level = 0
            if re.fullmatch(r"критерии при[её]мки", heading[2], re.I):
                acceptance_level = level
        elif acceptance_level:
            actions.extend((number, text, name) for name in _CLICK.findall(text))
        elif _SCREEN.search(text):
            screens.append((number, text))

    if len(screens) != 1:
        return []
    screen_line, description = screens[0]
    buttons = _BUTTON.findall(description)
    if not buttons:
        return []
    hits = []
    known = {name.casefold() for name in buttons}
    for number, text, name in actions:
        if name.casefold() not in known:
            hits.append(RuleHit(
                "acceptance-button", "S-TRC", "minor", number, number, text,
                f"Критерий приёмки требует нажать «{name}». В описании экрана указана "
                f"кнопка «{'», «'.join(buttons)}». Уточните, какой элемент выполняет действие.",
                "Согласовать подпись элемента или указать переход на другой экран.",
                related_line=(screen_line, description)))
    return hits


def run(doc: Document) -> list[RuleHit]:
    hits: list[RuleHit] = []
    checkbox_lines: list[int] = []
    in_code = False
    for i, line in enumerate(doc.lines, 1):
        s = line.strip()
        if s.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not s or _SEPARATOR_RE.match(s):
            continue
        if _CHECKBOX_RE.search(s):
            checkbox_lines.append(i)
            continue
        for m in _VAGUE_RE.finditer(s):
            hits.append(RuleHit(
                "vague-word", "UNA", "minor", i, i, s,
                f"Размытое слово «{m.group(0)}» без измеримого критерия.",
                "Заменить на конкретное значение: число, единицу измерения, условие."))
        # «—» в многоколоночной таблице обычно значит «неприменимо», а в таблице
        # «поле | значение» — незаполненное поле.
        wide_table = s.startswith("|") and s.count("|") > 3
        if _PLACEHOLDER_RE.search(s) and not (wide_table and "—" in s and not
                                              re.search(r"TBD|TODO|уточнить|заполн", s, re.I)):
            # Имя поля в тексте находки: без него заглушки в разных полях выглядят
            # одинаково и сливаются при поиске дублей.
            field_name = _cells(s)[0].strip("*").strip() if s.startswith("|") else ""
            what = f"Поле «{field_name}» не заполнено" if field_name else "Раздел не заполнен"
            hits.append(RuleHit(
                "placeholder", "S-CMP", "major", i, i, s,
                f"{what} (заглушка/прочерк): «{s.strip('|').strip()[:80]}».",
                "Заполнить раздел или явно указать «не применимо» с обоснованием."))
        if _EMPTY_ROW_RE.match(s):
            hits.append(RuleHit(
                "empty-row", "DOC", "major", i, i, s,
                "Пустая строка таблицы: данные не внесены.",
                "Заполнить таблицу (например, ответственных и согласующих)."))
    hits += _acceptance_buttons(doc)
    hits += _type_mismatches(doc)
    if checkbox_lines:
        a, b = checkbox_lines[0], checkbox_lines[-1]
        hits.append(RuleHit(
            "unfilled-checklist", "DOC", "major", a, a, doc.lines[a - 1].strip(),
            f"Чек-лист ревью не заполнен: {len(checkbox_lines)} пунктов с вариантами "
            f"«Да / Нет» без выбранного значения (строки {a}–{b}).",
            "Пройти чек-лист ревью и проставить отметки до передачи в разработку."))
    if not _VERSION_RE.search(doc.text):
        # Строка «версия не заполнена», если она есть, — лучшая цитата; отдельная
        # находка-заглушка для неё не нужна, иначе одна проблема попадёт в отчёт дважды.
        vline = next((h.line_start for h in hits
                      if h.rule == "placeholder" and re.search(r"верси", h.quote, re.I)), 1)
        hits[:] = [h for h in hits if not (h.rule == "placeholder" and h.line_start == vline)]
        hits.append(RuleHit(
            "no-version", "S-MOD", "major", vline, vline, doc.lines[vline - 1].strip(),
            "В документе не указаны номер версии и дата.",
            "Добавить версию, дату и историю изменений документа."))
    return hits
