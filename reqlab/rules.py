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
            hits.append(RuleHit(
                "placeholder", "S-CMP", "major", i, i, s,
                "Поле или раздел не заполнен (заглушка/прочерк).",
                "Заполнить раздел или явно указать «не применимо» с обоснованием."))
        if _EMPTY_ROW_RE.match(s):
            hits.append(RuleHit(
                "empty-row", "DOC", "major", i, i, s,
                "Пустая строка таблицы: данные не внесены.",
                "Заполнить таблицу (например, ответственных и согласующих)."))
    if checkbox_lines:
        a, b = checkbox_lines[0], checkbox_lines[-1]
        hits.append(RuleHit(
            "unfilled-checklist", "DOC", "major", a, a, doc.lines[a - 1].strip(),
            f"Чек-лист ревью не заполнен: {len(checkbox_lines)} пунктов с вариантами "
            f"«Да / Нет» без выбранного значения (строки {a}–{b}).",
            "Пройти чек-лист ревью и проставить отметки до передачи в разработку."))
    if not _VERSION_RE.search(doc.text):
        hits.append(RuleHit(
            "no-version", "S-MOD", "major", 1, 1, doc.lines[0].strip(),
            "В документе не указаны номер версии и дата.",
            "Добавить версию, дату и историю изменений документа."))
    return hits
