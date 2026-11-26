"""Привязка цитат модели к тексту документа.

Модель может ошибиться в номере строки или «пересказать» цитату. Здесь мы
сами ищем цитату в документе и вычисляем строки. Находка, цитату которой не
удалось найти, считается неподтверждённой и в отчёт не попадает.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .loader import Document

_TRANSLATE = str.maketrans({
    "«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "'": '"', "’": '"',
    "—": "-", "–": "-", "−": "-", "ё": "е", " ": " ", " ": " ",
})
_STRIP = re.compile(r"[*`_#>|]")
_WS = re.compile(r"\s+")


def normalize(s: str) -> str:
    s = s.lower().translate(_TRANSLATE)
    s = _STRIP.sub(" ", s)
    return _WS.sub(" ", s).strip()


@dataclass
class Match:
    start: int  # 1-based строка
    end: int
    exact: bool


class Locator:
    def __init__(self, doc: Document):
        self.doc = doc
        parts: list[str] = []
        owners: list[int] = []
        for i, line in enumerate(doc.lines, 1):
            n = normalize(line)
            if not n:
                continue
            if parts:
                parts.append(" ")
                owners.append(i)
            parts.append(n)
            owners.extend([i] * len(n))
        self.text = "".join(parts)
        self.owner = owners  # индекс символа -> номер строки
        self.norm_lines = [normalize(ln) for ln in doc.lines]

    def find(self, quote: str, hint: int | None = None) -> Match | None:
        q = normalize(quote.strip().strip('"').strip())
        if len(q) < 3:
            return None
        m = self._exact(q, hint)
        if m:
            return m
        # Модель могла склеить фрагменты через «...»: ищем самый длинный кусок.
        pieces = sorted((p.strip() for p in re.split(r"\.\.\.|…", q)), key=len, reverse=True)
        for piece in pieces:
            if len(piece) >= 12:
                m = self._exact(piece, hint)
                if m:
                    return Match(m.start, m.end, exact=False)
        return self._fuzzy(q, hint)

    def _exact(self, q: str, hint: int | None) -> Match | None:
        hits = [m.start() for m in re.finditer(re.escape(q), self.text)]
        if not hits:
            return None
        if hint:  # при нескольких совпадениях берём ближайшее к подсказке модели
            hits.sort(key=lambda pos: abs(self.owner[pos] - hint))
        pos = hits[0]
        return Match(self.owner[pos], self.owner[pos + len(q) - 1], exact=True)

    def _fuzzy(self, q: str, hint: int | None) -> Match | None:
        best, span = 0.0, None
        n = len(self.norm_lines)
        for i in range(n):
            if not self.norm_lines[i]:
                continue
            for width in (1, 2, 3):
                j = min(n, i + width)
                window = " ".join(x for x in self.norm_lines[i:j] if x)
                ratio = difflib.SequenceMatcher(None, q, window, autojunk=False).ratio()
                if hint and abs(i + 1 - hint) <= 2:
                    ratio += 0.03
                if ratio > best:
                    best, span = ratio, (i + 1, j)
        if span and best >= 0.75:
            return Match(span[0], span[1], exact=False)
        return None
