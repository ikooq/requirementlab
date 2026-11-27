"""Выжимка «Главное» в начале отчёта."""

from __future__ import annotations

from reqlab.pipeline import Evidence, Finding
from reqlab.report import _first_sentence, top_findings


def _f(fid: str, sev: str, crit: str, source: str = "llm") -> Finding:
    return Finding(id=fid, scope="SET", criterion=crit, severity=sev, problem=f"Проблема {fid}.",
                   recommendation="", question="", evidence=[Evidence("q", 1, 1, True)],
                   source=source)


def test_top_findings_order_and_filter():
    fs = [_f("F-001", "minor", "S-CON"), _f("F-002", "major", "UNA"), _f("F-003", "major", "S-CON"),
          _f("F-004", "critical", "CMP"), _f("F-005", "critical", "S-CON"),
          _f("F-006", "major", "S-CMP", source="rule")]
    top = [f.id for f in top_findings(fs)]
    # сначала критичные (противоречие раньше неполноты), затем существенные;
    # незначительные и находки правил не попадают
    assert top == ["F-005", "F-004", "F-003", "F-002"]
    assert len(top_findings(fs, limit=2)) == 2


def test_first_sentence_keeps_short_text_whole():
    long = ("Противоречие между требованием Ф-2 и критерием приёмки: месяц определяется по дате "
            "покупки. Подробности ниже.")
    assert _first_sentence(long).endswith("по дате покупки.")
    assert _first_sentence("Ф-1. Коротко.") == "Ф-1. Коротко."
