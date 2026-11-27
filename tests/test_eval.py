"""Тесты скрипта оценки отчёта по эталону."""

from __future__ import annotations

import json
from pathlib import Path

from eval.score import main, matches, score

GOLDEN = {
    "sha256": "abc",
    "defects": [
        {"id": "G1", "severity": "critical", "kind": "противоречие", "title": "GET для сохранения",
         "lines": [10], "keywords": ["\\bget\\b"]},
        {"id": "G2", "severity": "major", "kind": "оформление", "title": "нет версии",
         "lines": [3], "keywords": ["верси"]},
        {"id": "G3", "severity": "minor", "kind": "неполнота", "title": "нет приоритетов",
         "lines": None, "keywords": ["приоритет"]},
    ],
}


def _f(fid, line, problem, severity="major", recommendation=""):
    return {"id": fid, "severity": severity, "problem": problem, "recommendation": recommendation,
            "question": "", "evidence": [{"line_start": line, "line_end": line}]}


def test_match_needs_location_and_keywords():
    d = GOLDEN["defects"][0]
    assert matches(d, _f("F-1", 10, "Метод объявлен как GET"))
    assert matches(d, _f("F-1", 11, "Метод объявлен как GET"))      # допуск ±1 строка
    assert not matches(d, _f("F-1", 40, "Метод объявлен как GET"))  # не то место
    assert not matches(d, _f("F-1", 10, "Метод не описан"))         # не тот смысл
    # рекомендация не считается: в ней часто посторонние термины
    assert not matches(d, _f("F-1", 10, "Нет метода чтения", recommendation="добавить GET"))


def test_score_counts_recall_severity_and_extra():
    report = {"document": {"sha256": "abc"}, "findings": [
        _f("F-1", 10, "Сохранение через GET", severity="major"),
        _f("F-2", 99, "Не указаны приоритеты требований", severity="minor"),
        _f("F-3", 50, "Что-то вне эталона"),
    ]}
    s = score(report, GOLDEN)
    assert s.found == {"G1": ["F-1"], "G3": ["F-2"]}
    assert s.missed == ["G2"]
    assert s.extra == ["F-3"]
    assert s.severity_ok == {"G1": False, "G3": True}
    assert s.weighted_recall == (3 + 1) / 6
    assert s.critical_recall == 1.0
    assert not s.warnings


def test_sha_mismatch_warns():
    s = score({"document": {"sha256": "other"}, "findings": []}, GOLDEN)
    assert s.warnings and s.recall == 0


def test_cli_min_recall_exit_code(tmp_path: Path, capsys):
    g = tmp_path / "golden.json"
    g.write_text(json.dumps(GOLDEN, ensure_ascii=False), encoding="utf-8")
    r = tmp_path / "report.json"
    r.write_text(json.dumps({"findings": [_f("F-1", 3, "Нет версии документа")]},
                            ensure_ascii=False), encoding="utf-8")
    assert main([str(r), "--golden", str(g), "--min-recall", "0.9"]) == 1
    assert main([str(r), "--golden", str(g), "--min-recall", "0.3"]) == 0
    assert "1 из 3" in capsys.readouterr().out


def test_real_golden_is_valid():
    golden = json.loads((Path(__file__).parent.parent / "eval" / "golden.json")
                        .read_text(encoding="utf-8"))
    ids = [d["id"] for d in golden["defects"]]
    assert len(ids) == len(set(ids))
    for d in golden["defects"]:
        assert d["severity"] in ("critical", "major", "minor")
        assert d["keywords"]
