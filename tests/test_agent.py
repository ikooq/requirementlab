"""Тесты без сети: модель подменяется заранее заданными ответами."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reqlab import rules
from reqlab.grounding import Locator
from reqlab.llm import LLM, parse_json
from reqlab.loader import LoadError, load
from reqlab.pipeline import Agent
from reqlab.report import render

SPEC = """# Перевод между счетами

## Функциональные требования
* **Ф-1.** Клиент переводит от 100 до 500 000 ₸ между своими счетами.
* **Ф-2.** Перевод выполняется быстро.

### Нефункциональные требования
Сервис отвечает в течение 2 секунд.

### Критерии приемки
| Исходные условия | Действие | Ожидаемый результат |
| :--- | :--- | :--- |
| Баланс 1 000 ₸ | Перевести 100 ₸ | Баланс 900 ₸ |

| Подразделение | ФИО |
| :--- | :--- |
| | |
"""


class FakeLLM(LLM):
    provider = "fake"

    def __init__(self):
        super().__init__("fake-1", verbose=False)

    def _call(self, system: str, user: str) -> str:
        if "нарезать документ" in user:
            return json.dumps({
                "requirements": [
                    {"doc_id": "Ф-1", "section": "ФТ", "line_start": 4, "line_end": 4,
                     "quote": "Клиент переводит от 100 до 500 000 ₸ между своими счетами.",
                     "summary": "Перевод между своими счетами", "kind": "functional"},
                    {"doc_id": "**Ф-2**", "section": "ФТ", "line_start": 99, "line_end": 99,
                     "quote": "Перевод выполняется быстро.", "summary": "Скорость перевода"},
                    {"doc_id": None, "section": "НФТ", "line_start": 8, "line_end": 8,
                     "quote": "Сервис отвечает в течение 2 секунд.", "summary": "Время ответа"},
                ],
                "acceptance_criteria": [{"line": 13, "quote": "Перевести 100 ₸", "summary": "AC"}],
                "skipped": [{"section": "Согласующие", "reason": "служебный блок"}],
            }, ensure_ascii=False)
        if "проверить КАЖДОЕ" in user:
            ok = {c: "pass" for c in ("NEC", "APP", "UNA", "CMP", "SNG", "FEA", "VER", "COR", "CNF", "TRC")}
            return json.dumps({"results": [
                {"req_id": "R-01", "verdicts": ok, "findings": []},
                {"req_id": "R-02", "verdicts": {**ok, "UNA": "fail", "VER": "fail"}, "findings": [
                    {"criterion": "VER", "severity": "critical", "quotes": ["выполняется быстро"],
                     "problem": "«Быстро» нельзя проверить.", "recommendation": "Указать 3 с.",
                     "question": "Какое время допустимо?"},
                    {"criterion": "COR", "severity": "major", "quotes": ["этой цитаты нет в документе"],
                     "problem": "Выдуманная цитата.", "recommendation": "", "question": ""},
                ]},
                {"req_id": "R-03", "verdicts": {**ok, "CMP": "fail"}, "findings": []},
            ]}, ensure_ascii=False)
        if "набор требований ЦЕЛИКОМ" in user:
            return json.dumps({"verdicts": {"S-CON": {"status": "fail", "comment": "Есть конфликт"}},
                               "findings": [{"criterion": "S-CMP", "severity": "major",
                                             "quotes": ["Клиент переводит от 100 до 500 000 ₸"],
                                             "related": ["R-01"], "problem": "Нет сценария ошибки.",
                                             "recommendation": "Добавить.", "question": "?"}]},
                              ensure_ascii=False)
        if "найти ПРОТИВОРЕЧИЯ" in user:
            return json.dumps({"findings": []})
        if "второй ревьюер" in user:
            return "```json\n" + json.dumps({
                "decisions": [], "dev_ready": "yes", "test_ready": "yes",
                "verdict": "Есть критичная находка.",
                "priority_fixes": [{"action": "Измеримо описать скорость", "finding_ids": ["F-001"]}],
            }, ensure_ascii=False) + "\n```"
        raise AssertionError("неожиданный промпт")


@pytest.fixture
def spec(tmp_path: Path) -> Path:
    p = tmp_path / "spec.md"
    p.write_text(SPEC, encoding="utf-8")
    return p


def test_gemini_rotates_to_fallback_model(monkeypatch):
    from reqlab import llm as llm_mod

    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    g = llm_mod.GeminiLLM("k", "busy-model", fallbacks=["gone-model", "ok-model"], verbose=False)
    g._pick_model = lambda: None

    def fake_post(url, headers, body):
        if "busy-model" in url:
            raise llm_mod._Retryable("HTTP 503", status=503)
        if "gone-model" in url:
            raise llm_mod.LLMError("HTTP 404 от gemini: no longer available")
        return {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}]}

    g._post = fake_post
    assert g.complete_json("s", "u", step="t") == {"ok": True}
    assert g.model == "ok-model" and g.models_used == ["ok-model"]
    assert g.label == "gemini / ok-model"


def test_minute_rate_limit_waits_instead_of_switching(monkeypatch):
    from reqlab import llm as llm_mod

    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    g = llm_mod.GeminiLLM("k", "main", fallbacks=["other"], verbose=False)
    calls = {"n": 0}

    def fake_post(url, headers, body):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise llm_mod._Retryable("HTTP 429", wait=20, status=429)
        return {"candidates": [{"content": {"parts": [{"text": '{"ok": 1}'}]}}]}

    g._post = fake_post
    assert g.complete_json("s", "u") == {"ok": 1}
    assert g.model == "main"


def test_cache_survives_fallback_rotation(monkeypatch, tmp_path):
    from reqlab import llm as llm_mod

    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    g = llm_mod.GeminiLLM("k", "main", fallbacks=["fb"], cache_dir=tmp_path, verbose=False)
    g._pick_model = lambda: None

    def busy_main(url, headers, body):
        if "main" in url:
            raise llm_mod._Retryable("HTTP 503", status=503)
        return {"candidates": [{"content": {"parts": [{"text": '{"x": 1}'}]}}]}

    g._post = busy_main
    g.complete_json("s", "u1")
    g.complete_json("s", "u2")  # уже на запасной модели

    def no_network(*a, **kw):
        raise AssertionError("ответ должен браться из кэша")

    g2 = llm_mod.GeminiLLM("k", "main", fallbacks=["fb"], cache_dir=tmp_path, verbose=False)
    g2._post = no_network
    assert g2.complete_json("s", "u1") == {"x": 1}
    assert g2.complete_json("s", "u2") == {"x": 1}
    assert g2.calls == 0 and g2.cache_hits == 2
    assert g2.label == "gemini / fb"  # в паспорт попадает модель, которая реально отвечала


def test_parse_json_handles_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Вот ответ: {"a": 2} конец') == {"a": 2}


def test_loader_rejects_unknown_format(tmp_path):
    p = tmp_path / "x.xlsx"
    p.write_bytes(b"1")
    with pytest.raises(LoadError):
        load(p)


def test_locator_finds_quote_despite_markdown(spec):
    doc = load(spec)
    loc = Locator(doc)
    m = loc.find("Ф-1. Клиент переводит от 100 до 500 000 ₸")
    assert m and m.start == 4 and m.exact
    assert loc.find("совершенно другой текст про кредиты и ипотеку") is None


def test_inexact_quote_is_replaced_by_document_text(spec):
    from reqlab.pipeline import doc_text

    doc = load(spec)
    loc = Locator(doc)
    m = loc.find("Ф-1: клиент переводит от 100 до 500000 тенге между счетами")
    assert m and not m.exact and m.start == 4
    assert doc_text(doc, m.start, m.end) == "Ф-1. Клиент переводит от 100 до 500 000 ₸ между своими счетами."


def test_rules_detect_service_gaps(spec):
    hits = {h.rule for h in rules.run(load(spec))}
    assert {"vague-word", "empty-row", "no-version"} <= hits


def test_full_pipeline_and_report(spec):
    doc = load(spec)
    res = Agent(FakeLLM(), agent_version="test", verbose=False).run(doc)

    assert [r.doc_id for r in res.requirements] == ["Ф-1", "Ф-2", None]
    # неверный номер строки от модели исправлен по цитате
    assert res.requirements[1].line_start == 5
    # выдуманная цитата отсеяна
    assert any(d["reason"] == "цитата не найдена в документе" for d in res.dropped)
    assert all(f.evidence for f in res.findings)
    # «fail» без подтверждённой находки снят
    r3 = res.requirements[2]
    assert r3.verdicts["CMP"] == "unknown"
    # модель сказала «yes», но есть критичная находка — решение «нет»
    assert res.dev_ready == "no"

    md = render(res)
    for section in ("## 1. Паспорт прогона", "## 2. Как агент нарезал требования",
                    "## 3. Проверка каждого требования", "## 4. Проверка всего набора требований",
                    "## 5. Сводка", "## 6. Итог"):
        assert section in md
    assert "fake / fake-1" in md
    assert "строка 5" in md
    json.dumps(res.to_json(), ensure_ascii=False)
