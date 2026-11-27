"""Несколько независимых проходов проверки и объединение их находок."""

from __future__ import annotations

import json

from reqlab.llm import LLM
from reqlab.loader import load
from reqlab.pipeline import Agent, _merge_set_verdicts, _merge_verdicts

SPEC = """# Док

* **Ф-1.** Клиент переводит от 100 до 500 000 ₸.
* **Ф-2.** Перевод выполняется быстро.
* **Ф-3.** Сервис отвечает за 2 секунды.
"""
OK = {c: "pass" for c in ("NEC", "APP", "UNA", "CMP", "SNG", "FEA", "VER", "COR", "CNF", "TRC")}


class TwoMindsLLM(LLM):
    """Первый проход находит проблему в Ф-1, перепроверка — в Ф-2."""

    provider = "fake"

    def __init__(self):
        super().__init__("fake-1", verbose=False)
        self.prompts: list[str] = []

    def _call(self, system: str, user: str) -> str:
        self.prompts.append(user)
        second = "Независимая перепроверка" in user
        if "нарезать документ" in user:
            reqs = [{"doc_id": f"Ф-{i}", "line_start": 2 + i, "line_end": 2 + i,
                     "quote": q, "summary": "s"} for i, q in enumerate(
                         ["Клиент переводит от 100 до 500 000 ₸.", "Перевод выполняется быстро.",
                          "Сервис отвечает за 2 секунды."], 1)]
            return json.dumps({"requirements": reqs, "acceptance_criteria": []}, ensure_ascii=False)
        if "проверить КАЖДОЕ" in user:
            results = []
            for rid in ("R-01", "R-02", "R-03"):
                if f"{rid} [" not in user.split("Проверь требования:")[-1]:
                    continue
                v, fs = dict(OK), []
                if rid == "R-01" and not second:
                    v["CMP"] = "fail"
                    fs = [{"criterion": "CMP", "severity": "major", "quotes": ["от 100 до 500 000"],
                           "problem": "Нет сценария ошибки при сумме вне диапазона."}]
                if rid == "R-02" and second:
                    v["VER"] = "fail"
                    fs = [{"criterion": "VER", "severity": "critical", "quotes": ["выполняется быстро"],
                           "problem": "«Быстро» нельзя проверить."}]
                results.append({"req_id": rid, "verdicts": v, "findings": fs})
            return json.dumps({"results": results}, ensure_ascii=False)
        if "набор требований ЦЕЛИКОМ" in user:
            status = "fail" if second else "pass"
            return json.dumps({"verdicts": {"S-CON": {"status": status, "comment": f"проход {status}"}},
                               "findings": []}, ensure_ascii=False)
        if "второй ревьюер" in user:
            return json.dumps({"decisions": [], "dev_ready": "no", "test_ready": "no",
                               "verdict": "", "priority_fixes": []})
        return json.dumps({"findings": []})  # противоречия, пропущенные сценарии


def test_two_passes_union_findings_and_verdicts(tmp_path):
    p = tmp_path / "d.md"
    p.write_text(SPEC, encoding="utf-8")
    llm = TwoMindsLLM()
    res = Agent(llm, batch_size=2, passes=2, verbose=False).run(load(p))

    problems = {f.problem for f in res.findings if f.source == "llm"}
    assert problems == {"Нет сценария ошибки при сумме вне диапазона.", "«Быстро» нельзя проверить."}
    r1, r2 = res.requirements[0], res.requirements[1]
    assert r1.verdicts["CMP"] == "fail" and r2.verdicts["VER"] == "fail"
    assert res.set_verdicts["S-CON"]["status"] == "fail"  # худшая оценка из двух проходов

    checks = [u for u in llm.prompts if "проверить КАЖДОЕ" in u]
    first = [u for u in checks if "перепроверка" not in u]
    second = [u for u in checks if "перепроверка" in u]
    # первый проход: пакеты R-01…R-02 и R-03; второй — со сдвигом: R-01 и R-02…R-03
    assert len(first) == 2 and len(second) == 2
    assert "R-01 [" in second[0].split("Проверь требования:")[-1]
    assert "R-02 [" not in second[0].split("Проверь требования:")[-1]


def test_single_pass_prompts_unchanged(tmp_path):
    p = tmp_path / "d.md"
    p.write_text(SPEC, encoding="utf-8")
    llm = TwoMindsLLM()
    Agent(llm, batch_size=2, passes=1, verbose=False).run(load(p))
    assert not any("перепроверка" in u for u in llm.prompts)  # кэш прежних прогонов остаётся годным


def test_merge_helpers():
    assert _merge_verdicts({}, {"VER": "pass"}) == {"VER": "pass"}
    assert _merge_verdicts({"VER": "pass", "CMP": "fail"}, {"VER": "fail", "CMP": "pass"}) == \
        {"VER": "fail", "CMP": "fail"}
    old = {"S-CON": {"status": "partial", "comment": "a"}}
    assert _merge_set_verdicts(old, {"S-CON": {"status": "pass", "comment": "b"}}) == old
    assert _merge_set_verdicts(old, {"S-CON": {"status": "fail", "comment": "c"}})["S-CON"]["comment"] == "c"
