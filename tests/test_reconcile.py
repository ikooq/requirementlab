import pytest

from reqlab.reconcile import partition


def test_partition_preserves_every_candidate_once():
    assert partition({"groups": [[2, 1]], "rejected": [3]}, 3) == ([[2, 1]], [3])
    for reply in ({"groups": [[1]], "rejected": []}, {"groups": [[1, 2]], "rejected": [2]},
                  {"groups": [[True, 2]], "rejected": []}, {"groups": [[1, 9]], "rejected": []}):
        with pytest.raises(ValueError):
            partition(reply, 2)


def test_different_template_fields_and_content_gaps_keep_separate_fixes():
    from reqlab.reconcile import separate_rule_fixes
    def f(line, problem, source="rule"):
        return (None, {"source": source, "criterion": "S-CMP", "problem": problem,
                       "evidence": [{"line_start": line}]})
    candidates = {1: f(20, "Прочерк"), 2: f(25, "Поле паспорта пустое"),
                  3: f(20, "Не заполнено поле владельца"), 4: f(40, "Нет плана восстановления", "llm")}
    assert separate_rule_fixes([1, 2, 3, 4], candidates) == [[3, 1], [2], [4]]
