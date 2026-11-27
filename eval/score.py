"""Оценка отчёта агента по эталонному списку дефектов.

    python eval/score.py reports/report.json
    python eval/score.py old.json new.json          # сравнить два прогона
    python eval/score.py reports/report.json --min-recall 0.8   # для CI: код 1, если ниже

Находка засчитывается за эталонный дефект, если её цитата попадает в строки
дефекта (допуск ±1) и текст находки подходит под все группы ключевых слов.
Находки, не попавшие ни в один дефект, выводятся отдельно: это либо ложные
срабатывания, либо дефекты, которых нет в эталоне (тогда эталон стоит дополнить).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

WEIGHT = {"critical": 3, "major": 2, "minor": 1}
SEV_RU = {"critical": "критично", "major": "существенно", "minor": "незначительно"}
DEFAULT_GOLDEN = Path(__file__).with_name("golden.json")
TOLERANCE = 1


def _norm(s: str) -> str:
    return (s or "").lower().replace("ё", "е")


@dataclass
class Score:
    report: str
    found: dict[str, list[str]] = field(default_factory=dict)   # дефект -> находки
    missed: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)              # находки вне эталона
    severity_ok: dict[str, bool] = field(default_factory=dict)
    recall: float = 0.0
    weighted_recall: float = 0.0
    critical_recall: float = 0.0
    matched_share: float = 0.0
    findings_total: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return self.__dict__


def _lines(f: dict) -> set[int]:
    out: set[int] = set()
    for e in f.get("evidence") or []:
        a, b = int(e.get("line_start") or 0), int(e.get("line_end") or e.get("line_start") or 0)
        out.update(range(a - TOLERANCE, max(a, b) + TOLERANCE + 1))
    return out


def matches(defect: dict, finding: dict) -> bool:
    if defect.get("lines") is not None and not (_lines(finding) & set(defect["lines"])):
        return False
    # Только суть находки: рекомендация часто упоминает посторонние термины
    # (например, «GET /getLimit» в совете добавить метод чтения).
    text = _norm(" ".join(str(finding.get(k) or "") for k in ("problem", "question")))
    return all(re.search(_norm(g), text) for g in defect.get("keywords") or [])


def score(report: dict, golden: dict, name: str = "") -> Score:
    s = Score(report=name)
    sha = (report.get("document") or {}).get("sha256")
    if golden.get("sha256") and sha and sha != golden["sha256"]:
        s.warnings.append("SHA-256 документа в отчёте не совпадает с эталоном: строки могли сдвинуться.")
    findings = report.get("findings") or []
    s.findings_total = len(findings)
    used: set[str] = set()
    for d in golden["defects"]:
        hit = [f for f in findings if matches(d, f)]
        if hit:
            s.found[d["id"]] = [f["id"] for f in hit]
            used.update(f["id"] for f in hit)
            s.severity_ok[d["id"]] = any(f.get("severity") == d["severity"] for f in hit)
        else:
            s.missed.append(d["id"])
    s.extra = [f["id"] for f in findings if f["id"] not in used]

    defects = golden["defects"]
    total_w = sum(WEIGHT[d["severity"]] for d in defects)
    found_w = sum(WEIGHT[d["severity"]] for d in defects if d["id"] in s.found)
    crit = [d for d in defects if d["severity"] == "critical"]
    s.recall = len(s.found) / len(defects)
    s.weighted_recall = found_w / total_w
    s.critical_recall = (sum(d["id"] in s.found for d in crit) / len(crit)) if crit else 1.0
    s.matched_share = (len(used) / len(findings)) if findings else 0.0
    return s


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def render(scores: list[Score], golden: dict, reports: list[dict], verbose: bool) -> str:
    out =["# Оценка отчёта по эталону", ""]
    head = "| Показатель | " + " | ".join(Path(s.report).name for s in scores) + " |"
    out += [head, "| :--- |" + " ---: |" * len(scores)]
    n = len(golden["defects"])
    rows = [
        ("Найдено дефектов эталона", [f"{len(s.found)} из {n} ({_pct(s.recall)})" for s in scores]),
        ("Взвешенная полнота (3/2/1)", [_pct(s.weighted_recall) for s in scores]),
        ("Найдено критичных", [_pct(s.critical_recall) for s in scores]),
        ("Серьёзность совпала", [f"{sum(s.severity_ok.values())} из {len(s.found)}" for s in scores]),
        ("Находок в отчёте", [str(s.findings_total) for s in scores]),
        ("Находок вне эталона", [f"{len(s.extra)} ({_pct(1 - s.matched_share)})" for s in scores]),
    ]
    out += [f"| {k} | " + " | ".join(v) + " |" for k, v in rows]
    for s in scores:
        out += [f"> ⚠️ {Path(s.report).name}: {w}" for w in s.warnings]

    out += ["", "## Дефекты эталона", "",
            "| ID | Серьёзность | Вид | Дефект | " + " | ".join(Path(s.report).name for s in scores) + " |",
            "| :--- | :--- | :--- | :--- |" + " :--- |" * len(scores)]
    for d in golden["defects"]:
        cells = []
        for s in scores:
            if d["id"] in s.found:
                mark = "✅" if s.severity_ok[d["id"]] else "✅ (другая серьёзность)"
                cells.append(f"{mark} {', '.join(s.found[d['id']])}")
            else:
                cells.append("❌")
        out.append(f"| {d['id']} | {SEV_RU[d['severity']]} | {d['kind']} | {d['title']} | "
                   + " | ".join(cells) + " |")

    if verbose:
        for s, rep in zip(scores, reports):
            fmap = {f["id"]: f for f in rep.get("findings") or []}
            out += ["", f"## Находки вне эталона: {Path(s.report).name}", "",
                    "Проверьте вручную: ложное срабатывание или пропуск в эталоне.", ""]
            for fid in s.extra:
                f = fmap[fid]
                ln = ", ".join(str(e.get("line_start")) for e in f.get("evidence") or [])
                out.append(f"- {fid} ({SEV_RU.get(f.get('severity'), f.get('severity'))}, "
                           f"строки {ln}): {str(f.get('problem'))[:160]}")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Оценка отчёта RequirementLab по эталону дефектов")
    ap.add_argument("reports", nargs="+", help="report.json одного или нескольких прогонов")
    ap.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    ap.add_argument("--min-recall", type=float, default=None,
                    help="порог взвешенной полноты для CI (0..1)")
    ap.add_argument("--json", dest="json_out", help="записать результат в JSON")
    ap.add_argument("-v", "--verbose", action="store_true", help="показать находки вне эталона")
    a = ap.parse_args(argv)

    golden = json.loads(Path(a.golden).read_text(encoding="utf-8"))
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in a.reports]
    scores = [score(r, golden, p) for r, p in zip(reports, a.reports)]
    print(render(scores, golden, reports, a.verbose))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps([s.to_json() for s in scores], ensure_ascii=False,
                                               indent=2), encoding="utf-8")
    if a.min_recall is not None and any(s.weighted_recall < a.min_recall for s in scores):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
