"""Оценка отчёта агента по эталонному списку дефектов.

    python eval/score.py reports/report.json
    python eval/score.py old.json new.json          # сравнить два прогона
    python eval/score.py reports/report.json --min-recall 0.8   # для CI: код 1, если ниже
    python eval/score.py reports/report.json --labels eval/labels_case.json  # и точность

Находка засчитывается за эталонный дефект, если её цитата попадает в строки
дефекта (допуск ±1) и текст находки подходит под все группы ключевых слов.
Находки, не попавшие ни в один дефект, выводятся отдельно: это либо ложные
срабатывания, либо дефекты, которых нет в эталоне (тогда эталон стоит дополнить).

Точность считается по ручной разметке (--labels): каждая находка — «верно»,
«ложно» или «дубль». Находка, совпавшая с дефектом эталона, по умолчанию верна;
находка вне эталона без метки считается неразмеченной. Метка привязана к началу
текста находки: после пересборки отчёта старые метки не применяются молча.
"""

from __future__ import annotations

import argparse
import hashlib
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
    labels: dict[str, str] = field(default_factory=dict)        # находка -> true|false|dup
    precision: float | None = None
    f1: float | None = None
    exact_quote_share: float | None = None
    per_severity: dict[str, dict[str, float | int]] = field(default_factory=dict)
    duplicates: list[str] = field(default_factory=list)
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


def score(report: dict, golden: dict, name: str = "", labels: dict | None = None) -> Score:
    s = Score(report=name)
    sha = (report.get("document") or {}).get("sha256")
    if golden.get("sha256") and sha and sha != golden["sha256"]:
        s.warnings.append("SHA-256 документа в отчёте не совпадает с эталоном: строки могли сдвинуться.")
    findings = report.get("findings") or []
    s.findings_total = len(findings)
    quotes = [e for f in findings for e in f.get("evidence", []) if "exact" in e]
    if quotes:
        s.exact_quote_share = sum(e["exact"] is True for e in quotes) / len(quotes)
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
    s.duplicates = [fid for ids in s.found.values() for fid in ids[1:]]

    defects = golden["defects"]
    total_w = sum(WEIGHT[d["severity"]] for d in defects)
    found_w = sum(WEIGHT[d["severity"]] for d in defects if d["id"] in s.found)
    crit = [d for d in defects if d["severity"] == "critical"]
    s.recall = len(s.found) / len(defects)
    s.weighted_recall = found_w / total_w
    s.critical_recall = (sum(d["id"] in s.found for d in crit) / len(crit)) if crit else 1.0
    for sev in WEIGHT:
        ds = [d for d in defects if d["severity"] == sev]
        tp = sum(d["id"] in s.found for d in ds)
        s.per_severity[sev] = {"tp": tp, "fn": len(ds) - tp,
                               "recall": tp / len(ds) if ds else 1.0}
    s.matched_share = (len(used) / len(findings)) if findings else 0.0
    if labels is not None:
        _apply_labels(s, findings, used, labels)
        s.duplicates = [fid for fid, label in s.labels.items() if label == "dup"]
        if "unlabeled" not in s.labels.values() and s.precision is not None:
            def harmonic(p, r):
                return 2 * p * r / (p + r) if p + r else 0.0
            s.f1 = harmonic(s.precision, s.recall)
            for sev, metrics in s.per_severity.items():
                group = [s.labels[f["id"]] for f in findings if f["severity"] == sev]
                if group:
                    metrics["precision"] = group.count("true") / len(group)
                    metrics["f1"] = harmonic(metrics["precision"], metrics["recall"])
    return s


LABELS = ("true", "false", "dup")
_KEY_LEN = 40


def _apply_labels(s: Score, findings: list[dict], used: set[str], labels: dict) -> None:
    marks = labels.get("labels") or {}
    for f in findings:
        m = marks.get(f["id"])
        if m and _norm(m.get("problem", ""))[:_KEY_LEN] != _norm(f.get("problem", ""))[:_KEY_LEN]:
            s.warnings.append(f"метка {f['id']} относится к другой находке (отчёт пересобран?) — пропущена")
            m = None
        if m and m.get("label") in LABELS:
            s.labels[f["id"]] = m["label"]
        elif f["id"] in used:
            s.labels[f["id"]] = "true"
        else:
            s.labels[f["id"]] = "unlabeled"
    vals = list(s.labels.values())
    s.precision = vals.count("true") / len(vals) if vals else None


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
        ("Дубли", [str(len(s.duplicates)) for s in scores]),
        ("Recall по тяжести", [", ".join(f"{k}: {_pct(v['recall'])}" for k,v in s.per_severity.items()) for s in scores]),
    ]
    rows.append(("Дословные цитаты по журналу привязки", [
        _pct(s.exact_quote_share) if s.exact_quote_share is not None else "—" for s in scores]))
    if len(scores) > 1:
        sets = [set(s.found) for s in scores]
        agreed = len(set.intersection(*sets))
        total = len(set.union(*sets))
        out.insert(1, f"Стабильность обнаружения: во всех отчётах {agreed} из {total} дефектов, найденных хотя бы раз.")
    if any(s.precision is not None for s in scores):
        def cnt(s: Score, k: str) -> str:
            return str(list(s.labels.values()).count(k)) if s.precision is not None else "—"
        rows += [
            ("Точность (верные / все находки)",
             [f"{cnt(s, 'true')} из {s.findings_total} ({_pct(s.precision)})"
              if s.precision is not None else "—" for s in scores]),
            ("Ложных / дублей / не размечено",
             [f"{cnt(s, 'false')} / {cnt(s, 'dup')} / {cnt(s, 'unlabeled')}" for s in scores]),
            ("F1 (по размеченным находкам)", [_pct(s.f1 or 0) if s.f1 is not None else "—" for s in scores]),
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
    ap.add_argument("--labels", help="ручная разметка находок (верно/ложно/дубль) для точности")
    ap.add_argument("--holdout", action="store_true", help="использовать только эталон автоплатежа")
    ap.add_argument("--min-critical-recall", type=float, help="минимальная полнота критичных дефектов")
    ap.add_argument("--min-f1", type=float, help="минимальный F1; требует --labels")
    ap.add_argument("--json", dest="json_out", help="записать результат в JSON")
    ap.add_argument("-v", "--verbose", action="store_true", help="показать находки вне эталона")
    a = ap.parse_args(argv)

    path = DEFAULT_GOLDEN.with_name("golden_autopay.json") if a.holdout else Path(a.golden)
    golden = json.loads(path.read_text(encoding="utf-8"))
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in a.reports]
    labels = json.loads(Path(a.labels).read_text(encoding="utf-8")) if a.labels else None
    for report in reports:
        sha = report.get("document", {}).get("sha256")
        if golden.get("sha256") and sha and sha != golden["sha256"]:
            ap.error("Отчёт и эталон относятся к разным документам")
    if labels and labels.get("findings_sha256"):
        for report in reports:
            payload = json.dumps(report["findings"], ensure_ascii=False, sort_keys=True).encode()
            if hashlib.sha256(payload).hexdigest() != labels["findings_sha256"]:
                ap.error("Разметка относится к другой версии находок")
    scores = [score(r, golden, p, labels) for r, p in zip(reports, a.reports)]
    if a.min_f1 is not None and (not a.labels or any(s.f1 is None or s.f1 < a.min_f1 for s in scores)):
        print("F1 ниже порога или отсутствует ручная разметка", file=sys.stderr); return 1
    if a.min_critical_recall is not None and any(s.critical_recall < a.min_critical_recall for s in scores):
        print("Полнота критичных дефектов ниже порога", file=sys.stderr); return 1
    print(render(scores, golden, reports, a.verbose))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps([s.to_json() for s in scores], ensure_ascii=False,
                                               indent=2), encoding="utf-8")
    if a.min_recall is not None and any(s.weighted_recall < a.min_recall for s in scores):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
