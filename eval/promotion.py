"""Blind review packet and promotion gate for the organizer case."""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

from eval.evidence import audit
from eval.score import score
from reqlab.loader import load
from reqlab.pipeline import RunResult
from reqlab.report import render

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/requirements.md"
GOLD = ROOT / "eval/golden.json"
FIELDS = ["id", "decision", "duplicate_of", "severity", "reason", "report_hash"]


def digest(path):
    return hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()


def packet(report, output):
    data = json.loads(report.read_text(encoding="utf-8"))
    document = load(DOC)
    _, errors = audit(data, document)
    if errors:
        raise ValueError(errors[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    for f in data["findings"]:
        writer.writerow(dict(id=f["id"], report_hash=digest(report)))
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(DOC, "requirements.md")
        archive.write(report.with_suffix(".md"), "report.md")
        archive.writestr("review.csv", buffer.getvalue().encode("utf-8-sig"))
        archive.writestr("instructions.txt", """Просим аналитика, который не видел промптов агента, независимо сверить отчёт с документом.
Для каждого ID в review.csv заполните decision (true, false или dup), severity (critical, major или minor — ваша оценка) и reason (почему, со ссылкой на строку).
Для dup укажите в duplicate_of ID верного замечания, которое устраняется той же правкой. Проверьте смысл выводов и обе стороны каждого противоречия.
Отдельным текстом перечислите существенные пропуски. Верните CSV и текст с именем или инициалами и подтверждением, что вы не видели промптов.
""".encode("utf-8-sig"))


def gate(report, review):
    data = json.loads(report.read_text(encoding="utf-8"))
    findings = data["findings"]
    ids = {f["id"] for f in findings}
    doc = load(DOC)
    _, errors = audit(data, doc)
    if errors or not ids or len(ids) != len(findings):
        raise ValueError("Непроверенные цитаты, пустые или повторные ID")
    if render(RunResult.from_json(data, doc)).strip() != report.with_suffix(".md").read_text(encoding="utf-8").strip():
        raise ValueError("Markdown не соответствует проверенному JSON")
    with review.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    marks = {row["id"]: row for row in rows}
    if len(rows) != len(ids) or set(marks) != ids:
        raise ValueError("Нужна ровно одна оценка каждой находки")
    checksum = digest(report)
    for row in rows:
        if (row["decision"] not in {"true", "false", "dup"} or row["severity"] not in {"critical", "major", "minor"}
                or not row["reason"].strip() or row["report_hash"] != checksum):
            raise ValueError(f"Неполная или устаревшая оценка {row['id']}")
        if row["decision"] == "dup" and marks.get(row["duplicate_of"], {}).get("decision") != "true":
            raise ValueError(f"Нет подтверждённого оригинала для {row['id']}")
    accepted = [f for f in findings if marks[f["id"]]["decision"] == "true"]
    s = score({**data, "findings": accepted}, json.loads(GOLD.read_text(encoding="utf-8")))
    duplicates = sum(row["decision"] == "dup" for row in rows)
    precision = len(accepted) / len(findings)
    print(f"Эталон: {len(s.found)}/41. Критичные: {s.critical_recall:.0%}. Точность: {precision:.2%}. Дубли: {duplicates}.")
    return len(s.found) >= 37 and s.critical_recall == 1 and precision >= 0.95 and duplicates <= 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["packet", "gate"])
    parser.add_argument("report", type=Path)
    parser.add_argument("file", type=Path, help="выходной ZIP или заполненная CSV-разметка")
    args = parser.parse_args()
    try:
        if args.mode == "packet":
            packet(args.report, args.file)
        elif not gate(args.report, args.file):
            parser.exit(1, "Метрики ниже регрессионного порога.\n")
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, str(error) + "\n")
