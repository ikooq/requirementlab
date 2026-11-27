"""Проверка ссылок из сохранённого отчёта по исходному документу без API."""
import argparse
import json
from pathlib import Path

from reqlab.grounding import normalize
from reqlab.loader import load


def audit(report, document):
    errors = []
    checked = 0
    if report.get("document", {}).get("sha256") != document.sha256:
        return 0, ["SHA-256 исходного документа отличается от паспорта"]
    for finding in report["findings"]:
        references = finding.get("evidence") or []
        if not references:
            errors.append(f"{finding['id']}: отсутствует цитата")
        for evidence in references:
            start = evidence.get("line_start")
            end = evidence.get("line_end")
            if not (type(start) is int and type(end) is int and 1 <= start <= end <= len(document.lines)):
                errors.append(f"{finding['id']}: неверный диапазон {start}–{end}")
                continue
            source = " ".join(document.lines[start - 1:end])
            raw_quote = (evidence.get("quote") or "").strip()
            quote = normalize(raw_quote)
            if raw_quote and (raw_quote in source or (quote and quote in normalize(source))):
                checked += 1
            else:
                errors.append(f"{finding['id']}: цитата отсутствует в строках {start}–{end}")
    return checked, errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document")
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    count, errors = audit(json.loads(args.report.read_text(encoding="utf-8")), load(args.document))
    print(f"Подтверждено цитат: {count}. Ошибок: {len(errors)}.")
    for error in errors:
        print(error)
    raise SystemExit(bool(errors))
