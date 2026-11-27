"""Regression tests for the submission gate, using synthetic reviewer decisions."""
import csv
import json

import pytest

from eval.promotion import FIELDS, ROOT, digest, gate


REPORT = ROOT / "reports/promotion-candidate.json"


@pytest.fixture
def rows():
    findings = json.loads(REPORT.read_text(encoding="utf-8"))["findings"]
    checksum = digest(REPORT)
    return [dict(id=f["id"], decision="true", duplicate_of="", severity=f["severity"],
                 reason="Synthetic test fixture", report_hash=checksum) for f in findings]


def evaluate(rows, tmp_path):
    output = tmp_path / "review.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return gate(REPORT, output)


def test_complete_passing_fixture(rows, tmp_path):
    assert evaluate(rows, tmp_path)


@pytest.mark.parametrize("field,value", [("report_hash", "stale"), ("decision", ""), ("reason", "")])
def test_incomplete_or_stale_review(rows, tmp_path, field, value):
    rows[0][field] = value
    with pytest.raises(ValueError):
        evaluate(rows, tmp_path)


def test_missing_row(rows, tmp_path):
    with pytest.raises(ValueError):
        evaluate(rows[1:], tmp_path)


def test_two_duplicates_fail_even_at_95_percent(rows, tmp_path):
    for row in rows[:2]:
        row.update(decision="dup", duplicate_of=rows[-1]["id"])
    assert not evaluate(rows, tmp_path)


def test_duplicate_must_reference_a_confirmed_finding(rows, tmp_path):
    rows[0].update(decision="dup", duplicate_of=rows[0]["id"])
    with pytest.raises(ValueError):
        evaluate(rows, tmp_path)
