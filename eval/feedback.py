"""Apply human adjudication labels to a report to the selected JSON report.

Usage: py eval/feedback.py report.json F-003 accepted --note "confirmed in API contract"
"""
import argparse
import json
from pathlib import Path
import tempfile
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("finding_id")
    ap.add_argument("decision", choices=("accepted", "rejected"))
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    path = Path(args.report)
    data = json.loads(path.read_text(encoding="utf-8"))
    finding = next((f for f in data.get("findings", []) if f.get("id") == args.finding_id), None)
    if finding is None:
        ap.error(f"finding {args.finding_id} not found")
    finding["analyst_decision"] = args.decision
    finding["analyst_note"] = args.note
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
    try:
        os.replace(tmp.name, path)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    print(f"Updated {args.finding_id}: {args.decision}")


if __name__ == "__main__":
    main()
