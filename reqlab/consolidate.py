"""Apply previously reviewed duplicate pairs without renumbering unrelated findings."""
import argparse
import json
import re
from pathlib import Path

from .loader import load
from .pipeline import RunResult, _unique_evidence
from .report import render, render_brief


def consolidate(result, pairs):
    findings = {f.id: f for f in result.findings}
    if any(a not in findings or b not in findings or b in pairs for a, b in pairs.items()):
        raise ValueError("Неизвестные ID или цепочка дублей")
    def remap(text):
        return re.sub(r"\bF-\d+\b", lambda m: pairs.get(m[0], m[0]), text)
    for old, keep in pairs.items():
        src, dst = findings[old], findings[keep]
        dst.evidence = _unique_evidence(dst.evidence + src.evidence, limit=len(dst.evidence + src.evidence))
        dst.related = list(dict.fromkeys(dst.related + src.related))
        result.dropped.append(dict(id=old, scope=src.scope, criterion=src.criterion, problem=src.problem,
                                   reason=f"Ранее подтверждённый дубль {keep}."))
    result.findings = [f for f in result.findings if f.id not in pairs]
    result.verdict_text = remap(result.verdict_text)
    for fix in result.priority_fixes:
        fix["action"] = remap(fix["action"])
        fix["finding_ids"] = list(dict.fromkeys(pairs.get(i, i) for i in fix["finding_ids"]))
    for req in result.requirements:
        for dropped in pairs:
            f = findings[dropped]
            if req.id == f.scope and not any(x.scope == req.id and x.criterion == f.criterion for x in result.findings):
                req.verdicts[f.criterion] = "unknown"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ("document", "source", "labels", "output"):
        ap.add_argument(key, type=Path)
    args = ap.parse_args()
    doc = load(args.document)
    saved = json.loads(args.source.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if saved["document"]["sha256"] != doc.sha256 or labels["document_sha256"] != doc.sha256:
        ap.error("Другой документ")
    report = RunResult.from_json(saved, doc)
    pairs = {}
    descriptions = {f.id: f.problem for f in report.findings}
    for fid, mark in labels["labels"].items():
        if mark["label"] == "dup":
            dest = mark.get("duplicate_of")
            if descriptions.get(fid) != mark["problem"] or not dest or descriptions.get(dest) != labels["labels"][dest]["problem"]:
                ap.error("Пара дублей не подтверждена для этого отчёта")
            pairs[fid] = dest
    if args.output.suffix != ".md" or args.output.with_suffix(".json").resolve() == args.source.resolve():
        ap.error("Нужен отдельный выходной .md")
    consolidate(report, pairs)
    args.output.write_text(render(report), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(report.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.with_name(args.output.stem + "-brief.md").write_text(render_brief(report), encoding="utf-8")
