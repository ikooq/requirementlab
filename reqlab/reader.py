"""Офлайн-просмотр находок и исходного текста. python -m reqlab.reader DOC REPORT.json OUT.html"""
import argparse
import json
from html import escape
from pathlib import Path

from .criteria import SEVERITY_RU, name
from .loader import load
from .report import READY_RU


def render(document, report):
    if report["document"]["sha256"] != document.sha256:
        raise ValueError("Документ отличается от исходного файла отчёта")
    findings = report["findings"]
    out = ["""<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Проверка требований</title><style>
*{box-sizing:border-box}body{margin:auto;padding:1.5rem;max-width:110rem;color:#192b3a;background:#fafafa;font:1rem/1.5 system-ui,sans-serif}
main{display:grid;grid-template-columns:1fr 1fr;gap:2rem}section{min-width:0}article,.source{padding:1rem;background:white;border:1px solid #cad6dd;margin:.65rem 0;border-radius:.45rem}
a{color:#004f80}li{padding:.2rem;white-space:pre-wrap;overflow-wrap:anywhere}li:target{background:#ffedab;outline:2px solid #875c00}
.source{max-height:76vh;overflow:auto;position:sticky;top:.5rem}h1,h2,h3{line-height:1.2}
@media(max-width:760px){main{display:block}.source{position:static;max-height:none}}
</style>"""]
    out.append(f"<h1>{escape(document.title)}</h1>")
    out.append(f"<p>Требований: {len(report['requirements'])}. Находок: {len(findings)}. "
               f"Разработка: {escape(READY_RU[report['dev_ready']])}. "
               f"Тестирование: {escape(READY_RU[report['test_ready']])}.</p>")
    for gap in report.get("coverage_gaps") or []:
        out.append(f"<p>Проверка неполна: {escape(gap)}</p>")
    out.append("<main><section><h2>Находки</h2>")
    for severity, label in SEVERITY_RU.items():
        group = [f for f in findings if f["severity"] == severity]
        if not group:
            continue
        out.append(f"<h3>{escape(label)} · {len(group)}</h3>")
        for f in group:
            fid = escape(f["id"], quote=True)
            out.append(f'<article id="{fid}"><strong>{fid} · {escape(name(f["criterion"]))}</strong>'
                       f'<p>{escape(f["problem"])}</p>')
            for e in f["evidence"]:
                a, b = int(e["line_start"]), int(e["line_end"])
                if not 1 <= a <= b <= len(document.lines):
                    raise ValueError(f"{fid}: неверные строки")
                out.append(f'<p><a href="#L{a}">Стр. {a}–{b}</a> · {escape(e["quote"])}</p>')
            if f.get("recommendation"):
                out.append(f'<p><b>Как исправить.</b> {escape(f["recommendation"])}</p>')
            out.append("</article>")
    out.append('</section><section><h2>Исходный документ</h2><div class="source"><ol>')
    for number, line in enumerate(document.lines, 1):
        out.append(f'<li id="L{number}">{escape(line)}</li>')
    out.append("</ol></div></section></main></html>")
    return "\n".join(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document")
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    args.output.write_text(render(load(args.document), report), encoding="utf-8")
