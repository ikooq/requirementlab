"""Сборка отчёта в Markdown по структуре п. 6 ТЗ кейса."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path

from .criteria import ALL, CRITICAL_GROUPS, INDIVIDUAL, SET, SEVERITIES, SEVERITY_RU, name
from .pipeline import Finding, RunResult

ICON = {"pass": "✅", "fail": "❌", "na": "➖", "unknown": "❔", "partial": "⚠️"}
SET_RU = {"pass": "выполнено", "partial": "частично", "fail": "нарушено", "unknown": "не оценено"}
READY_RU = {"yes": "Да", "conditional": "С оговорками", "no": "Нет"}
SEV_ICON = {"critical": "🔴", "major": "🟠", "minor": "🟡"}


def _cell(s: str, limit: int = 0) -> str:
    s = " ".join(str(s).split()).replace("|", "\\|")
    if limit and len(s) > limit:
        s = s[: limit - 1].rstrip() + "…"
    return s


# Порядок важности критериев внутри одной серьёзности: противоречие и непроверяемость
# блокируют сильнее, чем неполнота, а она — сильнее оформления.
_CRIT_RANK = ["S-CON", "COR", "VER", "S-VAL", "CMP", "S-CMP", "APP", "FEA", "UNA", "TRC", "S-TRC"]


def top_findings(findings: list[Finding], limit: int = 5) -> list[Finding]:
    """Выжимка для первого экрана: все критичные, затем существенные находки модели
    по важности критерия. Находки правил (оформление) сюда не идут — они в разделе 4.2."""
    def key(f: Finding):
        rank = _CRIT_RANK.index(f.criterion) if f.criterion in _CRIT_RANK else len(_CRIT_RANK)
        return (SEVERITIES.index(f.severity), rank, f.id)
    pool = [f for f in findings if f.source != "rule" and f.severity in ("critical", "major")]
    return sorted(pool, key=key)[:limit]


def _first_sentence(text: str) -> str:
    """Первое предложение описания: в выжимке нужна суть, подробности — в разделах 3–4."""
    s = " ".join(str(text).split())
    for i, ch in enumerate(s):
        if ch in ".!?" and i >= 40 and (i + 1 == len(s) or s[i + 1] == " "):
            return s[:i + 1]
    return s


def _calls(res: RunResult) -> str:
    if res.llm_calls == 0 and res.cache_hits:
        return (f"{res.cache_hits}, все из кэша `.cache/llm`: ответы модели получены в предыдущем "
                "прогоне этого же документа и этих же промптов, отчёт пересобран из них без новых "
                f"запросов ({res.duration_s:.0f} с)")
    return f"{res.llm_calls} (+{res.cache_hits} из кэша), {res.duration_s:.0f} с"


def _head(w, res: RunResult, sev: Counter, limit: int, more: str) -> None:
    doc = res.doc
    w(f"> Итог: разработку начинать — **{READY_RU.get(res.dev_ready, res.dev_ready)}**, "
      f"тестирование — **{READY_RU.get(res.test_ready, res.test_ready)}**. "
      f"Требований: {len(res.requirements)}, нарушений: {len(res.findings)} "
      f"(критичных {sev['critical']}, существенных {sev['major']}, незначительных {sev['minor']}).")
    w("")
    if res.coverage_gaps:
        w("**Проверка неполна.** " + " ".join(res.coverage_gaps))
        w("")
    top = top_findings(res.findings, limit)
    if top:
        w("**Главное — что исправить до старта разработки:**")
        w("")
        for f in top:
            e = f.evidence[0]
            w(f"{SEV_ICON[f.severity]} **{f.id}** ({name(f.criterion)}, "
              f"{doc.location(e.line_start, e.line_end)}) — {_cell(_first_sentence(f.problem), 200)}")
            w("")
        rest = sev["critical"] - sum(1 for f in top if f.severity == "critical")
        if rest > 0:
            w(f"Ещё критичных: {rest} — {more}.")
            w("")


def render_brief(res: RunResult) -> str:
    """Короткая версия на один экран: итог, главные дефекты, счёт по критериям и что исправить.
    Полный отчёт по структуре ТЗ строится функцией render."""
    sev = Counter(f.severity for f in res.findings)
    out: list[str] = []
    w = out.append
    w(f"# Кратко: проверка требований «{res.doc.title}»")
    w("")
    _head(w, res, sev, 7, "см. полный отчёт, раздел 5")
    crit_counter = Counter(f.criterion for f in res.findings)
    w("Нарушения по критериям: " + ", ".join(
        f"{name(code)} — {n}" for code, n in crit_counter.most_common()) + ".")
    w("")
    fixes = res.priority_fixes or [
        {"action": f.recommendation or f.problem, "finding_ids": [f.id]}
        for f in sorted(res.findings, key=lambda x: SEVERITIES.index(x.severity))[:5]]
    w("**Что исправить в первую очередь:**")
    w("")
    for n, fx in enumerate(fixes, 1):
        ids = ", ".join(fx.get("finding_ids") or [])
        w(f"{n}. {_cell(fx['action'])}" + (f" ({ids})" if ids else ""))
    w("")
    w(f"Модель: {res.model}. Агент: {res.agent_version}. Каждая цитата в полном отчёте "
      "проверена по тексту документа; номера находок совпадают с полным отчётом.")
    w("")
    return "\n".join(out) + "\n"


def render(res: RunResult) -> str:
    doc = res.doc
    reqs = res.requirements
    by_scope: dict[str, list[Finding]] = {}
    for f in res.findings:
        by_scope.setdefault(f.scope, []).append(f)
    sev = Counter(f.severity for f in res.findings)
    out: list[str] = []
    w = out.append

    w(f"# Отчёт о проверке требований: «{doc.title}»")
    w("")
    _head(w, res, sev, 5, "см. раздел 5")

    # 1. Паспорт ------------------------------------------------------------
    w("## 1. Паспорт прогона")
    w("")
    w("| Параметр | Значение |")
    w("| :--- | :--- |")
    rows = [
        ("Документ", f"{doc.title} (`{doc.name}`, {doc.fmt.upper()}, {len(doc.lines)} строк)"),
        ("SHA-256 документа", f"`{doc.sha256[:16]}…`"),
        ("Дата прогона", res.started.strftime("%d.%m.%Y %H:%M:%S") + " (UTC"
         + res.started.strftime("%z")[:3] + ":" + res.started.strftime("%z")[3:] + ")"),
        ("Модель", res.model),
        ("Версия агента при формировании данных", res.agent_version),
        ("Оформление Markdown", "reqlab/report.py · SHA-256 исходника (UTF-8, LF) "
         + sha256(Path(__file__).read_text(encoding="utf-8").encode("utf-8")).hexdigest()),
        ("Методика", "ISO/IEC/IEEE 29148:2018 п. 5.2.5 и 5.2.6; IEEE Std 830-1998 п. 4.3"),
        ("Шаги", "правила → нарезка → проверка требований → проверка набора → привязка цитат → второй ревьюер"),
        ("Вызовы модели", _calls(res)),
        ("Использованные модели", ", ".join(res.models_used) or res.model),
        ("Повторы API", str(res.retries) if res.stage_durations or res.request_log else "не записано"),
        ("Время по этапам", ", ".join(f"{k}: {v:.1f} с" for k, v in res.stage_durations.items()) or "не записано"),
        ("Отсеяно находок", f"{len(res.dropped)} (неподтверждённая цитата или отклонено вторым ревьюером)"),
    ]
    for parent in res.lineage:
        rows.append(("Исходный отчёт", f"{parent['file']} · {parent['agent_version']} · "
                     f"{parent['model']} · SHA-256 {parent['sha256']}"))
    for k, v in rows:
        w(f"| {k} | {_cell(v)} |")
    w("")
    w("Ссылки на место в документе даны как номер строки исходного файла. Каждая цитата "
      "проверена программно: агент сам нашёл её в тексте документа и вычислил строку. "
      "Если модель процитировала неточно, в отчёте приведён фактический текст этих строк "
      "документа с пометкой «текст строки документа». Находки, для которых место не нашлось, "
      "в отчёт не попали (см. приложение).")
    w("")

    # 2. Нарезка --------------------------------------------------------------
    w("## 2. Как агент нарезал требования")
    w("")
    w("| ID в отчёте | ID в документе | Раздел | Строки | Краткая формулировка | Цитата |")
    w("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in reqs:
        lines = str(r.line_start) if r.line_start == r.line_end else f"{r.line_start}–{r.line_end}"
        w(f"| {r.id} | {_cell(r.doc_id or '— (присвоен агентом)')} | {_cell(r.section, 40)} | "
          f"{lines} | {_cell(r.summary, 90)} | «{_cell(r.quote, 120)}» |")
    w("")
    if res.acceptance:
        w(f"Критерии приёмки, найденные в документе ({len(res.acceptance)}): "
          + "; ".join(f"{a['id']} — строка {a['line']}" for a in res.acceptance) + ".")
        w("")
    if res.skipped:
        w("Не считались требованиями: " + "; ".join(
            f"{_cell(s.get('section', ''))} ({_cell(s.get('reason', ''))})" for s in res.skipped) + ".")
        w("")

    # 3. Проверка каждого требования -------------------------------------------
    w("## 3. Проверка каждого требования")
    w("")
    w("### 3.1. Матрица критериев")
    w("")
    w("Обозначения: ✅ выполнено, ❌ нарушено (см. находки), ➖ неприменимо, ❔ модель не дала "
      "подтверждённой оценки.")
    w("")
    w("| Требование | " + " | ".join(c.name for c in INDIVIDUAL) + " |")
    w("| :--- | " + " | ".join(":---:" for _ in INDIVIDUAL) + " |")
    for r in reqs:
        w(f"| {r.id} | " + " | ".join(ICON.get(r.verdicts.get(c.code, "unknown"), "❔")
                                       for c in INDIVIDUAL) + " |")
    w("")
    w("Критерии: " + "; ".join(f"**{c.name}** ({c.source}) — {c.definition}" for c in INDIVIDUAL))
    w("")
    w("### 3.2. Замечания по требованиям")
    w("")
    for r in reqs:
        fs = by_scope.get(r.id, [])
        head = f"#### {r.id}" + (f" · {r.doc_id}" if r.doc_id else "") + f" · {doc.location(r.line_start, r.line_end)}"
        w(head)
        w("")
        w(f"> {_cell(r.quote)}")
        w("")
        if not fs:
            if all(v == "unknown" for v in r.verdicts.values()):
                w("Оценка отсутствует. Требование нужно проверить повторно.")
            else:
                w("Нарушений не найдено.")
            w("")
            continue
        for f in fs:
            _finding(w, f, doc)

    # 4. Проверка набора ---------------------------------------------------------
    w("## 4. Проверка всего набора требований")
    w("")
    w("| Критерий | Источник | Оценка | Комментарий |")
    w("| :--- | :--- | :--- | :--- |")
    for c in SET:
        v = res.set_verdicts.get(c.code, {"status": "unknown", "comment": ""})
        w(f"| {c.name} | {c.source} | {ICON.get(v['status'], '❔')} {SET_RU.get(v['status'], v['status'])} "
          f"| {_cell(v.get('comment', ''))} |")
    w("")
    set_f = by_scope.get("SET", [])
    llm_set = [f for f in set_f if f.source == "llm"]
    rule_set = [f for f in set_f if f.source == "rule"]
    w("### 4.1. Замечания к набору")
    w("")
    if not llm_set:
        w("Межтребовательных нарушений не найдено.")
        w("")
    for f in llm_set:
        _finding(w, f, doc)
    if rule_set:
        w("### 4.2. Автоматические проверки оформления (правила, без LLM)")
        w("")
        for f in rule_set:
            _finding(w, f, doc)

    # 5. Сводка -------------------------------------------------------------------
    w("## 5. Сводка")
    w("")
    w("| Показатель | Значение |")
    w("| :--- | :--- |")
    w(f"| Требований | {len(reqs)} |")
    w(f"| Требований с нарушениями | {sum(1 for r in reqs if by_scope.get(r.id))} |")
    w(f"| Нарушений всего | {len(res.findings)} |")
    for s in SEVERITIES:
        w(f"| — {SEVERITY_RU[s].lower()} | {sev[s]} |")
    w(f"| Критериев приёмки в документе | {len(res.acceptance)} |")
    w("")
    crit_counter = Counter(f.criterion for f in res.findings)
    w("Нарушения по критериям: " + ", ".join(
        f"{name(code)} — {n}" for code, n in crit_counter.most_common()) + ".")
    w("")
    w("### Критичные дефекты")
    w("")
    critical = [f for f in res.findings if f.severity == "critical"]
    if not critical:
        w("Критичных дефектов не найдено.")
        w("")
    for group, codes in CRITICAL_GROUPS.items():
        items = [f for f in critical if f.criterion in codes]
        if items:
            w(f"**{group}:**")
            w("")
            for f in items:
                w(f"- {f.id} ({f.scope}, {doc.location(f.evidence[0].line_start, f.evidence[0].line_end)}): "
                  f"{_cell(f.problem, 220)}")
            w("")
    other = [f for f in critical if not any(f.criterion in c for c in CRITICAL_GROUPS.values())]
    if other:
        w("**Прочие критичные:**")
        w("")
        for f in other:
            w(f"- {f.id} ({f.scope}, {name(f.criterion)}, "
              f"{doc.location(f.evidence[0].line_start, f.evidence[0].line_end)}): {_cell(f.problem, 220)}")
        w("")

    # 6. Итог ---------------------------------------------------------------------
    w("## 6. Итог")
    w("")
    w(f"- **Можно ли начинать разработку:** {READY_RU.get(res.dev_ready, res.dev_ready)}")
    w(f"- **Можно ли начинать тестирование (проектирование тестов):** "
      f"{READY_RU.get(res.test_ready, res.test_ready)}")
    w("")
    if res.verdict_text:
        w(res.verdict_text.strip())
        w("")
    w("Правило решения: есть хотя бы одно критичное нарушение — «Нет»; есть только существенные — "
      "«С оговорками». Модель не может ослабить это решение.")
    w("")
    w("### Что исправить в первую очередь")
    w("")
    fixes = res.priority_fixes or [
        {"action": f.recommendation or f.problem, "finding_ids": [f.id]}
        for f in sorted(res.findings, key=lambda x: SEVERITIES.index(x.severity))[:5]]
    for n, fx in enumerate(fixes, 1):
        ids = ", ".join(fx.get("finding_ids") or [])
        w(f"{n}. {_cell(fx['action'])}" + (f" ({ids})" if ids else ""))
    w("")

    # Приложение ------------------------------------------------------------------
    if res.dropped:
        w("## Приложение. Отсеянные находки")
        w("")
        w("Агент не включил эти находки в отчёт. Список приведён для прозрачности.")
        w("")
        w("| Область | Критерий | Причина | Суть |")
        w("| :--- | :--- | :--- | :--- |")
        for d in res.dropped:
            w(f"| {d.get('scope')} | {name(d.get('criterion', ''))} | {_cell(d.get('reason', ''), 80)} "
              f"| {_cell(d.get('problem', ''), 140)} |")
        w("")
    w("---")
    w(f"_Отчёт сформирован автоматически агентом RequirementLab {res.agent_version}. "
      "Выводы модели требуют проверки аналитиком._")
    return "\n".join(out) + "\n"


def _finding(w, f: Finding, doc) -> None:
    c = ALL.get(f.criterion)
    w(f"**{SEV_ICON[f.severity]} {f.id} · {SEVERITY_RU[f.severity]} · {c.name if c else f.criterion}**"
      + (f" · связано: {', '.join(f.related)}" if f.related else ""))
    w("")
    for e in f.evidence:
        mark = "" if e.exact else " (текст строки документа)"
        w(f"- Место: {doc.location(e.line_start, e.line_end)}{mark} — «{_cell(e.quote, 300)}»")
    w(f"- Проблема: {_cell(f.problem)}")
    if c:
        w(f"- Основание: {c.code} · {c.source}")
    w("- Уверенность: не калибровалась; требуется подтверждение аналитиком.")
    w(f"- Решение аналитика: {f.analyst_decision}" + (f" — {_cell(f.analyst_note)}" if f.analyst_note else ""))
    if f.recommendation:
        w(f"- Рекомендация: {_cell(f.recommendation)}")
    if f.question:
        w(f"- Вопрос владельцу требования: {_cell(f.question)}")
    if f.reviewer_note:
        w(f"- _{_cell(f.reviewer_note)}_")
    w("")
