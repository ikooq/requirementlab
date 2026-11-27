"""Конвейер агента.

    документ → правила → нарезка (LLM) → проверка требований пакетами (LLM)
    → проверка набора (LLM) → привязка цитат → второй ревьюер (LLM) → итог

Каждый шаг возвращает структурированные данные; отчёт строится из них
детерминированно, поэтому обязательные разделы присутствуют всегда.
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

from . import prompts, rules
from .criteria import ALL, INDIVIDUAL, SET, SEVERITIES
from .grounding import Locator
from .llm import LLM, LLMBadJSON
from .loader import Document

_EXPLICIT_ID = re.compile(r"\*{0,2}\b([ФF]Т?-\d+|НФТ-\d+|NFR-\d+|FR-\d+|BR-\d+|REQ-\d+)\b")


@dataclass
class Requirement:
    id: str
    doc_id: str | None
    section: str
    line_start: int
    line_end: int
    quote: str
    summary: str
    kind: str = ""
    verdicts: dict[str, str] = field(default_factory=dict)


@dataclass
class Evidence:
    quote: str
    line_start: int
    line_end: int
    exact: bool


@dataclass
class Finding:
    id: str
    scope: str                 # R-xx или SET
    criterion: str
    severity: str
    problem: str
    recommendation: str
    question: str
    evidence: list[Evidence]
    related: list[str] = field(default_factory=list)
    source: str = "llm"        # llm | rule
    reviewer_note: str = ""
    analyst_decision: str = "pending"  # pending | accepted | rejected
    analyst_note: str = ""


@dataclass
class RunResult:
    doc: Document
    model: str
    started: datetime
    duration_s: float
    requirements: list[Requirement]
    acceptance: list[dict]
    skipped: list[dict]
    findings: list[Finding]
    dropped: list[dict]
    set_verdicts: dict[str, dict]
    dev_ready: str
    test_ready: str
    verdict_text: str
    priority_fixes: list[dict]
    llm_calls: int
    cache_hits: int
    agent_version: str
    stage_durations: dict[str, float] = field(default_factory=dict)
    models_used: list[str] = field(default_factory=list)
    coverage_gaps: list[str] | None = None
    retries: int = 0
    request_log: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "doc"}
        d["started"] = self.started.isoformat(timespec="seconds")
        d["document"] = {"name": self.doc.name, "title": self.doc.title,
                         "sha256": self.doc.sha256, "lines": len(self.doc.lines)}
        return d

    @classmethod
    def from_json(cls, d: dict, doc: Document) -> "RunResult":
        """Обратная операция к to_json: пересобрать отчёт из сохранённых данных без модели."""
        d = {k: v for k, v in d.items() if k != "document"}
        d["started"] = datetime.fromisoformat(d["started"])
        d["requirements"] = [Requirement(**r) for r in d["requirements"]]
        d["findings"] = [Finding(**{**f, "evidence": [Evidence(**e) for e in f["evidence"]]})
                         for f in d["findings"]]
        return cls(doc=doc, **d)


class Agent:
    def __init__(self, llm: LLM, *, batch_size: int = 6, verify: bool = True,
                 agent_version: str = "", verbose: bool = True, passes: int = 1):
        self.llm = llm
        self.batch_size = max(1, batch_size)
        self.passes = max(1, passes)
        self._note = ""  # добавка к промпту в повторных проходах
        self.verify = verify
        self.agent_version = agent_version
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr, flush=True)

    # ------------------------------------------------------------------
    def run(self, doc: Document) -> RunResult:
        started, t0 = datetime.now().astimezone(), time.time()
        stages: dict[str, float] = {}
        loc = Locator(doc)
        numbered = doc.numbered()

        self.log("1/5 Детерминированные проверки…")
        stage_t = time.time()
        hits = rules.run(doc)
        stages["rules"] = time.time() - stage_t

        self.log("2/5 Нарезка требований…")
        stage_t = time.time()
        reqs, acs, skipped = self._segment(doc, numbered, loc)
        stages["segmentation"] = time.time() - stage_t
        self.log(f"    требований: {len(reqs)}, критериев приёмки: {len(acs)}")

        # Шаги 3–4 выполняются несколько раз независимо, находки объединяются.
        # Один проход модели ловит ~80% дефектов, но каждый раз немного разные;
        # объединение двух проходов на кейсе дало 91%. Дубли снимает второй ревьюер.
        raw: list[dict] = []
        set_verdicts: dict[str, dict] = {}
        for n in range(1, self.passes + 1):
            stage_t = time.time()
            pass_raw, pass_verdicts = self._review_pass(doc, numbered, reqs, acs, hits, n)
            stages[f"review_pass_{n}"] = time.time() - stage_t
            raw += pass_raw
            set_verdicts = _merge_set_verdicts(set_verdicts, pass_verdicts)

        # Capture model coverage before reconciling the matrix with grounded findings.
        gaps = [] if reqs else ["Нарезка не выделила ни одного требования."]
        for requirement in reqs:
            if not any(v in ("pass", "fail", "na") for v in requirement.verdicts.values()):
                gaps.append(f"{requirement.id}: отсутствует оценка критериев.")
        for criterion in SET:
            if set_verdicts.get(criterion.code, {}).get("status", "unknown") == "unknown":
                gaps.append(f"{criterion.code}: нет оценки набора требований.")

        findings, dropped = self._ground(raw, loc)
        findings += self._rule_findings(hits)
        if not self.verify:
            findings = _drop_covered_placeholders(findings)
        findings = _dedupe(findings)
        known = {r.id for r in reqs}
        _scrub_refs(findings, known)
        for d in dropped:
            d["problem"] = _scrub_text(d.get("problem", ""), known)
        _renumber(findings)
        self._reconcile_verdicts(reqs, findings)

        dev, test, text, fixes = "", "", "", []
        if self.verify and findings:
            self.log("5/5 Второй ревьюер: отсев ложных находок и итог…")
            stage_t = time.time()
            dev, test, text, fixes, more_dropped = self._critic(doc, numbered, findings)
            stages["critic"] = time.time() - stage_t
            dropped += more_dropped
            findings = [f for f in findings if f.id not in {d["id"] for d in more_dropped}]
            # Дубли правил снимаем только после второго ревьюера: если он отклонил
            # находку модели, находка правила на той же строке должна остаться.
            findings = _drop_covered_placeholders(findings)
            old_ids = {id(f): f.id for f in findings}
            _renumber(findings)  # серьёзность могла измениться — пересортировать
            remap = {old_ids[id(f)]: f.id for f in findings}
            for fx in fixes:
                fx["finding_ids"] = [remap[i] for i in fx["finding_ids"] if i in remap]
                fx["action"] = _remap_ids(fx["action"], remap)
            text = _remap_ids(text, remap)
            for d in dropped:
                d["reason"] = _scrub_text(_remap_ids(d["reason"], remap), known)
            text = _scrub_text(text, known)
            for fx in fixes:
                fx["action"] = _scrub_text(fx["action"], known)
            self._reconcile_verdicts(reqs, findings)
        dev, test = _gate(findings, acs, dev, test)

        if gaps:
            dev = "conditional" if dev == "yes" else dev
            test = "conditional" if test == "yes" else test

        return RunResult(
            doc=doc, model=self.llm.label, started=started, duration_s=time.time() - t0,
            requirements=reqs, acceptance=acs, skipped=skipped, findings=findings,
            dropped=dropped, set_verdicts=set_verdicts, dev_ready=dev, test_ready=test,
            verdict_text=text, priority_fixes=fixes, llm_calls=self.llm.calls,
            cache_hits=self.llm.cache_hits, agent_version=self.agent_version,
            coverage_gaps=gaps, stage_durations=stages, models_used=list(self.llm.models_used),
            retries=getattr(self.llm, "retries", 0), request_log=list(self.llm.request_log))

    # --- шаг 2: нарезка ----------------------------------------------
    def _segment(self, doc: Document, numbered: str, loc: Locator):
        data = self.llm.complete_json(
            prompts.SYSTEM, prompts.SEGMENT.format(title=doc.title, doc=numbered), step="нарезка")
        reqs: list[Requirement] = []
        seen: set[str] = set()
        for item in data.get("requirements") or []:
            quote = str(item.get("quote") or "").strip()
            if not quote:
                continue
            hint = _int(item.get("line_start"))
            m = loc.find(quote, hint)
            if m:
                start, end = m.start, m.end
                if not m.exact:
                    quote = doc_text(doc, start, end)
            elif hint and 1 <= hint <= len(doc.lines):
                start, end = hint, _int(item.get("line_end")) or hint
                quote = doc.lines[start - 1].strip()
            else:
                continue
            key = f"{start}:{quote[:60]}"
            if key in seen:
                continue
            seen.add(key)
            reqs.append(Requirement(
                id="", doc_id=_clean_id(item.get("doc_id")), section=str(item.get("section") or ""),
                line_start=start, line_end=max(start, end), quote=quote,
                summary=str(item.get("summary") or "").strip(), kind=str(item.get("kind") or "")))

        # Страховка: явные номера требований из документа не должны потеряться.
        have = {r.doc_id for r in reqs if r.doc_id}
        for i, line in enumerate(doc.lines, 1):
            for m in _EXPLICIT_ID.finditer(line):
                rid = m.group(1)
                if rid not in have and line.lstrip().startswith(("*", "-", rid, "**")):
                    have.add(rid)
                    reqs.append(Requirement(id="", doc_id=rid, section="", line_start=i,
                                            line_end=i, quote=line.strip(" *-"),
                                            summary="(добавлено по номеру в документе)"))
        reqs.sort(key=lambda r: (r.line_start, r.line_end))
        for n, r in enumerate(reqs, 1):
            r.id = f"R-{n:02d}"

        acs = []
        for item in data.get("acceptance_criteria") or []:
            quote = str(item.get("quote") or "").strip()
            m = loc.find(quote, _int(item.get("line"))) if quote else None
            if not m:
                continue
            if not m.exact:
                quote = doc_text(doc, m.start, m.end)
            acs.append({"id": f"AC-{len(acs) + 1}", "line": m.start,
                        "quote": quote, "summary": str(item.get("summary") or "")})
        skipped = [s for s in (data.get("skipped") or []) if isinstance(s, dict)]
        return reqs, acs, skipped

    # --- шаги 3–4 одним проходом ---------------------------------------
    def _review_pass(self, doc, numbered, reqs, acs, hits, n: int):
        """Один независимый проход проверки. Первый проход — обычный. Следующие
        группируют требования в пакеты со сдвигом и помечают промпт как перепроверку:
        модель видит требования в другом окружении и отвечает заново, а не из кэша."""
        tag = f" (проход {n}/{self.passes})" if self.passes > 1 else ""
        self._note = "" if n == 1 else (
            f"\n\nНезависимая перепроверка №{n}: другой ревьюер уже проверял этот документ. "
            "Проверь заново и самостоятельно, не полагаясь на чужие выводы; особенно ищи то, "
            "что легко пропустить при первом чтении.")
        offset = 0 if n == 1 else (self.batch_size // 2) * (n - 1) % self.batch_size
        cuts = [0] + list(range(offset or self.batch_size, len(reqs), self.batch_size)) + [len(reqs)]

        self.log(f"3/5 Проверка каждого требования{tag}…")
        raw: list[dict] = []
        for a, b in zip(cuts, cuts[1:]):
            batch = reqs[a:b]
            if not batch:
                continue
            self.log(f"    пакет {batch[0].id}…{batch[-1].id}")
            raw += self._check_batch(doc, numbered, reqs, batch, hits)

        self.log(f"4/5 Проверка набора требований целиком{tag}…")
        set_verdicts, set_raw = self._check_set(doc, numbered, reqs, acs, raw)
        raw += set_raw
        self.log("    поиск противоречий…")
        raw += self._conflicts(doc, numbered, reqs, acs, raw)
        self.log("    поиск пропущенных сценариев…")
        raw += self._gaps(doc, numbered, reqs, raw)
        self._note = ""
        return raw, set_verdicts

    def _ask(self, user: str, step: str) -> dict:
        tag = "" if not self._note else " (перепроверка)"
        return self.llm.complete_json(prompts.SYSTEM, user + self._note, step=step + tag)

    # --- шаг 3: проверка требований ------------------------------------
    def _check_batch(self, doc, numbered, reqs, batch, hits) -> list[dict]:
        lines = set()
        for r in batch:
            lines.update(range(r.line_start, r.line_end + 1))
        hint_txt = "\n".join(f"- строка {h.line_start}: {h.problem}" for h in hits
                             if h.line_start in lines) or "нет"
        user = prompts.CHECK_REQUIREMENTS.format(
            criteria=prompts.individual_criteria(), severity=prompts.CHECK_GUIDE,
            hints=hint_txt, all_reqs=_req_list(reqs), targets=_req_targets(batch),
            title=doc.title, doc=numbered)
        try:
            data = self._ask(user, f"проверка {batch[0].id}…{batch[-1].id}")
        except LLMBadJSON as e:
            # Длинный ответ на большой пакет ломается чаще: делим пакет пополам.
            if len(batch) == 1:
                self.log(f"    {batch[0].id} не проверено: {e}")
                return []
            mid = len(batch) // 2
            self.log(f"    пакет {batch[0].id}…{batch[-1].id} делю пополам: модель не дала "
                     "корректный JSON")
            return (self._check_batch(doc, numbered, reqs, batch[:mid], hits)
                    + self._check_batch(doc, numbered, reqs, batch[mid:], hits))
        by_id = {r.id: r for r in batch}
        out: list[dict] = []
        for res in data.get("results") or []:
            r = by_id.get(str(res.get("req_id") or "").strip())
            if not r:
                continue
            verdicts = res.get("verdicts") or {}
            new = {c.code: _verdict(verdicts.get(c.code)) for c in INDIVIDUAL}
            r.verdicts = _merge_verdicts(r.verdicts, new)
            for f in res.get("findings") or []:
                out.append({**f, "scope": r.id, "_hint": r.line_start})
        missing = [r.id for r in batch if not r.verdicts]
        if missing:
            self.log(f"    модель не вернула вердикты для {', '.join(missing)}")
        return out

    # --- шаг 4: проверка набора ----------------------------------------
    def _check_set(self, doc, numbered, reqs, acs, raw):
        known = "\n".join(f"- {f.get('scope')} {f.get('criterion')}: {str(f.get('problem'))[:140]}"
                          for f in raw) or "нет"
        ac_txt = "\n".join(f"- {a['id']} (строка {a['line']}): {a['quote']}" for a in acs) or "нет"
        user = prompts.CHECK_SET.format(
            criteria=prompts.set_criteria(), severity=prompts.CHECK_GUIDE,
            all_reqs=_req_list(reqs), acs=ac_txt, known=known, title=doc.title, doc=numbered)
        data = self._ask(user, "проверка набора")
        verdicts = {}
        for c in SET:
            v = (data.get("verdicts") or {}).get(c.code) or {}
            if isinstance(v, str):
                v = {"status": v, "comment": ""}
            verdicts[c.code] = {"status": _set_status(v.get("status")),
                                "comment": str(v.get("comment") or "")}
        found = [{**f, "scope": "SET"} for f in data.get("findings") or []]
        return verdicts, found

    def _conflicts(self, doc, numbered, reqs, acs, raw) -> list[dict]:
        known = "\n".join(f"- {f.get('scope')} {f.get('criterion')}: {str(f.get('problem'))[:140]}"
                          for f in raw if f.get("criterion") in ("COR", "S-CON")) or "нет"
        req_txt = "\n".join(f"- {r.id} [{r.doc_id or 'без номера'}; строки {r.line_start}–"
                            f"{r.line_end}]: «{r.quote}»" for r in reqs)
        ac_txt = "\n".join(f"- {a['id']} (строка {a['line']}): {a['quote']}" for a in acs) or "нет"
        user = prompts.CONFLICTS.format(known=known, reqs=req_txt, acs=ac_txt,
                                        title=doc.title, doc=numbered)
        data = self._ask(user, "поиск противоречий")
        return [{**f, "scope": "SET", "criterion": f.get("criterion") or "S-CON"}
                for f in data.get("findings") or []]

    def _gaps(self, doc, numbered, reqs, raw) -> list[dict]:
        """Отдельный проход на пропущенные сценарии: в общей проверке набора модель
        их часто пропускает, увлекшись противоречиями."""
        known = "\n".join(f"- {f.get('scope')} {f.get('criterion')}: {str(f.get('problem'))[:140]}"
                          for f in raw) or "нет"
        req_txt = "\n".join(f"- {r.id} [{r.doc_id or 'без номера'}; строки {r.line_start}–"
                            f"{r.line_end}]: «{r.quote}»" for r in reqs)
        user = prompts.GAPS.format(severity=prompts.CHECK_GUIDE, known=known, reqs=req_txt,
                                   title=doc.title, doc=numbered)
        data = self._ask(user, "поиск пропущенных сценариев")
        return [{**f, "scope": "SET", "criterion": f.get("criterion") or "S-CMP"}
                for f in data.get("findings") or []]

    # --- привязка цитат --------------------------------------------------
    def _ground(self, raw: list[dict], loc: Locator):
        findings, dropped = [], []
        for n, f in enumerate(raw, 1):
            crit = str(f.get("criterion") or "").strip().upper()
            if crit not in ALL:
                crit = "S-CMP" if f.get("scope") == "SET" else "CMP"
            quotes = f.get("quotes") or ([f["quote"]] if f.get("quote") else [])
            ev = []
            for q in quotes[:4]:
                m = loc.find(str(q), f.get("_hint"))
                if m:
                    # Неточную цитату модели в отчёт не выводим: показываем текст документа.
                    text = str(q).strip() if m.exact else doc_text(loc.doc, m.start, m.end)
                    ev.append(Evidence(text, m.start, m.end, m.exact))
            problem = str(f.get("problem") or "").strip()
            if not ev or not problem:
                dropped.append({"id": f"raw-{n}", "scope": f.get("scope"), "criterion": crit,
                                "reason": "цитата не найдена в документе" if problem else
                                "нет описания проблемы", "problem": problem[:200]})
                continue
            sev = str(f.get("severity") or "major").lower()
            findings.append(Finding(
                id="", scope=f.get("scope", "SET"), criterion=crit,
                severity=sev if sev in SEVERITIES else "major", problem=problem,
                recommendation=str(f.get("recommendation") or "").strip(),
                question=str(f.get("question") or "").strip(), evidence=_unique_evidence(ev),
                related=[str(x) for x in (f.get("related") or [])]))
        return findings, dropped

    def _rule_findings(self, hits) -> list[Finding]:
        out = []
        for h in hits:
            if h.rule == "vague-word":
                continue  # размытые слова — только подсказка модели, без отдельной находки
            evidence = [Evidence(h.quote, h.line_start, h.line_end, True)]
            if h.related_line:
                line, quote = h.related_line
                evidence.append(Evidence(quote, line, line, True))
            out.append(Finding(id="", scope="SET", criterion=h.criterion, severity=h.severity,
                               problem=h.problem, recommendation=h.recommendation, question="",
                               evidence=evidence,
                               source="rule"))
        return out

    def _reconcile_verdicts(self, reqs, findings):
        """Вердикт «нарушено» без подтверждённой находки снимается, и наоборот."""
        by_scope: dict[str, set[str]] = {}
        for f in findings:
            by_scope.setdefault(f.scope, set()).add(f.criterion)
        for r in reqs:
            crits = by_scope.get(r.id, set())
            if not r.verdicts:
                r.verdicts = {c.code: ("fail" if c.code in crits else "unknown") for c in INDIVIDUAL}
                continue
            for code in r.verdicts:
                if code in crits:
                    r.verdicts[code] = "fail"
                elif r.verdicts[code] == "fail":
                    r.verdicts[code] = "unknown"

    # --- шаг 5: второй ревьюер ------------------------------------------
    def _critic(self, doc, numbered, findings):
        lines = []
        for f in findings:
            q = " | ".join(f"«{e.quote[:160]}» (стр. {e.line_start})" for e in f.evidence)
            lines.append(f"- {f.id} [{f.scope}, {f.criterion}, {f.severity}, {f.source}] "
                         f"{f.problem} Цитаты: {q}")
        user = prompts.CRITIC.format(severity=prompts.SEVERITY_GUIDE, findings="\n".join(lines),
                                     title=doc.title, doc=numbered)
        data = self.llm.complete_json(prompts.SYSTEM, user, step="второй ревьюер")
        by_id = {f.id: f for f in findings}
        dropped = []
        dup_of: dict[str, str] = {}
        for d in data.get("decisions") or []:
            f = by_id.get(str(d.get("id") or ""))
            if not f:
                continue
            reason = str(d.get("reason") or "").strip()
            if d.get("keep") is False and f.source != "rule":
                target = str(d.get("duplicate_of") or "").strip()
                if not target:
                    m = re.search(r"дубл\w*\s+(F-\d{3})", reason, re.I)
                    target = m.group(1) if m else ""
                if target:
                    if target not in by_id or target == f.id:
                        continue
                    # Contradictory behaviour and missing coverage require distinct fixes.
                    conflict = {"S-CON", "COR"}
                    coverage = {"TRC", "S-TRC", "S-VAL", "VER"}
                    pair = {f.criterion, by_id[target].criterion}
                    if pair & conflict and pair & coverage:
                        continue
                    dup_of[f.id] = target
                dropped.append({"id": f.id, "scope": f.scope, "criterion": f.criterion,
                                "reason": reason or "отклонено вторым ревьюером",
                                "problem": f.problem[:200]})
                continue
            sev = str(d.get("severity") or "").lower()
            if sev in SEVERITIES and sev != f.severity:
                f.reviewer_note = f"Серьёзность изменена вторым ревьюером: {f.severity} → {sev}. {reason}".strip()
                f.severity = sev
        # Дубликат не теряется: его цитаты и связи переходят к оставленной находке.
        gone = {d["id"] for d in dropped}
        for src, dst in dup_of.items():
            seen = {src}
            while dst in gone and dst in dup_of and dst not in seen:  # цепочка дубликатов
                seen.add(dst)
                dst = dup_of[dst]
            if dst in gone:
                continue
            _absorb(by_id[dst], by_id[src])
            for d in dropped:
                if d["id"] == src:
                    d["reason"] = f"объединена с {dst}. {d['reason']}".strip()
        fixes = []
        for p in data.get("priority_fixes") or []:
            ids = [i for i in (p.get("finding_ids") or []) if i in by_id]
            if p.get("action"):
                fixes.append({"action": str(p["action"]), "finding_ids": ids})
        return (str(data.get("dev_ready") or ""), str(data.get("test_ready") or ""),
                str(data.get("verdict") or ""), fixes, dropped)


# --- вспомогательные функции ---------------------------------------------
_MD = re.compile(r"\*\*|__|`")


def doc_text(doc: Document, start: int, end: int) -> str:
    """Текст строк документа start..end без markdown-разметки — то, что реально написано."""
    parts = [_MD.sub("", ln).strip(" \t*-#>|") for ln in doc.lines[start - 1:end]]
    return " ".join(p for p in parts if p)


def _drop_covered_placeholders(findings: list[Finding]) -> list[Finding]:
    """Находка правила «заглушка» (S-CMP) не нужна, если модель уже разобрала эту строку.

    Считается только основное место находки модели (первая цитата): вспомогательная
    цитата на ту же строку может быть про другую проблему."""
    covered = {ln for f in findings if f.source != "rule"
               for ln in range(f.evidence[0].line_start, f.evidence[0].line_end + 1)}
    return [f for f in findings if not (f.source == "rule" and f.criterion == "S-CMP"
                                        and f.evidence[0].line_start in covered)]


_REQ_REF = re.compile(r"\bR-\d+\b")


def _scrub_text(text: str, known: set[str]) -> str:
    """Ссылка модели на несуществующее требование (R-103 при 43 требованиях) — не в отчёт."""
    return _REQ_REF.sub(lambda m: m.group(0) if m.group(0) in known else "[требование вне нарезки]",
                        text or "")


def _scrub_refs(findings: list[Finding], known: set[str]) -> None:
    for f in findings:
        f.related = [r for r in f.related if r in known]
        f.problem = _scrub_text(f.problem, known)
        f.recommendation = _scrub_text(f.recommendation, known)
        f.question = _scrub_text(f.question, known)


def _remap_ids(text: str, remap: dict[str, str]) -> str:
    """Номера находок в тексте модели — на номера после перенумерации."""
    def sub(m: re.Match) -> str:
        old = m.group(0)
        return remap.get(old, f"{old} (отсеяна)")
    return re.sub(r"F-\d{3}", sub, text or "")


def _gate(findings, acs, dev: str, test: str) -> tuple[str, str]:
    """Решение «можно ли начинать» не отдаётся модели целиком: критичная находка — всегда «нет»."""
    crit = [f for f in findings if f.severity == "critical"]
    major = [f for f in findings if f.severity == "major"]
    rule_dev = "no" if crit else ("conditional" if major else "yes")
    untestable = any(f.criterion in ("VER", "S-VAL") and f.severity == "critical" for f in findings)
    rule_test = "no" if (untestable or not acs) else ("conditional" if crit or major else "yes")
    order = {"no": 0, "conditional": 1, "yes": 2}
    dev = min(rule_dev, dev, key=lambda x: order.get(x, 3)) if dev in order else rule_dev
    test = min(rule_test, test, key=lambda x: order.get(x, 3)) if test in order else rule_test
    return dev, test


def _dedupe(findings: list[Finding]) -> list[Finding]:
    """Точные повторы снимаются, почти одинаковые находки склеиваются в одну.

    Почти одинаковые — основное место (первая цитата) на тех же строках и описания
    проблемы совпадают по словам хотя бы наполовину. Так склеиваются повторы из двух
    проходов, а разные дефекты на одной строке остаются разными."""
    out: list[Finding] = []
    for finding in findings:
        # A shared opening or source line is not enough to establish a duplicate.
        for previous in out:
            if (previous.scope != finding.scope or previous.source != finding.source
                    or previous.criterion != finding.criterion or not _same_place(previous, finding)):
                continue
            same_text = previous.problem.strip().casefold() == finding.problem.strip().casefold()
            if same_text or (finding.source == "llm" and _similar(previous.problem, finding.problem) >= 0.5
                             and previous.problem[:50].casefold() != finding.problem[:50].casefold()):
                _absorb(previous, finding)
                break
        else:
            out.append(finding)
    return out


_WORD = re.compile(r"[\wА-Яа-яЁё-]{4,}")


def _stems(text: str) -> set[str]:
    return {w.lower()[:6] for w in _WORD.findall(text or "")}


def _similar(a: str, b: str) -> float:
    sa, sb = _stems(a), _stems(b)
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def _same_place(a: Finding, b: Finding) -> bool:
    ea, eb = a.evidence[0], b.evidence[0]
    return (ea.line_start, ea.line_end) == (eb.line_start, eb.line_end)


def _unique_evidence(ev: list[Evidence], limit: int = 5) -> list[Evidence]:
    """Одна и та же цитата дважды в находке не нужна."""
    out, seen = [], set()
    for e in ev:
        key = (e.line_start, e.line_end, e.quote.strip().lower())
        if key not in seen and (e.line_start, e.line_end) not in {(o.line_start, o.line_end) for o in out}:
            seen.add(key)
            out.append(e)
    return out[:limit]


def _absorb(keep: Finding, dup: Finding) -> None:
    """Склеить дубликат с оставленной находкой: цитаты, связи и худшая серьёзность."""
    keep.evidence = _unique_evidence(keep.evidence + dup.evidence)
    keep.related = list(dict.fromkeys(keep.related + dup.related))
    if SEVERITIES.index(dup.severity) < SEVERITIES.index(keep.severity):
        keep.severity = dup.severity


def _renumber(findings: list[Finding]) -> None:
    def order(f: Finding):
        scope = 10_000 if f.scope == "SET" else int(re.sub(r"\D", "", f.scope) or 0)
        return (scope, SEVERITIES.index(f.severity), f.evidence[0].line_start)
    findings.sort(key=order)
    for n, f in enumerate(findings, 1):
        f.id = f"F-{n:03d}"


def _req_list(reqs: list[Requirement]) -> str:
    return "\n".join(f"- {r.id} [{r.doc_id or 'без номера'}; строки {r.line_start}–{r.line_end}] "
                     f"{r.summary}" for r in reqs)


def _req_targets(batch: list[Requirement]) -> str:
    return "\n".join(f"{r.id} [{r.doc_id or 'без номера'}; раздел «{r.section}»; строки "
                     f"{r.line_start}–{r.line_end}]: «{r.quote}»" for r in batch)


def _merge_verdicts(old: dict[str, str], new: dict[str, str]) -> dict[str, str]:
    """Вердикты двух проходов: «нарушено» из любого прохода сильнее «выполнено»."""
    if not old:
        return new
    rank = {"fail": 0, "pass": 1, "na": 2, "unknown": 3}
    return {k: min(old.get(k, "unknown"), new.get(k, "unknown"), key=lambda v: rank.get(v, 3))
            for k in set(old) | set(new)}


def _merge_set_verdicts(old: dict[str, dict], new: dict[str, dict]) -> dict[str, dict]:
    """Оценка набора по проходам: берём худшую с её комментарием."""
    rank = {"fail": 0, "partial": 1, "pass": 2, "unknown": 3}
    out = dict(old)
    for code, v in new.items():
        if code not in out or rank.get(v["status"], 3) < rank.get(out[code]["status"], 3):
            out[code] = v
    return out


def _verdict(v) -> str:
    v = str(v or "").lower().strip()
    return v if v in ("pass", "fail", "na") else "unknown"


def _set_status(v) -> str:
    v = str(v or "").lower().strip()
    return v if v in ("pass", "partial", "fail") else "unknown"


def _clean_id(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip().strip("*").strip()
    return s if s and s.lower() not in ("null", "none", "—", "-") else None


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
