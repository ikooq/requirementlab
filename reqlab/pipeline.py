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
from .llm import LLM
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

    def to_json(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "doc"}
        d["started"] = self.started.isoformat(timespec="seconds")
        d["document"] = {"name": self.doc.name, "title": self.doc.title,
                         "sha256": self.doc.sha256, "lines": len(self.doc.lines)}
        return d


class Agent:
    def __init__(self, llm: LLM, *, batch_size: int = 6, verify: bool = True,
                 agent_version: str = "", verbose: bool = True):
        self.llm = llm
        self.batch_size = max(1, batch_size)
        self.verify = verify
        self.agent_version = agent_version
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr, flush=True)

    # ------------------------------------------------------------------
    def run(self, doc: Document) -> RunResult:
        started, t0 = datetime.now().astimezone(), time.time()
        loc = Locator(doc)
        numbered = doc.numbered()

        self.log("1/5 Детерминированные проверки…")
        hits = rules.run(doc)

        self.log("2/5 Нарезка требований…")
        reqs, acs, skipped = self._segment(doc, numbered, loc)
        self.log(f"    требований: {len(reqs)}, критериев приёмки: {len(acs)}")

        self.log("3/5 Проверка каждого требования…")
        raw: list[dict] = []
        for i in range(0, len(reqs), self.batch_size):
            batch = reqs[i:i + self.batch_size]
            self.log(f"    пакет {batch[0].id}…{batch[-1].id}")
            raw += self._check_batch(doc, numbered, reqs, batch, hits)

        self.log("4/5 Проверка набора требований целиком…")
        set_verdicts, set_raw = self._check_set(doc, numbered, reqs, acs, raw)
        raw += set_raw

        findings, dropped = self._ground(raw, loc)
        findings += self._rule_findings(hits)
        findings = _dedupe(findings)
        _renumber(findings)
        self._reconcile_verdicts(reqs, findings)

        dev, test, text, fixes = "", "", "", []
        if self.verify and findings:
            self.log("5/5 Второй ревьюер: отсев ложных находок и итог…")
            dev, test, text, fixes, more_dropped = self._critic(doc, numbered, findings)
            dropped += more_dropped
            findings = [f for f in findings if f.id not in {d["id"] for d in more_dropped}]
            old_ids = {id(f): f.id for f in findings}
            _renumber(findings)  # серьёзность могла измениться — пересортировать
            remap = {old_ids[id(f)]: f.id for f in findings}
            for fx in fixes:
                fx["finding_ids"] = [remap[i] for i in fx["finding_ids"] if i in remap]
            self._reconcile_verdicts(reqs, findings)
        dev, test = _gate(findings, acs, dev, test)

        return RunResult(
            doc=doc, model=self.llm.label, started=started, duration_s=time.time() - t0,
            requirements=reqs, acceptance=acs, skipped=skipped, findings=findings,
            dropped=dropped, set_verdicts=set_verdicts, dev_ready=dev, test_ready=test,
            verdict_text=text, priority_fixes=fixes, llm_calls=self.llm.calls,
            cache_hits=self.llm.cache_hits, agent_version=self.agent_version)

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
        for n, item in enumerate(data.get("acceptance_criteria") or [], 1):
            quote = str(item.get("quote") or "").strip()
            m = loc.find(quote, _int(item.get("line"))) if quote else None
            acs.append({"id": f"AC-{n}", "line": m.start if m else _int(item.get("line")),
                        "quote": quote, "summary": str(item.get("summary") or "")})
        skipped = [s for s in (data.get("skipped") or []) if isinstance(s, dict)]
        return reqs, acs, skipped

    # --- шаг 3: проверка требований ------------------------------------
    def _check_batch(self, doc, numbered, reqs, batch, hits) -> list[dict]:
        lines = set()
        for r in batch:
            lines.update(range(r.line_start, r.line_end + 1))
        hint_txt = "\n".join(f"- строка {h.line_start}: {h.problem}" for h in hits
                             if h.line_start in lines) or "нет"
        user = prompts.CHECK_REQUIREMENTS.format(
            criteria=prompts.individual_criteria(), severity=prompts.SEVERITY_GUIDE,
            hints=hint_txt, all_reqs=_req_list(reqs), targets=_req_targets(batch),
            title=doc.title, doc=numbered)
        data = self.llm.complete_json(prompts.SYSTEM, user,
                                      step=f"проверка {batch[0].id}…{batch[-1].id}")
        by_id = {r.id: r for r in batch}
        out: list[dict] = []
        for res in data.get("results") or []:
            r = by_id.get(str(res.get("req_id") or "").strip())
            if not r:
                continue
            verdicts = res.get("verdicts") or {}
            r.verdicts = {c.code: _verdict(verdicts.get(c.code)) for c in INDIVIDUAL}
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
            criteria=prompts.set_criteria(), severity=prompts.SEVERITY_GUIDE,
            all_reqs=_req_list(reqs), acs=ac_txt, known=known, title=doc.title, doc=numbered)
        data = self.llm.complete_json(prompts.SYSTEM, user, step="проверка набора")
        verdicts = {}
        for c in SET:
            v = (data.get("verdicts") or {}).get(c.code) or {}
            if isinstance(v, str):
                v = {"status": v, "comment": ""}
            verdicts[c.code] = {"status": _set_status(v.get("status")),
                                "comment": str(v.get("comment") or "")}
        found = [{**f, "scope": "SET"} for f in data.get("findings") or []]
        return verdicts, found

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
                    ev.append(Evidence(str(q).strip(), m.start, m.end, m.exact))
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
                question=str(f.get("question") or "").strip(), evidence=ev,
                related=[str(x) for x in (f.get("related") or [])]))
        return findings, dropped

    def _rule_findings(self, hits) -> list[Finding]:
        out = []
        for h in hits:
            if h.rule == "vague-word":
                continue  # размытые слова — только подсказка модели, без отдельной находки
            out.append(Finding(id="", scope="SET", criterion=h.criterion, severity=h.severity,
                               problem=h.problem, recommendation=h.recommendation, question="",
                               evidence=[Evidence(h.quote, h.line_start, h.line_end, True)],
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
        for d in data.get("decisions") or []:
            f = by_id.get(str(d.get("id") or ""))
            if not f:
                continue
            reason = str(d.get("reason") or "").strip()
            if d.get("keep") is False and f.source != "rule":
                dropped.append({"id": f.id, "scope": f.scope, "criterion": f.criterion,
                                "reason": reason or "отклонено вторым ревьюером",
                                "problem": f.problem[:200]})
                continue
            sev = str(d.get("severity") or "").lower()
            if sev in SEVERITIES and sev != f.severity:
                f.reviewer_note = f"Серьёзность изменена вторым ревьюером: {f.severity} → {sev}. {reason}".strip()
                f.severity = sev
        fixes = []
        for p in data.get("priority_fixes") or []:
            ids = [i for i in (p.get("finding_ids") or []) if i in by_id]
            if p.get("action"):
                fixes.append({"action": str(p["action"]), "finding_ids": ids})
        return (str(data.get("dev_ready") or ""), str(data.get("test_ready") or ""),
                str(data.get("verdict") or ""), fixes, dropped)


# --- вспомогательные функции ---------------------------------------------
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
    seen, out = set(), []
    for f in findings:
        key = (f.scope, f.criterion, f.evidence[0].line_start, f.problem[:50].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


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
