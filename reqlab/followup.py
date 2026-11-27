"""Дополнительное ревью источника с учётом уже опубликованного отчёта."""
import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

from eval.evidence import audit
from .cli import ROOT, _load_env, agent_version
from .criteria import criteria_block, SET
from .grounding import Locator
from .llm import from_env, LLMError
from .loader import load, LoadError
from .pipeline import Agent, RunResult, _dedupe, _gate
from .prompts import SYSTEM, SEVERITY_GUIDE
from .report import render, render_brief


def extend(document, source, client):
    raw = source.read_bytes()
    saved = json.loads(raw)
    _, errors = audit(saved, document)
    if errors:
        raise ValueError(errors[0])
    result = RunResult.from_json(saved, document)
    previous = [[f.id, f.problem] for f in result.findings]
    context = document.numbered() + "\nУже зарегистрировано:\n" + json.dumps(previous, ensure_ascii=False)
    prompt = """Найди пропущенные дефекты, мысленно исполняя спецификацию. Для каждого
наблюдаемого результата пройди от входного события до его значения, получателя и момента.
У каждого выходного поля проверь, из какой величины оно берётся и что произойдёт при
нескольких допустимых источниках. Для каждого перехода состояния пройди путь вперёд, назад,
при задержке и повторе. Сверь повторённые условия в бизнес-правилах, таблицах, алгоритме
и примерах: различай противоречие, независимое дублирование изменяемой нормы и необходимую ссылку.
Попробуй вывести два разных поведения или теста из текста. Если оба допустимы, опиши конкретный
контрпример. Причины уже зарегистрированных находок пропускай. Для каждой новой нужны поле
criterion из списка, severity, дословные quotes, problem, recommendation и question.
Верни один объект {"findings":[...]}. Пустой список допустим.
""" + criteria_block(SET) + "\n" + SEVERITY_GUIDE + "\n" + context
    started = datetime.now().astimezone()
    tick = time.monotonic()
    reply = client.complete_json(SYSTEM, prompt, step="поиск пропущенных сценариев: контрпримеры")
    agent = Agent(client, verbose=False)
    additions, discarded = agent._ground([dict(f, scope="SET") for f in reply["findings"]], Locator(document))
    additions = _dedupe(additions)
    if additions:
        items = [[i, f.problem, [e.quote for e in f.evidence]] for i, f in enumerate(additions, 1)]
        check = """Оставь только подтверждённые новые проблемы, для которых нужна отдельная правка.
Сверь утверждения со всем источником и уже зарегистрированными находками. Отклони придирки,
необоснованные ограничения и повторы. Ответ: {"decisions":[{"id":1,"keep":true,"reason":"..."}]}.
Укажи один ответ для каждого номера.\n""" + context + "\nКандидаты:\n" + json.dumps(items, ensure_ascii=False)
        decisions = client.complete_json(SYSTEM, check, step="проверка дополнительных замечаний").get("decisions", [])
        by_id = {d["id"]: d for d in decisions if isinstance(d, dict) and type(d.get("id")) is int
                 and type(d.get("keep")) is bool}
        if len(decisions) != len(additions) or set(by_id) != set(range(1, len(additions) + 1)):
            raise ValueError("Ревьюер вернул неполный список решений")
        for i, f in enumerate(additions, 1):
            if not by_id[i]["keep"]:
                discarded.append(dict(scope="SET", criterion=f.criterion, problem=f.problem,
                                      reason=str(by_id[i].get("reason") or "отклонено ревьюером")))
                continue
            f.id = f"F-{len(result.findings) + 1:03d}"
            result.findings.append(f)
            if f.criterion in result.set_verdicts:
                result.set_verdicts[f.criterion] = dict(status="fail", comment=f.problem)
    result.dropped.extend(discarded)
    result.lineage.append({"file": source.name, "sha256": hashlib.sha256(raw).hexdigest(),
                           "started": saved["started"], "model": saved["model"],
                           "agent_version": saved["agent_version"], "duration_s": saved["duration_s"]})
    result.started, result.agent_version = started, agent_version()
    result.model = f"{client.label} (дополнительное ревью)"
    result.stage_durations = {"followup": time.monotonic() - tick}
    result.duration_s = result.stage_durations["followup"]
    result.models_used = client.models_used
    result.llm_calls, result.retries = client.calls, client.retries
    result.request_log, result.cache_hits = client.request_log, client.cache_hits
    result.dev_ready, result.test_ready = _gate(result.findings, result.acceptance, result.dev_ready, result.test_ready)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("document", "source", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--model")
    args = parser.parse_args()
    output = args.output
    targets = [output, output.with_suffix(".json"), output.with_name(output.stem + "-brief.md")]
    if output.suffix.lower() != ".md" or any(p.resolve() in {args.source.resolve(), args.document.resolve()} for p in targets):
        parser.error("укажите отдельный выходной файл .md")
    _load_env()
    try:
        client = from_env(model=args.model, cache_dir=ROOT / ".cache" / "followup")
        result = extend(load(args.document), args.source, client)
        targets[0].parent.mkdir(parents=True, exist_ok=True)
        for target, content in zip(targets, (render(result), json.dumps(result.to_json(), ensure_ascii=False, indent=2),
                                             render_brief(result))):
            target.write_text(content, encoding="utf-8")
    except (ValueError, KeyError, OSError, LoadError, LLMError) as error:
        parser.exit(2, f"Ошибка дополнительного ревью: {error}\n")


if __name__ == "__main__":
    main()
