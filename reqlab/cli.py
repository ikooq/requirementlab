"""Командная строка: python run.py <документ> <отчёт.md>"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
import hashlib
from pathlib import Path

from . import __version__
from .llm import LLMError, from_env
from .loader import LoadError, load
from .pipeline import Agent
from .report import render, render_brief

ROOT = Path(__file__).resolve().parent.parent


def agent_version() -> str:
    """Версия и коммит агента. «-dirty» — только если изменён код, а не отчёты в reports/."""
    def git(*args: str) -> str:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    try:
        desc = git("describe", "--tags", "--always")
        if desc:
            if git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)reports"):
                desc += "-dirty"
            return f"v{__version__} (git {desc})"
    except (OSError, subprocess.SubprocessError):
        pass
    return f"v{__version__}"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(ROOT / ".env")
    load_dotenv()  # .env в текущей папке, если запускают не из корня


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass
    p = argparse.ArgumentParser(
        prog="reqlab",
        description="AI-агент статической проверки требований (ISO/IEC/IEEE 29148, IEEE 830).")
    p.add_argument("input", help="путь к документу требований (.md, .txt, .pdf, .docx)")
    p.add_argument("output", nargs="?", default="reports/report.md",
                   help="путь к файлу отчёта (по умолчанию reports/report.md)")
    p.add_argument("--provider", help="gemini | openai (по умолчанию из .env)")
    p.add_argument("--model", help="имя модели (по умолчанию из .env)")
    p.add_argument("--batch-size", type=int, default=6, help="требований в одном запросе к модели")
    p.add_argument("--passes", type=int, default=2,
                   help="независимых проходов проверки, находки объединяются (по умолчанию 2; "
                        "1 — вдвое меньше запросов к модели)")
    p.add_argument("--no-verify", action="store_true", help="пропустить второго ревьюера")
    cache = p.add_mutually_exclusive_group()
    cache.add_argument("--fresh-run", action="store_true", help="начать чистый прогон с сохранением промежуточных ответов")
    cache.add_argument("--continue-run", type=uuid.UUID, metavar="ID", help="продолжить сохранённый прогон")
    cache.add_argument("--no-cache", action="store_true", help="не использовать кэш ответов модели")
    p.add_argument("--brief", action="store_true",
                   help="дополнительно записать короткий отчёт на один экран (<отчёт>-brief.md)")
    p.add_argument("--quiet", action="store_true", help="не печатать ход прогона")
    p.add_argument("--resume", metavar="REPORT.json", help="пересобрать Markdown из готового JSON отчёта без повторного вызова модели")
    args = p.parse_args(argv)

    if args.resume:
        try:
            from .pipeline import RunResult
            saved = json.loads(Path(args.resume).read_text(encoding="utf-8"))
            doc = load(args.input)
            if (saved.get("document") or {}).get("sha256") != doc.sha256:
                raise ValueError("SHA-256 документа не совпадает с сохранённым отчётом")
            result = RunResult.from_json(saved, doc)
            out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(render(result), encoding="utf-8")
            print(f"Отчёт восстановлен из {args.resume}: {out}")
            return 0
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"Ошибка восстановления: {e}", file=sys.stderr)
            return 2

    _load_env()
    verbose = not args.quiet
    try:
        doc = load(args.input)
        llm = from_env(args.provider, args.model,
                       cache_dir=None if args.no_cache else ROOT / ".cache" / "llm",
                       verbose=verbose)
    except (LoadError, LLMError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 2

    version = agent_version()
    agent = Agent(llm, batch_size=args.batch_size, verify=not args.no_verify,
                  agent_version=version, verbose=verbose, passes=args.passes)
    run_id = args.continue_run or (uuid.uuid4() if args.fresh_run else None)
    if run_id:
        directory = ROOT / ".cache" / "runs" / str(run_id)
        manifest = directory / "settings.json"
        settings = {
            "document": doc.sha256, "model": llm.requested_model, "provider": llm.provider,
            "endpoint_hash": hashlib.sha256(getattr(llm, "base_url", "").encode()).hexdigest(),
            "passes": agent.passes, "batch_size": agent.batch_size, "verify": agent.verify,
        }
        try:
            if args.continue_run:
                if json.loads(manifest.read_text(encoding="utf-8")) != settings:
                    raise ValueError("Документ или настройки отличаются от исходного прогона.")
            else:
                directory.mkdir(parents=True, exist_ok=False)
                manifest.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        except (OSError, ValueError) as e:
            print(f"Ошибка сохранённого прогона: {e}", file=sys.stderr)
            return 2
        llm.cache_dir = directory
        print(f"Для продолжения замените --fresh-run на --continue-run {run_id}", file=sys.stderr)
    if verbose:
        print(f"RequirementLab {version}: «{doc.title}» ({len(doc.lines)} строк), модель {llm.label}",
              file=sys.stderr)
    try:
        result = agent.run(doc)
    except LLMError as e:
        print(f"Ошибка модели: {e}", file=sys.stderr)
        return 3

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(result), encoding="utf-8")
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(result.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Отчёт: {out}\nДанные: {json_path}")
    if args.brief:
        brief_path = out.with_name(out.stem + "-brief.md")
        brief_path.write_text(render_brief(result), encoding="utf-8")
        print(f"Кратко: {brief_path}")
    print(f"Требований: {len(result.requirements)}, нарушений: {len(result.findings)}, "
          f"разработку начинать: {result.dev_ready}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
