"""Командная строка: python run.py <документ> <отчёт.md>"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import __version__
from .llm import LLMError, from_env
from .loader import LoadError, load
from .pipeline import Agent
from .report import render

ROOT = Path(__file__).resolve().parent.parent


def agent_version() -> str:
    try:
        out = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return f"v{__version__} (git {out.stdout.strip()})"
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
    p.add_argument("--no-verify", action="store_true", help="пропустить второго ревьюера")
    p.add_argument("--no-cache", action="store_true", help="не использовать кэш ответов модели")
    p.add_argument("--quiet", action="store_true", help="не печатать ход прогона")
    args = p.parse_args(argv)

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
    if verbose:
        print(f"RequirementLab {version}: «{doc.title}» ({len(doc.lines)} строк), модель {llm.label}",
              file=sys.stderr)
    agent = Agent(llm, batch_size=args.batch_size, verify=not args.no_verify,
                  agent_version=version, verbose=verbose)
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
    print(f"Требований: {len(result.requirements)}, нарушений: {len(result.findings)}, "
          f"разработку начинать: {result.dev_ready}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
