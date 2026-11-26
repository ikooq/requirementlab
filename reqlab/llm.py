"""Клиенты LLM: Google Gemini (по умолчанию) и любой OpenAI-совместимый API.

Оба клиента возвращают разобранный JSON. Повторы при 429/5xx, пауза между
вызовами (бесплатный тариф Gemini ограничивает число запросов в минуту) и
дисковый кэш ответов, чтобы повторный прогон не тратил квоту.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


class LLMError(Exception):
    pass


class _Retryable(Exception):
    def __init__(self, msg: str, wait: float | None = None):
        super().__init__(msg)
        self.wait = wait


def parse_json(text: str) -> dict:
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("в ответе нет JSON-объекта")
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("ожидался JSON-объект")
    return data


class LLM:
    provider = "base"

    def __init__(self, model: str, *, cache_dir: Path | None = None, min_interval: float = 0.0,
                 max_retries: int = 6, timeout: float = 240.0, temperature: float = 0.1,
                 verbose: bool = True):
        self.model = model
        self.cache_dir = cache_dir
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.temperature = temperature
        self.verbose = verbose
        self.calls = 0          # реальные обращения к API
        self.cache_hits = 0
        self._last = 0.0

    @property
    def label(self) -> str:
        return f"{self.provider} / {self.model}"

    def complete_json(self, system: str, user: str, *, step: str = "") -> dict:
        key = hashlib.sha256(
            json.dumps([self.provider, self.model, self.temperature, system, user]).encode()
        ).hexdigest()[:32]
        cached = self._cache_get(key)
        if cached is not None:
            self.cache_hits += 1
            self._log(f"  [{step}] ответ из кэша")
            return cached

        prompt = user
        last_err = ""
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                t0 = time.time()
                text = self._call(system, prompt)
                self.calls += 1
                self._log(f"  [{step}] ответ модели за {time.time() - t0:.1f} с")
            except _Retryable as e:
                wait = e.wait or min(60.0, 5.0 * 2 ** (attempt - 1))
                last_err = str(e)
                self._log(f"  [{step}] {e}; повтор через {wait:.0f} с ({attempt}/{self.max_retries})")
                time.sleep(wait)
                continue
            try:
                data = parse_json(text)
            except (ValueError, json.JSONDecodeError) as e:
                last_err = f"невалидный JSON: {e}"
                self._log(f"  [{step}] {last_err}; прошу модель повторить")
                prompt = (user + "\n\nВНИМАНИЕ: предыдущий ответ не был валидным JSON. "
                          "Верни только один JSON-объект без пояснений и без markdown.")
                continue
            self._cache_put(key, data)
            return data
        raise LLMError(f"Модель не дала корректный ответ на шаге «{step}»: {last_err}")

    # --- служебное -------------------------------------------------------
    def _call(self, system: str, user: str) -> str:  # pragma: no cover - абстрактный
        raise NotImplementedError

    def _throttle(self) -> None:
        delta = time.time() - self._last
        if self.min_interval and delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.time()

    def _cache_get(self, key: str) -> dict | None:
        if not self.cache_dir:
            return None
        f = self.cache_dir / f"{key}.json"
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        return None

    def _cache_put(self, key: str, data: dict) -> None:
        if not self.cache_dir:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr, flush=True)

    def _post(self, url: str, headers: dict, body: dict) -> dict:
        try:
            r = requests.post(url, headers=headers, json=body, timeout=self.timeout)
        except requests.RequestException as e:
            raise _Retryable(f"сетевая ошибка: {e.__class__.__name__}") from e
        if r.status_code in (408, 429, 500, 502, 503, 504):
            wait = None
            ra = r.headers.get("Retry-After")
            if ra and ra.isdigit():
                wait = float(ra)
            m = re.search(r'"retryDelay":\s*"(\d+)s"', r.text)
            if m:
                wait = float(m.group(1)) + 1
            raise _Retryable(f"HTTP {r.status_code}", wait)
        if r.status_code >= 400:
            raise LLMError(f"HTTP {r.status_code} от {self.provider}: {r.text[:500]}")
        return r.json()


class GeminiLLM(LLM):
    provider = "gemini"
    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str, model: str = GEMINI_DEFAULT_MODEL, **kw):
        super().__init__(model, **kw)
        self.api_key = api_key
        self._switched = False

    def _pick_model(self) -> str | None:
        try:
            r = requests.get(f"{self.BASE}/models", headers={"x-goog-api-key": self.api_key},
                             params={"pageSize": 200}, timeout=30)
            models = r.json().get("models", []) if r.ok else []
        except (requests.RequestException, ValueError):
            return None
        names = [m["name"].split("/", 1)[-1] for m in models
                 if "generateContent" in (m.get("supportedGenerationMethods") or [])]
        flash = [n for n in names if "flash" in n and "lite" not in n and "image" not in n
                 and "tts" not in n and "live" not in n and "exp" not in n]

        def ver(n: str):
            m = re.search(r"gemini-(\d+)(?:\.(\d+))?", n)
            return (int(m.group(1)), int(m.group(2) or 0), "preview" not in n) if m else (0, 0, False)

        flash.sort(key=ver, reverse=True)
        return flash[0] if flash else (names[0] if names else None)

    def _call(self, system: str, user: str) -> str:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
                "maxOutputTokens": 32768,
            },
        }
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        try:
            data = self._post(f"{self.BASE}/models/{self.model}:generateContent", headers, body)
        except LLMError as e:
            # Имя модели устарело — выбираем актуальную flash-модель из доступных ключу.
            if "HTTP 404" not in str(e) or self._switched:
                raise
            self._switched = True
            new = self._pick_model()
            if not new:
                raise
            self._log(f"  модель {self.model} недоступна, переключаюсь на {new}")
            self.model = new
            data = self._post(f"{self.BASE}/models/{self.model}:generateContent", headers, body)
        cands = data.get("candidates") or []
        if not cands:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "пустой ответ")
            raise _Retryable(f"нет кандидатов ({reason})")
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            raise _Retryable(f"пустой текст (finishReason={cands[0].get('finishReason')})")
        return text


class OpenAICompatLLM(LLM):
    """OpenAI, Groq, OpenRouter, Ollama, vLLM — всё, что понимает /chat/completions."""

    provider = "openai-compatible"

    def __init__(self, api_key: str, model: str = OPENAI_DEFAULT_MODEL,
                 base_url: str = "https://api.openai.com/v1", **kw):
        super().__init__(model, **kw)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _call(self, system: str, user: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        data = self._post(f"{self.base_url}/chat/completions", headers, body)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise _Retryable("неожиданный формат ответа") from e


def from_env(provider: str | None = None, model: str | None = None, *,
             cache_dir: Path | None = None, verbose: bool = True) -> LLM:
    provider = (provider or os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if not provider:
        if os.environ.get("GEMINI_API_KEY"):
            provider = "gemini"
        elif os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
            provider = "openai"
        else:
            raise LLMError(
                "Не найден ключ LLM. Скопируйте .env.example в .env и укажите GEMINI_API_KEY "
                "(бесплатно: https://aistudio.google.com/apikey).")
    interval = float(os.environ.get("LLM_MIN_INTERVAL", "0") or 0)
    common = dict(cache_dir=cache_dir, verbose=verbose)
    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise LLMError("GEMINI_API_KEY не задан в .env")
        return GeminiLLM(key, model or os.environ.get("GEMINI_MODEL") or GEMINI_DEFAULT_MODEL,
                         min_interval=interval or 4.0, **common)
    if provider in ("openai", "openai-compatible", "groq", "openrouter", "ollama"):
        return OpenAICompatLLM(
            os.environ.get("OPENAI_API_KEY", ""),
            model or os.environ.get("OPENAI_MODEL") or OPENAI_DEFAULT_MODEL,
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            min_interval=interval, **common)
    raise LLMError(f"Неизвестный провайдер LLM: {provider}")
