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
import tempfile
from pathlib import Path

import requests

# Стабильная (не preview) модель: preview-версии Google снимает и урезает без предупреждения.
GEMINI_DEFAULT_MODEL = "gemini-3.5-flash"
# Запасные модели на случай 404/429/503; переопределяются GEMINI_FALLBACK_MODELS в .env
GEMINI_FALLBACKS = ["gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.6-flash",
                    "gemini-flash-latest", "gemini-3-flash-preview"]
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


class LLMError(Exception):
    pass


class LLMBadJSON(LLMError):
    """Модель несколько раз подряд вернула ответ, который не разбирается как JSON.
    Вызывающий код может уменьшить запрос (например, разбить пакет) и спросить снова."""


# Сколько раз просить модель исправить свой невалидный JSON, прежде чем сдаться.
JSON_REPAIRS = 2
# Сколько символов плохого ответа показывать модели при просьбе исправить его.
_REPAIR_ECHO = 20000


class _Retryable(Exception):
    def __init__(self, msg: str, wait: float | None = None, status: int = 0,
                 daily_quota: bool = False):
        super().__init__(msg)
        self.wait = wait
        self.status = status
        self.daily_quota = daily_quota

    @property
    def model_unavailable(self) -> bool:
        """Перегрузка (5xx) или долгая блокировка квоты — есть смысл сменить модель.
        Минутный лимит (429 с коротким retryDelay) — дешевле подождать."""
        if self.status >= 500:
            return True
        return self.status == 429 and (self.wait is None or self.wait > 65)


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
                 max_retries: int = 8, overload_patience: int = 6, timeout: float = 600.0,
                 temperature: float = 0.0, verbose: bool = True):
        self.model = model
        # Неудач 5xx подряд до смены модели: между ними паузы 5+10+20+40+65 с ≈ 2,5 минуты.
        self.overload_patience = min(overload_patience, max_retries)
        # Ключ кэша строится от запрошенной модели, а не от текущей: после перехода
        # на запасную модель повторный прогон всё равно находит ответы в кэше.
        self.requested_model = model
        self.cache_dir = cache_dir
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self.temperature = temperature
        self.verbose = verbose
        self.calls = 0          # реальные обращения к API
        self.retries = 0
        self.request_log = []
        self.cache_hits = 0
        self.models_used: list[str] = []
        # (номер ключа, модель) с исчерпанной дневной квотой
        self._exhausted: set[tuple[int, str]] = set()
        self._key_idx = 0
        self._last = 0.0

    @property
    def label(self) -> str:
        used = self.models_used or [self.model]
        return f"{self.provider} / {', '.join(used)}"

    def _should_switch(self, e: _Retryable, fails: int) -> bool:
        """Когда менять ключ или модель, а когда подождать.

        Смена модели портит единообразие отчёта, поэтому переключаемся неохотно:
        - дневная квота — сразу: она не восстановится до завтра;
        - долгая блокировка квоты (429 с ожиданием больше минуты) — после двух неудач;
        - перегрузка сервера (5xx) обычно проходит за минуты — ждём с нарастающей
          паузой и меняем модель только после overload_patience неудач подряд."""
        if e.daily_quota:
            return True
        if e.status >= 500:
            return fails >= self.overload_patience
        return e.model_unavailable and fails >= 2

    def _mark_exhausted(self) -> None:
        self._exhausted.add((self._key_idx, self.model))

    def _rotate(self, reason: str, daily: bool = False) -> bool:
        """Переключиться на другой ключ или запасную модель. По умолчанию их нет."""
        return False

    def complete_json(self, system: str, user: str, *, step: str = "") -> dict:
        key = hashlib.sha256(
            json.dumps([self.provider, self.requested_model, self.temperature, system, user]).encode()
        ).hexdigest()[:32]
        cached = self._cache_get(key)
        if cached is not None:
            data, answered_by = cached
            self.cache_hits += 1
            if answered_by and answered_by not in self.models_used:
                self.models_used.append(answered_by)
            self._log(f"  [{step}] ответ из кэша ({answered_by or 'модель не записана'})")
            return data

        prompt = user
        last_err = ""
        fails_on_model = 0
        bad_json = 0
        attempt = 0
        # Лимит попыток считается на модель: переход на запасную модель даёт ей
        # полный набор попыток, иначе до последних запасных дело не доходит.
        requests_made = 0
        while attempt < self.max_retries:
            attempt += 1
            self._throttle()
            try:
                t0 = time.time()
                if requests_made:
                    self.retries += 1
                requests_made += 1
                entry = {"step": step, "model": self.model, "retry": requests_made > 1}
                self.request_log.append(entry)
                start = time.monotonic()
                try:
                    text = self._call(system, prompt)
                finally:
                    entry["duration_s"] = round(time.monotonic() - start, 3)
                self.calls += 1
                if self.model not in self.models_used:
                    self.models_used.append(self.model)
                self._log(f"  [{step}] ответ модели {self.model} за {time.time() - t0:.1f} с")
            except _Retryable as e:
                last_err = str(e)
                fails_on_model += 1
                if e.daily_quota:
                    self._mark_exhausted()
                if self._should_switch(e, fails_on_model) and self._rotate(last_err,
                                                                          daily=e.daily_quota):
                    fails_on_model = 0
                    attempt = 0
                    time.sleep(2)
                    continue
                if e.daily_quota:  # запасных нет, а квота кончилась до завтра — не ждём зря
                    break
                wait = min(e.wait or 5.0 * 2 ** (attempt - 1), 65.0)
                self._log(f"  [{step}] {e}; повтор через {wait:.0f} с ({attempt}/{self.max_retries})")
                time.sleep(wait)
                continue
            try:
                data = parse_json(text)
            except (ValueError, json.JSONDecodeError) as e:
                last_err = f"невалидный JSON: {e}"
                bad_json += 1
                if bad_json > JSON_REPAIRS:
                    raise LLMBadJSON(f"Модель не дала корректный JSON на шаге «{step}»: {e}")
                # Повтор того же запроса при температуре 0 даёт тот же ответ. Поэтому
                # показываем модели её ответ и ошибку разбора и просим исправить именно его.
                self._log(f"  [{step}] {last_err}; прошу модель исправить свой ответ")
                prompt = (user + "\n\nТвой предыдущий ответ не разбирается как JSON. Ошибка "
                          f"разборщика: {e}. Вот этот ответ:\n<<<\n{text[:_REPAIR_ECHO]}\n>>>\n"
                          "Верни тот же ответ, исправив синтаксис: один JSON-объект целиком, "
                          "без пояснений и без markdown. Проверь запятые между элементами и "
                          "экранирование кавычек внутри строк.")
                attempt -= 1  # ошибка формата не тратит попытки на перегрузку сервера
                continue
            self._cache_put(key, data, self.model)
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

    def _cache_get(self, key: str) -> tuple[dict, str] | None:
        """Возвращает (ответ, модель, которая его дала)."""
        if not self.cache_dir:
            return None
        f = self.cache_dir / f"{key}.json"
        if not f.exists():
            return None
        try:
            entry = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(entry, dict):
            return None
        if set(entry) == {"model", "response"} and isinstance(entry["response"], dict):
            return entry["response"], str(entry["model"] or "")
        return entry, ""  # запись старого формата: только ответ

    def _cache_put(self, key: str, data: dict, model: str) -> None:
        if not self.cache_dir:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = self.cache_dir / f"{key}.json"
        with tempfile.NamedTemporaryFile(dir=self.cache_dir, delete=False) as stream:
            temp = Path(stream.name)
        try:
            temp.write_text(json.dumps({"model": model, "response": data}, ensure_ascii=False),
                            encoding="utf-8")
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)

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
            m = re.search(r'"retryDelay":\s*"(\d+)(?:\.\d+)?s"', r.text)
            if m:
                wait = float(m.group(1)) + 1
            daily = r.status_code == 429 and bool(re.search(r"PerDay", r.text))
            if daily:
                wait = None  # дневная квота: ждать бессмысленно, нужна другая модель
            raise _Retryable(f"HTTP {r.status_code}" + (" (дневная квота)" if daily else ""),
                             wait, r.status_code, daily_quota=daily)
        if r.status_code >= 400:
            raise LLMError(f"HTTP {r.status_code} от {self.provider}: {r.text[:500]}")
        return r.json()


# Ответы, после которых ключ бесполезен до конца прогона.
_KEY_REJECTED = re.compile(r"HTTP 40[13]\b|API_KEY_INVALID|PERMISSION_DENIED")


class GeminiLLM(LLM):
    provider = "gemini"
    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str | list[str], model: str = GEMINI_DEFAULT_MODEL,
                 fallbacks: list[str] | None = None, **kw):
        super().__init__(model, **kw)
        keys = [api_key] if isinstance(api_key, str) else list(api_key)
        self.api_keys = [k for k in keys if k] or [""]
        chain = [model] + [m for m in (fallbacks or GEMINI_FALLBACKS) if m != model]
        self._queue = chain[1:]
        self._discovered = False
        self._dead_keys: set[int] = set()

    @property
    def api_key(self) -> str:
        return self.api_keys[self._key_idx]

    def _free_key(self, model: str) -> int | None:
        """Первый рабочий ключ, у которого для модели ещё есть дневная квота."""
        return next((i for i in range(len(self.api_keys))
                     if i not in self._dead_keys and (i, model) not in self._exhausted), None)

    def _drop_key(self, reason: str) -> bool:
        """Ключ отклонён сервером: исключить его до конца прогона и взять следующий."""
        self._dead_keys.add(self._key_idx)
        k = self._free_key(self.model)
        if k is None:
            return False
        self._log(f"  ключ №{self._key_idx + 1} отклонён ({reason}), переключаюсь на ключ №{k + 1}")
        self._key_idx = k
        return True

    def _rotate(self, reason: str, daily: bool = False) -> bool:
        # Дневная квота кончилась на этом ключе: сначала другой ключ с той же моделью —
        # качество отчёта не падает. Квота считается на проект Google, поэтому ключи
        # должны быть из разных проектов.
        if daily and len(self.api_keys) > 1:
            k = self._free_key(self.model)
            if k is not None:
                self._log(f"  ключ №{self._key_idx + 1}: дневная квота {self.model} исчерпана, "
                          f"переключаюсь на ключ №{k + 1}")
                self._key_idx = k
                return True
        if not self._queue and not self._discovered:
            self._discovered = True
            picked = self._pick_model()
            if picked and picked != self.model:
                self._queue.append(picked)
        self._queue = [m for m in self._queue if self._free_key(m) is not None]
        if not self._queue:
            return False
        new = self._queue.pop(0)
        self._log(f"  модель {self.model} недоступна ({reason}), переключаюсь на {new}")
        self.model = new
        self._key_idx = self._free_key(new)
        return True

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
        while True:
            headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
            try:
                data = self._post(f"{self.BASE}/models/{self.model}:generateContent", headers, body)
                break
            except LLMError as e:
                msg = str(e)
                # Ключ отклонён (нет бесплатного тарифа у проекта, ключ недействителен) —
                # он не заработает до конца прогона, берём следующий.
                if _KEY_REJECTED.search(msg) and self._drop_key(msg.split(":", 1)[0]):
                    continue
                # Модель снята для новых пользователей (404) — сразу берём запасную.
                if "HTTP 404" not in msg or not self._rotate("HTTP 404"):
                    raise
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


def gemini_keys_from_env() -> list[str]:
    """Ключи по порядку: GEMINI_API_KEY (можно несколько через запятую), затем
    GEMINI_API_KEY_2 … GEMINI_API_KEY_9. Повторы отбрасываются."""
    raw = os.environ.get("GEMINI_API_KEY", "").split(",")
    raw += [os.environ.get(f"GEMINI_API_KEY_{i}", "") for i in range(2, 10)]
    keys: list[str] = []
    for k in (x.strip() for x in raw):
        if k and k not in keys:
            keys.append(k)
    return keys


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
        keys = gemini_keys_from_env()
        if not keys:
            raise LLMError("GEMINI_API_KEY не задан в .env")
        fb = [m.strip() for m in os.environ.get("GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()]
        return GeminiLLM(keys, model or os.environ.get("GEMINI_MODEL") or GEMINI_DEFAULT_MODEL,
                         fallbacks=fb or None, min_interval=interval or 13.0, **common)
    if provider in ("openai", "openai-compatible", "groq", "openrouter", "ollama"):
        return OpenAICompatLLM(
            os.environ.get("OPENAI_API_KEY", ""),
            model or os.environ.get("OPENAI_MODEL") or OPENAI_DEFAULT_MODEL,
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            min_interval=interval, **common)
    raise LLMError(f"Неизвестный провайдер LLM: {provider}")
