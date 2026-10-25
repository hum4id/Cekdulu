"""Antarmuka LLM yang provider-agnostic, dengan retry, fallback model, dan NullLLM.

Peran LLM di sistem ini SENGAJA dibatasi: OCR, ekstraksi, dan penjelasan bahasa awam.
Verdict risiko selalu berasal dari classifier + RiskPolicy agar terukur dan dapat dievaluasi.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class ModelNotFoundError(LLMError):
    pass


class LLMClient(ABC):
    enabled: bool = True

    @abstractmethod
    def generate_json(
        self,
        system: str,
        user_text: str,
        schema: dict[str, Any],
        image_png: bytes | None = None,
    ) -> dict[str, Any]:
        """Kembalikan objek JSON yang (diharapkan) sesuai ``schema``. Raise LLMError bila gagal."""


class NullLLM(LLMClient):
    enabled = False

    def generate_json(
        self, system: str, user_text: str, schema: dict[str, Any], image_png: bytes | None = None
    ) -> dict[str, Any]:
        raise LLMError("LLM tidak dikonfigurasi")


class HttpLLMClient(LLMClient):
    """Basis untuk provider HTTP: retry eksponensial pada 429/5xx, fallback antar model pada 404."""

    def __init__(
        self,
        models: list[str],
        timeout_s: float,
        max_retries: int = 2,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not models:
            raise ValueError("Minimal satu nama model LLM wajib diisi")
        self._models = models
        self._max_retries = max_retries
        self._client = httpx.Client(timeout=timeout_s, transport=transport)

    @abstractmethod
    def _request(
        self,
        model: str,
        system: str,
        user_text: str,
        schema: dict[str, Any],
        image_png: bytes | None,
    ) -> httpx.Request: ...

    @abstractmethod
    def _extract_text(self, body: dict[str, Any]) -> str: ...

    def generate_json(
        self, system: str, user_text: str, schema: dict[str, Any], image_png: bytes | None = None
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for model in self._models:
            try:
                return self._call_with_retry(model, system, user_text, schema, image_png)
            except ModelNotFoundError as exc:
                logger.warning("llm_model_not_found", extra={"model": model})
                last_error = exc
                continue
        raise LLMError(f"Semua model LLM gagal: {last_error}")

    def _call_with_retry(
        self,
        model: str,
        system: str,
        user_text: str,
        schema: dict[str, Any],
        image_png: bytes | None,
    ) -> dict[str, Any]:
        for attempt in range(self._max_retries + 1):
            start = time.perf_counter()
            try:
                resp = self._client.send(self._request(model, system, user_text, schema, image_png))
            except httpx.HTTPError as exc:
                if attempt == self._max_retries:
                    raise LLMError(f"Gagal menghubungi LLM: {exc}") from exc
                time.sleep(0.5 * 2**attempt)
                continue
            latency = round((time.perf_counter() - start) * 1000, 1)
            if resp.status_code == 404:
                raise ModelNotFoundError(model)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self._max_retries:
                time.sleep(0.5 * 2**attempt)
                continue
            if resp.status_code >= 400:
                raise LLMError(f"LLM HTTP {resp.status_code}: {resp.text[:200]}")
            text = self._extract_text(resp.json())
            logger.info("llm_call", extra={"model": model, "latency_ms": latency})
            return parse_json_object(text)
        raise LLMError("Retry LLM habis")


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise LLMError("Output LLM bukan JSON")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError("Output LLM bukan JSON valid") from exc
    if not isinstance(data, dict):
        raise LLMError("Output LLM bukan objek JSON")
    return data
