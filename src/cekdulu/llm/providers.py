"""Provider LLM: Gemini (REST generateContent) dan OpenAI-compatible (chat/completions).

Nama model TIDAK di-hardcode: model LLM sering di-deprecate. Isi ``CEKDULU_LLM_MODELS``
dengan beberapa kandidat dipisah koma; bila model pertama 404, kandidat berikutnya dicoba.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx

from cekdulu.config import Settings
from cekdulu.llm.base import HttpLLMClient, LLMClient, LLMError, NullLLM

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiClient(HttpLLMClient):
    def __init__(
        self,
        api_key: str,
        models: list[str],
        timeout_s: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(models, timeout_s, transport=transport)
        self._api_key = api_key

    def _request(
        self,
        model: str,
        system: str,
        user_text: str,
        schema: dict[str, Any],
        image_png: bytes | None,
    ) -> httpx.Request:
        parts: list[dict[str, Any]] = [{"text": user_text}]
        if image_png is not None:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": base64.b64encode(image_png).decode(),
                    }
                }
            )
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        return self._client.build_request(
            "POST",
            f"{GEMINI_BASE}/models/{model}:generateContent",
            headers={"x-goog-api-key": self._api_key},
            json=body,
        )

    def _extract_text(self, body: dict[str, Any]) -> str:
        try:
            parts = body["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as exc:
            candidates = body.get("candidates") or [{}]
            reason = body.get("promptFeedback") or candidates[0].get("finishReason")
            raise LLMError(f"Respons Gemini tanpa konten (alasan: {reason})") from exc


class OpenAICompatibleClient(HttpLLMClient):
    def __init__(
        self,
        api_key: str,
        models: list[str],
        timeout_s: float,
        base_url: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(models, timeout_s, transport=transport)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def _request(
        self,
        model: str,
        system: str,
        user_text: str,
        schema: dict[str, Any],
        image_png: bytes | None,
    ) -> httpx.Request:
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        if image_png is not None:
            b64 = base64.b64encode(image_png).decode()
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            )
        system_full = (
            f"{system}\n\nBalas HANYA dengan objek JSON yang sesuai skema berikut:\n{schema}"
        )
        body = {
            "model": model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_full},
                {"role": "user", "content": content},
            ],
        }
        return self._client.build_request(
            "POST",
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=body,
        )

    def _extract_text(self, body: dict[str, Any]) -> str:
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Respons OpenAI-compatible tanpa konten") from exc


def build_llm(settings: Settings) -> LLMClient:
    if not settings.llm_enabled or settings.llm_api_key is None:
        return NullLLM()
    key = settings.llm_api_key.get_secret_value()
    if settings.llm_provider == "gemini":
        return GeminiClient(key, settings.llm_model_list, settings.llm_timeout_s)
    return OpenAICompatibleClient(
        key, settings.llm_model_list, settings.llm_timeout_s, settings.llm_base_url
    )
