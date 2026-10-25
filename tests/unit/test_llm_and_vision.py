import json

import httpx
import numpy as np
import pytest
from PIL import Image

from cekdulu.llm.base import LLMError, parse_json_object
from cekdulu.llm.providers import GeminiClient
from cekdulu.vision.image_io import InvalidImageError, load_image, to_bgr_array
from cekdulu.vision.qr import decode_qr_codes
from tests.conftest import make_qr_png


def gemini_ok(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def test_gemini_falls_back_to_next_model_on_404():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        assert request.headers["x-goog-api-key"] == "k"
        if "old-model" in request.url.path:
            return httpx.Response(404, json={"error": "not found"})
        body = json.loads(request.content)
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        return httpx.Response(200, json=gemini_ok('{"explanation": "aman"}'))

    client = GeminiClient(
        "k", ["old-model", "new-model"], 5, transport=httpx.MockTransport(handler)
    )
    assert client.generate_json("s", "u", {"type": "object"}) == {"explanation": "aman"}
    assert len(calls) == 2


def test_gemini_retries_then_fails_on_persistent_500(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    transport = httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))
    client = GeminiClient("k", ["m"], 5, transport=transport)
    with pytest.raises(LLMError):
        client.generate_json("s", "u", {"type": "object"})


def test_parse_json_object_handles_fences_and_rejects_garbage():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(LLMError):
        parse_json_object("bukan json")


def test_decode_qr_from_generated_png():
    payload = "https://example.com/hello"
    img = load_image(make_qr_png(payload), max_bytes=5_000_000)
    assert decode_qr_codes(to_bgr_array(img)) == [payload]


def test_decode_small_qr_via_upscale_variant():
    png = make_qr_png("000201010211", box_size=2)
    img = load_image(png, max_bytes=5_000_000)
    assert decode_qr_codes(to_bgr_array(img)) == ["000201010211"]


def test_no_qr_returns_empty():
    blank = Image.fromarray(np.full((200, 200, 3), 255, dtype=np.uint8))
    assert decode_qr_codes(to_bgr_array(blank)) == []


def test_load_image_rejects_non_image_and_oversize():
    with pytest.raises(InvalidImageError):
        load_image(b"not an image", max_bytes=1000)
    with pytest.raises(InvalidImageError):
        load_image(b"x" * 2000, max_bytes=1000)
