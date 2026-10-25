"""OCR berantai: LLM multimodal (bila aktif) -> Tesseract (bila terpasang) -> tidak ada.

Screenshot chat punya tata letak campur (bubble, emoji, timestamp); LLM multimodal biasanya
lebih tahan, sedangkan Tesseract menjadi fallback offline yang tidak mengirim data ke pihak
ketiga.
"""

from __future__ import annotations

import logging
import shutil

from PIL import Image

from cekdulu.llm.base import LLMClient, LLMError
from cekdulu.llm.prompts import OCR_SCHEMA, OCR_SYSTEM
from cekdulu.vision.image_io import to_png_bytes

logger = logging.getLogger(__name__)


class OcrEngine:
    def __init__(self, llm: LLMClient, use_tesseract: bool = True, lang: str = "ind+eng") -> None:
        self._llm = llm
        self._lang = lang
        self._tesseract = use_tesseract and shutil.which("tesseract") is not None
        if self._tesseract:
            try:
                import pytesseract  # noqa: F401
            except ImportError:
                self._tesseract = False

    @property
    def available(self) -> bool:
        return self._llm.enabled or self._tesseract

    def extract(self, img: Image.Image) -> tuple[str | None, str | None]:
        """Kembalikan (teks, sumber)."""
        if self._llm.enabled:
            try:
                data = self._llm.generate_json(
                    system=OCR_SYSTEM,
                    user_text="Transkripsikan teks pada gambar terlampir.",
                    schema=OCR_SCHEMA,
                    image_png=to_png_bytes(img),
                )
                text = str(data.get("text", "")).strip()
                if text:
                    return text[:5000], "llm"
            except LLMError as exc:
                logger.warning("ocr_llm_failed", extra={"error": str(exc)})
        if self._tesseract:
            import pytesseract

            try:
                text = pytesseract.image_to_string(img.convert("L"), lang=self._lang, timeout=20)
            except (RuntimeError, pytesseract.TesseractError) as exc:
                logger.warning("ocr_tesseract_failed", extra={"error": str(exc)})
                try:
                    text = pytesseract.image_to_string(img.convert("L"), lang="eng", timeout=20)
                except (RuntimeError, pytesseract.TesseractError):
                    return None, None
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            if text:
                return text[:5000], "tesseract"
        return None, None
