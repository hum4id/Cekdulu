from __future__ import annotations

import io
from pathlib import Path

import pytest
import qrcode

from cekdulu.config import Settings

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "baseline"
API_KEY = "test-key-0123456789abcdefghij"


def make_qr_png(payload: str, box_size: int = 8) -> bytes:
    img = qrcode.make(payload, box_size=box_size, border=4)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        env="test",
        api_keys=API_KEY,
        hash_secret="test-secret",  # type: ignore[arg-type]
        model_dir=MODEL_DIR,
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        rate_limit_per_minute=1000,
        log_json=False,
        log_level="WARNING",
        ocr_tesseract=False,
    )


requires_model = pytest.mark.skipif(
    not (MODEL_DIR / "model.joblib").exists(), reason="model baseline belum dilatih (make train)"
)
