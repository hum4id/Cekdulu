"""Konfigurasi aplikasi berbasis environment variable (12-factor).

Semua variabel memakai prefix ``CEKDULU_``. Lihat ``.env.example`` untuk daftar lengkap.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CEKDULU_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Runtime ---
    env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_json: bool = True

    # --- Keamanan API ---
    api_keys: str = Field(default="", description="Daftar API key dipisah koma.")
    hash_secret: SecretStr = SecretStr("dev-only-change-me")
    rate_limit_per_minute: int = 30
    cors_origins: str = ""
    max_upload_mb: float = 5.0
    max_text_chars: int = 5000

    # --- Model ---
    classifier_backend: Literal["sklearn", "transformers"] = "sklearn"
    model_dir: Path = Path("models/baseline")
    fraud_threshold: float | None = Field(
        default=None, description="Override threshold; default dibaca dari metadata model."
    )

    # --- LLM (opsional; sistem tetap jalan tanpa LLM) ---
    llm_provider: Literal["none", "gemini", "openai"] = "none"
    llm_api_key: SecretStr | None = None
    llm_models: str = "gemini-2.5-flash"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_timeout_s: float = 20.0

    # --- OCR ---
    ocr_tesseract: bool = True
    ocr_tesseract_lang: str = "ind+eng"

    # --- Telegram ---
    telegram_bot_token: SecretStr | None = None

    # --- Penyimpanan ---
    database_url: str = "sqlite:///./var/cekdulu.db"
    store_raw_text: bool = False
    report_ttl_hours: int = 24

    @property
    def api_key_list(self) -> list[str]:
        return _split_csv(self.api_keys)

    @property
    def llm_model_list(self) -> list[str]:
        return _split_csv(self.llm_models)

    @property
    def cors_origin_list(self) -> list[str]:
        return _split_csv(self.cors_origins)

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider != "none" and self.llm_api_key is not None

    @model_validator(mode="after")
    def _production_guards(self) -> Settings:
        if self.env == "production":
            if not self.api_key_list:
                raise ValueError("CEKDULU_API_KEYS wajib diisi di production.")
            if self.hash_secret.get_secret_value() in {"", "dev-only-change-me"}:
                raise ValueError("CEKDULU_HASH_SECRET wajib diganti di production.")
            if any(len(k) < 24 for k in self.api_key_list):
                raise ValueError("Setiap API key minimal 24 karakter di production.")
        if not 0.0 < (self.fraud_threshold or 0.5) < 1.0:
            raise ValueError("CEKDULU_FRAUD_THRESHOLD harus di antara 0 dan 1.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
