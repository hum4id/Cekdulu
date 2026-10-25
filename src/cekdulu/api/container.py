"""Composition root: membangun seluruh dependensi sekali saat startup."""

from __future__ import annotations

from dataclasses import dataclass

from cekdulu.config import Settings
from cekdulu.llm.providers import build_llm
from cekdulu.ml.classifier import load_classifier
from cekdulu.services.analysis import AnalysisService
from cekdulu.services.telegram import TelegramUpdateHandler
from cekdulu.storage.db import Repository, make_engine
from cekdulu.vision.ocr import OcrEngine


@dataclass
class Container:
    settings: Settings
    service: AnalysisService
    telegram: TelegramUpdateHandler
    repo: Repository


def build_container(settings: Settings) -> Container:
    classifier = load_classifier(settings.classifier_backend, settings.model_dir)
    llm = build_llm(settings)
    repo = Repository(make_engine(settings.database_url))
    repo.create_schema()
    ocr = OcrEngine(llm, use_tesseract=settings.ocr_tesseract, lang=settings.ocr_tesseract_lang)
    service = AnalysisService(settings, classifier, llm, repo, ocr)
    token = settings.telegram_bot_token.get_secret_value() if settings.telegram_bot_token else None
    return Container(settings, service, TelegramUpdateHandler(service, token), repo)
