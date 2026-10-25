"""Abstraksi classifier teks + backend scikit-learn.

Artefak model berupa direktori berisi:
- ``model.joblib``  : sklearn Pipeline (input = teks ter-normalisasi)  [backend sklearn]
- ``metadata.json`` : versi, label order, threshold, hash data, metrik

Kontrak metadata identik untuk backend transformers sehingga keduanya dapat
dipertukarkan lewat konfigurasi tanpa mengubah service.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from cekdulu.domain.schemas import Label
from cekdulu.ml.preprocessing import normalize_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelMetadata:
    model_version: str
    labels: list[Label]
    fraud_threshold: float
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> ModelMetadata:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            model_version=data["model_version"],
            labels=[Label(lbl) for lbl in data["labels"]],
            fraud_threshold=float(data["fraud_threshold"]),
            raw=data,
        )


class TextClassifier(Protocol):
    metadata: ModelMetadata

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Matriks (n_texts, n_labels), baris berjumlah 1, kolom mengikuti metadata.labels."""
        ...


class SklearnTextClassifier:
    def __init__(self, model_dir: Path) -> None:
        import joblib  # lazy: hanya dibutuhkan backend ini

        model_path = model_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan di {model_path}. Jalankan `make train` terlebih dahulu."
            )
        self.metadata = ModelMetadata.load(model_dir / "metadata.json")
        self._pipeline = joblib.load(model_path)
        classes = [Label(c) for c in self._pipeline.classes_]
        if classes != self.metadata.labels:
            raise ValueError(f"Urutan label model {classes} != metadata {self.metadata.labels}")
        logger.info(
            "model_loaded",
            extra={"backend": "sklearn", "model_version": self.metadata.model_version},
        )

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._pipeline.predict_proba([normalize_text(t) for t in texts]))


def load_classifier(backend: str, model_dir: Path) -> TextClassifier:
    if backend == "sklearn":
        return SklearnTextClassifier(model_dir)
    if backend == "transformers":
        from cekdulu.ml.transformers_backend import TransformersTextClassifier

        return TransformersTextClassifier(model_dir)
    raise ValueError(f"Backend classifier tidak dikenal: {backend}")
