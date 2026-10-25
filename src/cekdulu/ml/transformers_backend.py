"""Backend IndoBERT (Hugging Face). Dipakai bila ``CEKDULU_CLASSIFIER_BACKEND=transformers``.

Direktori model adalah output ``scripts/train_indobert.py`` (``save_pretrained`` +
``metadata.json``). Inference di CPU dengan ``torch.inference_mode`` dan batching kecil.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from cekdulu.domain.schemas import Label
from cekdulu.ml.classifier import ModelMetadata
from cekdulu.ml.preprocessing import normalize_text

logger = logging.getLogger(__name__)


class TransformersTextClassifier:
    def __init__(self, model_dir: Path, max_length: int = 128, batch_size: int = 16) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.metadata = ModelMetadata.load(model_dir / "metadata.json")
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self._model.eval()
        self._max_length = max_length
        self._batch_size = batch_size
        id2label = {int(k): Label(v) for k, v in self._model.config.id2label.items()}
        ordered = [id2label[i] for i in range(len(id2label))]
        if ordered != self.metadata.labels:
            raise ValueError(f"id2label {ordered} != metadata {self.metadata.labels}")
        torch.set_num_threads(max(1, torch.get_num_threads()))
        logger.info(
            "model_loaded",
            extra={"backend": "transformers", "model_version": self.metadata.model_version},
        )

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        outputs: list[np.ndarray] = []
        normalized = [normalize_text(t) for t in texts]
        with self._torch.inference_mode():
            for i in range(0, len(normalized), self._batch_size):
                batch = self._tokenizer(
                    normalized[i : i + self._batch_size],
                    truncation=True,
                    max_length=self._max_length,
                    padding=True,
                    return_tensors="pt",
                )
                logits = self._model(**batch).logits
                outputs.append(self._torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(outputs, axis=0)
