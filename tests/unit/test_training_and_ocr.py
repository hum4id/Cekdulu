import numpy as np
import pandas as pd
from PIL import Image

from cekdulu.llm.base import LLMClient, LLMError, NullLLM
from cekdulu.training.data import clean, split
from cekdulu.training.metrics import (
    bootstrap_ci,
    classification_metrics,
    fraud_recall,
    predict_with_threshold,
    tune_fraud_threshold,
)
from cekdulu.vision.ocr import OcrEngine

LABELS = ["normal", "penipuan", "promo"]


def test_predict_with_threshold_overrides_argmax():
    proba = np.array([[0.6, 0.35, 0.05], [0.1, 0.2, 0.7]])
    assert list(predict_with_threshold(proba, LABELS, "penipuan", 0.3)) == ["penipuan", "promo"]


def test_tune_threshold_meets_target_recall():
    y = np.array(["penipuan"] * 10 + ["normal"] * 10)
    p = np.array([0.9] * 8 + [0.4, 0.35] + [0.5, 0.3] + [0.1] * 8)
    res = tune_fraud_threshold(y, p, "penipuan", target_recall=1.0)
    assert res["recall"] == 1.0 and res["threshold"] <= 0.35
    assert res["strategy"] == "max_precision_at_target_recall"


def test_bootstrap_ci_contains_point():
    y = np.array(["penipuan"] * 20 + ["normal"] * 20)
    pred = y.copy()
    pred[:4] = "normal"
    ci = bootstrap_ci(y, pred, fraud_recall("penipuan"), n_boot=200)
    assert ci["ci_low"] <= ci["point"] == 0.8 <= ci["ci_high"]


def test_classification_metrics_shape():
    y = np.array(["normal", "penipuan", "promo", "penipuan"])
    m = classification_metrics(y, y, LABELS, p_fraud=np.array([0.1, 0.9, 0.2, 0.8]))
    assert m["macro_f1"] == 1.0 and m["fraud_roc_auc"] == 1.0
    assert m["confusion_matrix"]["matrix"][1][1] == 2


def test_clean_drops_normalized_duplicates_and_split_is_stratified():
    rows = [("Pulsa Rp10.000 masuk", "normal"), ("Pulsa Rp20.000 masuk", "normal")]
    rows += [(f"pesan normal {chr(97 + i)}", "normal") for i in range(18)]
    rows += [(f"menang undian hadiah {chr(97 + i)}", "penipuan") for i in range(20)]
    df = pd.DataFrame(rows, columns=["text", "label"])
    cleaned, stats = clean(df)
    assert stats["dropped_duplicates"] == 1 and len(cleaned) == 39
    parts = split(cleaned, seed=1, val_size=0.2, test_size=0.2)
    assert sum(len(p) for p in parts.values()) == 39
    assert set(parts["test"]["label"]) == {"normal", "penipuan"}


class FakeVisionLLM(LLMClient):
    def __init__(self, text: str | None) -> None:
        self.text = text

    def generate_json(self, system, user_text, schema, image_png=None):
        assert image_png is not None and image_png.startswith(b"\x89PNG")
        if self.text is None:
            raise LLMError("down")
        return {"text": self.text}


def test_ocr_prefers_llm_and_degrades_gracefully():
    img = Image.new("RGB", (50, 50), "white")
    assert OcrEngine(FakeVisionLLM("halo"), use_tesseract=False).extract(img) == ("halo", "llm")
    assert OcrEngine(FakeVisionLLM(None), use_tesseract=False).extract(img) == (None, None)
    engine = OcrEngine(NullLLM(), use_tesseract=False)
    assert engine.available is False and engine.extract(img) == (None, None)
