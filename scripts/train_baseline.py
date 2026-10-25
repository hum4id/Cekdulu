"""Latih & evaluasi baseline TF-IDF (word + char) + Logistic Regression.

Protokol (mencegah kebocoran test):
1. Seleksi model & hyperparameter: Stratified K-Fold CV pada train+val (dev set).
2. Tuning threshold P(penipuan): dari prediksi out-of-fold model terpilih pada dev set,
   target recall penipuan >= target_fraud_recall.
3. Fit ulang model terpilih pada seluruh dev set.
4. Evaluasi SEKALI pada test set + bootstrap 95% CI. Baseline pembanding dievaluasi
   dengan protokol yang sama.

    python scripts/train_baseline.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline

from cekdulu.ml.preprocessing import normalize_text
from cekdulu.training.data import LABELS, read_split, sha256_file
from cekdulu.training.metrics import (
    bootstrap_ci,
    classification_metrics,
    fraud_precision,
    fraud_recall,
    macro_f1,
    predict_with_threshold,
    tune_fraud_threshold,
)

FRAUD = "penipuan"


def build_word_char_lr(c: float, seed: int) -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True
                ),
            ),
        ]
    )
    clf = LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=seed)
    return Pipeline([("features", features), ("clf", clf)])


def candidates(cfg: dict) -> dict[str, Pipeline]:
    seed = cfg["seed"]
    models: dict[str, Pipeline] = {
        "majority": Pipeline(
            [("features", TfidfVectorizer()), ("clf", DummyClassifier(strategy="most_frequent"))]
        ),
        "word_tfidf_complementnb": Pipeline(
            [
                ("features", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
                ("clf", ComplementNB()),
            ]
        ),
        "word_tfidf_lr": Pipeline(
            [
                ("features", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                (
                    "clf",
                    LogisticRegression(
                        C=5.0, class_weight="balanced", max_iter=5000, random_state=seed
                    ),
                ),
            ]
        ),
    }
    for c in cfg["c_grid"]:
        models[f"word_char_tfidf_lr_C{c}"] = build_word_char_lr(c, seed)
    return models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    cfg = yaml.safe_load(Path(ap.parse_args().config).read_text())
    seed = cfg["seed"]
    np.random.seed(seed)
    processed = Path(cfg["processed_dir"])
    model_dir, report_dir = Path(cfg["model_dir"]), Path(cfg["report_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    dev = pd.concat(
        [read_split(processed, "train"), read_split(processed, "val")], ignore_index=True
    )
    test = read_split(processed, "test")
    x_dev = dev["text"].map(normalize_text).to_numpy()
    y_dev = dev["label"].to_numpy()
    x_test = test["text"].map(normalize_text).to_numpy()
    y_test = test["label"].to_numpy()

    cv = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=seed)
    cv_rows = []
    oof_store: dict[str, np.ndarray] = {}
    for name, pipe in candidates(cfg).items():
        oof = cross_val_predict(pipe, x_dev, y_dev, cv=cv, method="predict_proba")
        classes = sorted(set(y_dev))  # urutan kelas sklearn = sorted
        oof = oof[:, [classes.index(lbl) for lbl in LABELS]]
        pred = np.array(LABELS)[oof.argmax(1)]
        cv_rows.append(
            {
                "model": name,
                "cv_macro_f1": macro_f1(LABELS)(y_dev, pred),
                "cv_fraud_recall_argmax": fraud_recall(FRAUD)(y_dev, pred),
            }
        )
        oof_store[name] = oof
        print(f"[cv] {name}: macro-F1={cv_rows[-1]['cv_macro_f1']:.4f}")

    cv_table = pd.DataFrame(cv_rows).sort_values("cv_macro_f1", ascending=False)
    selectable = cv_table[cv_table["model"].str.startswith("word_char_tfidf_lr")].copy()
    top = selectable["cv_macro_f1"].max()
    within = selectable[selectable["cv_macro_f1"] >= top - cfg["selection_tolerance"]].copy()
    within["C"] = within["model"].str.extract(r"_C([\d.]+)$")[0].astype(float)
    best_name = within.sort_values("C").iloc[0]["model"]  # paling teregularisasi
    oof_best = oof_store[best_name]
    thr = tune_fraud_threshold(
        y_dev, oof_best[:, LABELS.index(FRAUD)], FRAUD, cfg["target_fraud_recall"]
    )
    print(f"[select] {best_name}; threshold={thr}")

    results: dict[str, dict] = {}
    final_model: Pipeline | None = None
    for name, pipe in candidates(cfg).items():
        if name.startswith("word_char_tfidf_lr") and name != best_name:
            continue
        pipe.fit(x_dev, y_dev)
        proba = pipe.predict_proba(x_test)
        proba = proba[:, [list(pipe.classes_).index(lbl) for lbl in LABELS]]
        p_fraud = proba[:, LABELS.index(FRAUD)]
        pred_argmax = np.array(LABELS)[proba.argmax(1)]
        entry = {"argmax": classification_metrics(y_test, pred_argmax, LABELS, p_fraud)}
        if name == best_name:
            final_model = pipe
            pred_thr = predict_with_threshold(proba, LABELS, FRAUD, thr["threshold"])
            entry["tuned_threshold"] = classification_metrics(y_test, pred_thr, LABELS, p_fraud)
            entry["bootstrap_95ci_tuned"] = {
                "macro_f1": bootstrap_ci(
                    y_test, pred_thr, macro_f1(LABELS), cfg["bootstrap_samples"], seed
                ),
                "fraud_recall": bootstrap_ci(
                    y_test, pred_thr, fraud_recall(FRAUD), cfg["bootstrap_samples"], seed
                ),
                "fraud_precision": bootstrap_ci(
                    y_test, pred_thr, fraud_precision(FRAUD), cfg["bootstrap_samples"], seed
                ),
            }
            errors = test.assign(pred=pred_thr, p_fraud=p_fraud.round(4))
            errors[errors["pred"] != errors["label"]].to_csv(
                report_dir / "test_errors.csv", index=False
            )
        results[name] = entry
    assert final_model is not None

    # Latensi inference CPU (1 pesan per panggilan, seperti di produksi)
    sample = [normalize_text(t) for t in test["text"].head(200)]
    timings = []
    for text in sample:
        t0 = time.perf_counter()
        final_model.predict_proba([text])
        timings.append((time.perf_counter() - t0) * 1000)
    latency = {
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
    }

    version = f"baseline-wordchar-lr-{datetime.now(UTC):%Y%m%d%H%M}"
    joblib.dump(final_model, model_dir / "model.joblib", compress=3)
    manifest = json.loads((processed / "manifest.json").read_text())
    metadata = {
        "model_version": version,
        "backend": "sklearn",
        "labels": LABELS,
        "fraud_threshold": round(thr["threshold"], 4),
        "threshold_selection": thr,
        "selected_model": best_name,
        "trained_on": "train+val (dev set)",
        "data_source_sha256": manifest["source_sha256"],
        "split_sha256": {k: v["sha256"] for k, v in manifest["splits"].items()},
        "test_metrics_tuned": results[best_name]["tuned_threshold"],
        "inference_latency_cpu": latency,
        "env": {"python": platform.python_version(), "sklearn": sklearn.__version__},
        "created_at": datetime.now(UTC).isoformat(),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    report = {
        "model_version": version,
        "cv": cv_table.to_dict(orient="records"),
        "threshold": thr,
        "test": results,
        "latency": latency,
        "model_sha256": sha256_file(model_dir / "model.joblib"),
    }
    (report_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    _plot_confusion(
        results[best_name]["tuned_threshold"]["confusion_matrix"],
        report_dir / "confusion_matrix.png",
    )
    print(
        json.dumps(
            {
                "model_version": version,
                "threshold": thr["threshold"],
                "test_tuned": {
                    k: results[best_name]["tuned_threshold"][k] for k in ("macro_f1", "accuracy")
                },
                "fraud": results[best_name]["tuned_threshold"]["per_class"][FRAUD],
                "latency": latency,
            },
            indent=2,
        )
    )


def _plot_confusion(cm: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.array(cm["matrix"])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(cm["labels"])), cm["labels"])
    ax.set_yticks(range(len(cm["labels"])), cm["labels"])
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label sebenarnya")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
            )
    ax.set_title("Confusion matrix (test, threshold dituning)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
