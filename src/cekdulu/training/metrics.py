"""Metrik evaluasi, tuning threshold, dan bootstrap confidence interval."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def predict_with_threshold(
    proba: np.ndarray, labels: list[str], fraud_label: str, threshold: float
) -> np.ndarray:
    """Jika P(penipuan) >= threshold -> penipuan; selain itu argmax kelas lain."""
    fraud_idx = labels.index(fraud_label)
    masked = proba.copy()
    masked[:, fraud_idx] = -1
    other = np.array(labels)[masked.argmax(axis=1)]
    return np.where(proba[:, fraud_idx] >= threshold, fraud_label, other)


def tune_fraud_threshold(
    y_true: np.ndarray, p_fraud: np.ndarray, fraud_label: str, target_recall: float
) -> dict[str, float]:
    """Cari threshold dengan precision tertinggi yang masih memenuhi recall >= target.
    Bila tak ada yang memenuhi, pilih yang memaksimalkan F2 (recall lebih diutamakan)."""
    y = (y_true == fraud_label).astype(int)
    grid = np.unique(np.round(np.concatenate([p_fraud, np.linspace(0.05, 0.95, 91)]), 4))
    best: dict[str, float] | None = None
    best_f2: dict[str, float] | None = None
    for t in grid:
        pred = (p_fraud >= t).astype(int)
        tp = int((pred & y).sum())
        fp = int((pred & (1 - y)).sum())
        fn = int(((1 - pred) & y).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f2 = 5 * prec * rec / (4 * prec + rec) if prec + rec else 0.0
        row = {"threshold": float(t), "precision": prec, "recall": rec, "f2": f2}
        if rec >= target_recall and (
            best is None
            or prec > best["precision"]
            or (prec == best["precision"] and t > best["threshold"])
        ):
            best = row
        if best_f2 is None or f2 > best_f2["f2"]:
            best_f2 = row
    chosen = best or best_f2
    assert chosen is not None
    chosen["strategy"] = "max_precision_at_target_recall" if best else "max_f2_fallback"  # type: ignore[assignment]
    chosen["target_recall"] = target_recall
    return chosen


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    p_fraud: np.ndarray | None = None,
    fraud_label: str = "penipuan",
) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out: dict = {
        "n": len(y_true),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "accuracy": float((y_true == y_pred).mean()),
        "per_class": {
            lbl: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i, lbl in enumerate(labels)
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
    }
    if p_fraud is not None:
        y_bin = (y_true == fraud_label).astype(int)
        if 0 < y_bin.sum() < len(y_bin):
            out["fraud_roc_auc"] = float(roc_auc_score(y_bin, p_fraud))
            out["fraud_pr_auc"] = float(average_precision_score(y_bin, p_fraud))
    return out


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        scores.append(metric(y_true[idx], y_pred[idx]))
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return {"point": float(metric(y_true, y_pred)), "ci_low": float(lo), "ci_high": float(hi)}


def fraud_recall(fraud_label: str) -> Callable[[np.ndarray, np.ndarray], float]:
    def _m(y: np.ndarray, p: np.ndarray) -> float:
        mask = y == fraud_label
        return float((p[mask] == fraud_label).mean()) if mask.any() else float("nan")

    return _m


def fraud_precision(fraud_label: str) -> Callable[[np.ndarray, np.ndarray], float]:
    def _m(y: np.ndarray, p: np.ndarray) -> float:
        mask = p == fraud_label
        return float((y[mask] == fraud_label).mean()) if mask.any() else float("nan")

    return _m


def macro_f1(labels: list[str]) -> Callable[[np.ndarray, np.ndarray], float]:
    def _m(y: np.ndarray, p: np.ndarray) -> float:
        return float(f1_score(y, p, labels=labels, average="macro", zero_division=0))

    return _m
