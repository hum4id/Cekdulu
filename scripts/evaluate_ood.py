"""Evaluasi out-of-distribution: model saja vs model + RiskPolicy (aturan taktik).

Challenge set ``data/ood/challenge_2026_v0.csv`` DISUSUN PENGEMBANG berdasarkan modus yang
dipublikasikan OJK/IASC dan media 2025-2026 (bukan pesan nyata, belum dianotasi ganda).
Gunakan untuk mengukur *pergeseran distribusi*, bukan sebagai klaim akurasi lapangan.

    python scripts/evaluate_ood.py --model-dir models/baseline --data data/ood/challenge_2026_v0.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cekdulu.ml.classifier import load_classifier
from cekdulu.ml.risk import RiskPolicy
from cekdulu.ml.tactics import detect_tactics
from cekdulu.training.data import LABELS
from cekdulu.training.metrics import (
    bootstrap_ci,
    classification_metrics,
    fraud_precision,
    fraud_recall,
    predict_with_threshold,
)

FRAUD = "penipuan"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/baseline")
    ap.add_argument("--backend", default="sklearn")
    ap.add_argument("--data", default="data/ood/challenge_2026_v0.csv")
    ap.add_argument("--out", default="reports/baseline/ood_metrics.json")
    args = ap.parse_args()

    clf = load_classifier(args.backend, Path(args.model_dir))
    df = pd.read_csv(args.data)
    y = df["label"].to_numpy()
    proba = clf.predict_proba(df["text"].tolist())
    labels = [lbl.value for lbl in clf.metadata.labels]
    assert labels == LABELS
    thr = clf.metadata.fraud_threshold
    p_fraud = proba[:, labels.index(FRAUD)]
    pred_model = predict_with_threshold(proba, labels, FRAUD, thr)

    policy = RiskPolicy(thr, low_cutoff=min(0.25, thr / 2))
    decisions = [
        policy.decide(p, detect_tactics(t)) for p, t in zip(p_fraud, df["text"], strict=True)
    ]
    pred_policy = np.where([RiskPolicy.is_fraud(d.level) for d in decisions], FRAUD, pred_model)

    report = {
        "dataset": args.data,
        "n": len(df),
        "label_counts": df["label"].value_counts().to_dict(),
        "model_version": clf.metadata.model_version,
        "threshold": thr,
        "model_only": classification_metrics(y, pred_model, LABELS, p_fraud),
        "model_plus_policy": classification_metrics(y, pred_policy, LABELS, p_fraud),
        "bootstrap_95ci": {
            "model_only_fraud_recall": bootstrap_ci(y, pred_model, fraud_recall(FRAUD)),
            "policy_fraud_recall": bootstrap_ci(y, pred_policy, fraud_recall(FRAUD)),
            "policy_fraud_precision": bootstrap_ci(y, pred_policy, fraud_precision(FRAUD)),
        },
    }
    missed = df[(y == FRAUD) & (pred_policy != FRAUD)].assign(
        p_fraud=p_fraud[(y == FRAUD) & (pred_policy != FRAUD)].round(3)
    )
    false_alarm = df[(y != FRAUD) & (pred_policy == FRAUD)].assign(
        p_fraud=p_fraud[(y != FRAUD) & (pred_policy == FRAUD)].round(3)
    )
    report["missed_fraud"] = missed[["modus", "p_fraud", "text"]].to_dict(orient="records")
    report["false_alarms"] = false_alarm[["modus", "p_fraud", "text"]].to_dict(orient="records")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    for key in ("model_only", "model_plus_policy"):
        m = report[key]
        f = m["per_class"][FRAUD]
        print(
            f"{key:18s} macroF1={m['macro_f1']:.3f} fraud R={f['recall']:.3f} "
            f"P={f['precision']:.3f}"
        )
    print(f"missed fraud: {len(missed)} | false alarms: {len(false_alarm)}")
    for row in report["missed_fraud"]:
        print("  MISS", row["modus"], row["p_fraud"])
    for row in report["false_alarms"]:
        print("  FA  ", row["modus"], row["p_fraud"])


if __name__ == "__main__":
    main()
