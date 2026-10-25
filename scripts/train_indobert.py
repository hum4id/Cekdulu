"""Fine-tune IndoBERT untuk klasifikasi normal/penipuan/promo (jalankan di GPU, mis. Colab T4).

Protokol: train untuk fitting, val untuk early stopping + tuning threshold, test SEKALI.
Output kompatibel dengan backend ``transformers`` di API (save_pretrained + metadata.json).

    pip install -e ".[transformers,train]"
    python scripts/train_indobert.py --config configs/indobert.yaml

Mode ``--smoke`` memakai BERT mini acak + tokenizer lokal (tanpa unduh bobot) untuk
memverifikasi pipeline end-to-end di CPU dalam < 1 menit. Hasil smoke TIDAK bermakna.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
import yaml
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from cekdulu.ml.preprocessing import normalize_text
from cekdulu.training.data import LABELS, read_split
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
LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int) -> None:  # type: ignore[no-untyped-def]
        self.enc = tokenizer(
            [normalize_text(t) for t in df["text"]], truncation=True, max_length=max_length
        )
        self.labels = [LABEL2ID[lbl] for lbl in df["label"]]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


class WeightedTrainer(Trainer):
    """Cross-entropy berbobot kelas (inverse frequency) untuk ketidakseimbangan label."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self._cw = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[no-untyped-def]
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.functional.cross_entropy(
            outputs.logits, labels, weight=self._cw.to(outputs.logits.device)
        )
        return (loss, outputs) if return_outputs else loss


def _warmup_kwargs(ratio: float) -> dict[str, float]:
    """transformers 4.x memakai ``warmup_ratio``; 5.x menerima rasio (float < 1) di
    ``warmup_steps``. Deteksi dari signature agar jalan di Colab (4.x/5.x) maupun lokal."""
    import inspect

    params = inspect.signature(TrainingArguments.__init__).parameters
    return {"warmup_ratio": ratio} if "warmup_ratio" in params else {"warmup_steps": ratio}


def compute_metrics(eval_pred) -> dict[str, float]:  # type: ignore[no-untyped-def]
    logits, labels = eval_pred
    pred = np.array(LABELS)[np.argmax(logits, axis=-1)]
    true = np.array(LABELS)[labels]
    return {
        "macro_f1": macro_f1(LABELS)(true, pred),
        "fraud_recall": fraud_recall(FRAUD)(true, pred),
    }


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def build_smoke_model(train_texts: list[str], out: Path):  # type: ignore[no-untyped-def]
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(
        train_texts,
        trainers.WordPieceTrainer(
            vocab_size=2000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        ),
    )
    from tokenizers.processors import TemplateProcessing

    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL2ID,
    )
    return tokenizer, BertForSequenceClassification(config)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/indobert.yaml")
    ap.add_argument("--smoke", action="store_true", help="Uji pipeline cepat tanpa unduh bobot")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["seed"])

    processed = Path(cfg["processed_dir"])
    train_df, val_df, test_df = (read_split(processed, s) for s in ("train", "val", "test"))
    model_dir, report_dir = Path(cfg["model_dir"]), Path(cfg["report_dir"])
    if args.smoke:
        train_df, val_df, test_df = train_df.head(120), val_df.head(60), test_df.head(60)
        model_dir, report_dir = (
            model_dir.with_name(model_dir.name + "_smoke"),
            report_dir.with_name(report_dir.name + "_smoke"),
        )
        cfg.update(epochs=1, train_batch_size=16, eval_batch_size=32, bootstrap_samples=50)
        tokenizer, model = build_smoke_model(
            [normalize_text(t) for t in train_df["text"]], model_dir
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg["pretrained"])
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["pretrained"],
            num_labels=len(LABELS),
            id2label=dict(enumerate(LABELS)),
            label2id=LABEL2ID,
        )

    ds = {
        name: TextDataset(df, tokenizer, cfg["max_length"])
        for name, df in (("train", train_df), ("val", val_df), ("test", test_df))
    }
    counts = train_df["label"].value_counts()
    weights = torch.tensor(
        [len(train_df) / (len(LABELS) * counts[lbl]) for lbl in LABELS], dtype=torch.float
    )

    targs = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        num_train_epochs=cfg["epochs"],
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=cfg["weight_decay"],
        **_warmup_kwargs(cfg["warmup_ratio"]),
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available() and not args.smoke,
        seed=cfg["seed"],
        report_to=[],
        logging_steps=20,
    )
    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])],
    )
    trainer.train()

    val_proba = softmax(trainer.predict(ds["val"]).predictions)
    thr = tune_fraud_threshold(
        val_df["label"].to_numpy(), val_proba[:, LABEL2ID[FRAUD]], FRAUD, cfg["target_fraud_recall"]
    )
    test_proba = softmax(trainer.predict(ds["test"]).predictions)
    y_test = test_df["label"].to_numpy()
    p_fraud = test_proba[:, LABEL2ID[FRAUD]]
    pred_thr = predict_with_threshold(test_proba, LABELS, FRAUD, thr["threshold"])
    pred_arg = np.array(LABELS)[test_proba.argmax(1)]
    n_boot = cfg["bootstrap_samples"]
    results = {
        "argmax": classification_metrics(y_test, pred_arg, LABELS, p_fraud),
        "tuned_threshold": classification_metrics(y_test, pred_thr, LABELS, p_fraud),
        "bootstrap_95ci_tuned": {
            "macro_f1": bootstrap_ci(y_test, pred_thr, macro_f1(LABELS), n_boot, cfg["seed"]),
            "fraud_recall": bootstrap_ci(
                y_test, pred_thr, fraud_recall(FRAUD), n_boot, cfg["seed"]
            ),
            "fraud_precision": bootstrap_ci(
                y_test, pred_thr, fraud_precision(FRAUD), n_boot, cfg["seed"]
            ),
        },
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    version = ("smoke-" if args.smoke else "") + f"indobert-{datetime.now(UTC):%Y%m%d%H%M}"
    manifest = json.loads((processed / "manifest.json").read_text())
    metadata = {
        "model_version": version,
        "backend": "transformers",
        "labels": LABELS,
        "fraud_threshold": round(thr["threshold"], 4),
        "threshold_selection": thr,
        "pretrained": None if args.smoke else cfg["pretrained"],
        "trained_on": "train (val untuk early stopping & threshold)",
        "data_source_sha256": manifest["source_sha256"],
        "split_sha256": {k: v["sha256"] for k, v in manifest["splits"].items()},
        "test_metrics_tuned": results["tuned_threshold"],
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    (report_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model_version": version,
                "threshold": thr,
                "test": results,
                "log_history": trainer.state.log_history,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
    print(
        json.dumps(
            {
                "model_version": version,
                "threshold": thr["threshold"],
                "macro_f1": results["tuned_threshold"]["macro_f1"],
                "fraud": results["tuned_threshold"]["per_class"][FRAUD],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
