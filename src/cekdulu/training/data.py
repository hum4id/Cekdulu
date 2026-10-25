"""Loading, pembersihan, dan split dataset yang deterministik dan terdokumentasi."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from cekdulu.domain.schemas import Label
from cekdulu.ml.preprocessing import normalize_text

LABEL_MAP = {0: Label.NORMAL.value, 1: Label.PENIPUAN.value, 2: Label.PROMO.value}
LABELS = [Label.NORMAL.value, Label.PENIPUAN.value, Label.PROMO.value]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if set(df.columns) != {"Teks", "label"}:
        raise ValueError(f"Kolom tidak sesuai: {list(df.columns)}")
    df = df.rename(columns={"Teks": "text"})
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].map(LABEL_MAP)
    if df["label"].isna().any():
        raise ValueError("Ada label di luar {0,1,2}")
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Hapus teks kosong dan duplikat SETELAH normalisasi (mencegah near-duplicate leakage
    lintas split, mis. SMS template operator yang hanya beda angka)."""
    stats = {"rows_raw": len(df)}
    df = df[df["text"].str.len() > 0].copy()
    df["norm"] = df["text"].map(normalize_text)
    conflicts = df.groupby("norm")["label"].nunique()
    conflict_norms = set(conflicts[conflicts > 1].index)
    stats["dropped_label_conflicts"] = int(df["norm"].isin(conflict_norms).sum())
    df = df[~df["norm"].isin(conflict_norms)]
    before = len(df)
    df = df.drop_duplicates(subset="norm", keep="first")
    stats["dropped_duplicates"] = before - len(df)
    stats["rows_clean"] = len(df)
    return df.reset_index(drop=True), stats


def split(
    df: pd.DataFrame, seed: int, val_size: float, test_size: float
) -> dict[str, pd.DataFrame]:
    dev, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(dev, test_size=rel_val, stratify=dev["label"], random_state=seed)
    return {"train": train, "val": val, "test": test}


def write_splits(splits: dict[str, pd.DataFrame], out_dir: Path, manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(manifest)
    manifest["splits"] = {}
    for name, part in splits.items():
        path = out_dir / f"{name}.csv"
        part[["text", "label"]].to_csv(path, index=False)
        manifest["splits"][name] = {
            "rows": len(part),
            "label_counts": part["label"].value_counts().to_dict(),
            "sha256": sha256_file(path),
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def read_split(dir_: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(dir_ / f"{name}.csv")
