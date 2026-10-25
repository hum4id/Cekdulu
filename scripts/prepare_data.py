"""Bersihkan dataset mentah dan buat split train/val/test yang deterministik.

python scripts/prepare_data.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from cekdulu.training.data import clean, load_raw, sha256_file, split, write_splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    cfg = yaml.safe_load(Path(ap.parse_args().config).read_text())

    raw_path = Path(cfg["raw_path"])
    digest = sha256_file(raw_path)
    if digest != cfg["raw_sha256"]:
        raise SystemExit(f"SHA-256 dataset berbeda dari yang terdaftar: {digest}")

    df, stats = clean(load_raw(raw_path))
    splits = split(df, seed=cfg["seed"], val_size=cfg["val_size"], test_size=cfg["test_size"])
    manifest = {
        "source": str(raw_path),
        "source_sha256": digest,
        "seed": cfg["seed"],
        "cleaning": stats,
    }
    write_splits(splits, Path(cfg["processed_dir"]), manifest)
    print(json.dumps(manifest | {k: len(v) for k, v in splits.items()}, indent=2))


if __name__ == "__main__":
    main()
