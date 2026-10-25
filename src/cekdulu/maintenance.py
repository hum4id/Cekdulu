"""Tugas pemeliharaan: retensi data. Jadwalkan harian (cron/k8s CronJob/n8n Schedule).

``python -m cekdulu.maintenance --days 30``
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta

from cekdulu.config import get_settings
from cekdulu.storage.db import Repository, make_engine, utcnow


def main() -> None:
    parser = argparse.ArgumentParser(description="Hapus laporan kedaluwarsa & analisis lama.")
    parser.add_argument("--days", type=int, default=30, help="Retensi tabel analyses (hari)")
    args = parser.parse_args()
    repo = Repository(make_engine(get_settings().database_url))
    repo.create_schema()
    print(json.dumps(repo.purge(utcnow() - timedelta(days=args.days))))


if __name__ == "__main__":
    main()
