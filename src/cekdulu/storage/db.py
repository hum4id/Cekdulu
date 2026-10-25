"""Persistensi minimal (SQLAlchemy 2.0). SQLite untuk MVP; ganti ``CEKDULU_DATABASE_URL``
ke PostgreSQL untuk multi-instance (skema kompatibel).

Kebijakan data:
- ``analyses`` : hanya metadata + HMAC teks (teks mentah disimpan HANYA jika
  ``CEKDULU_STORE_RAW_TEXT=true``, mis. untuk relabeling dengan persetujuan pengguna).
- ``incidents``: nomor rekening disimpan termasking + HMAC (untuk korelasi rekening berulang).
- ``reports``  : PDF laporan (berisi data lengkap untuk bank/IASC) dengan masa berlaku
  ``CEKDULU_REPORT_TTL_HOURS`` lalu dihapus.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
    delete,
    event,
    func,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def utcnow() -> datetime:
    """UTC naive: SQLite tidak menyimpan zona waktu, jadi semua kolom waktu = UTC naive."""
    return datetime.now(UTC).replace(tzinfo=None)


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class AnalysisRecord(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    created_at: Mapped[datetime] = mapped_column(DateTime(), default=utcnow, index=True)
    kind: Mapped[str] = mapped_column(String(16))  # text | image | qris
    channel: Mapped[str] = mapped_column(String(32), default="api")
    label: Mapped[str | None] = mapped_column(String(16), nullable=True)
    risk_level: Mapped[str] = mapped_column(String(16), index=True)
    risk_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    text_hmac: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    text_len: Mapped[int | None] = mapped_column(Integer, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    tactics: Mapped[list[str]] = mapped_column(JSON, default=list)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)


class IncidentRecord(Base):
    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    created_at: Mapped[datetime] = mapped_column(DateTime(), default=utcnow)
    transfer_time: Mapped[datetime] = mapped_column(DateTime())
    amount_idr: Mapped[int | None] = mapped_column(Integer, nullable=True)
    destination_bank: Mapped[str | None] = mapped_column(String(64), nullable=True)
    account_masked: Mapped[str | None] = mapped_column(String(40), nullable=True)
    account_hmac: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    channel: Mapped[str | None] = mapped_column(String(50), nullable=True)


class ReportRecord(Base):
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    incident_id: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(), index=True)
    pdf: Mapped[bytes] = mapped_column(LargeBinary)


def make_engine(database_url: str) -> Engine:
    if database_url.startswith("sqlite:///") and ":memory:" not in database_url:
        Path(database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)
    if database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _pragma(dbapi_conn, _):  # type: ignore[no-untyped-def]
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=5000")
            cur.close()

    return engine


class Repository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._session = sessionmaker(engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self._engine)

    def ping(self) -> bool:
        with self._session() as s:
            s.execute(select(1))
        return True

    def add(self, record: Base) -> None:
        with self._session.begin() as s:
            s.add(record)

    def get_report(self, report_id: str) -> ReportRecord | None:
        with self._session() as s:
            rec = s.get(ReportRecord, report_id)
            if rec is None:
                return None
            return rec if rec.expires_at > utcnow() else None

    def count_account_reports(self, account_hmac: str) -> int:
        with self._session() as s:
            stmt = (
                select(func.count())
                .select_from(IncidentRecord)
                .where(IncidentRecord.account_hmac == account_hmac)
            )
            return int(s.execute(stmt).scalar_one())

    def purge(self, analyses_older_than: datetime) -> dict[str, int]:
        with self._session.begin() as s:
            reports = s.execute(delete(ReportRecord).where(ReportRecord.expires_at < utcnow()))
            analyses = s.execute(
                delete(AnalysisRecord).where(AnalysisRecord.created_at < analyses_older_than)
            )
        return {"reports": reports.rowcount or 0, "analyses": analyses.rowcount or 0}

    def session(self) -> Session:
        return self._session()

    def close(self) -> None:
        self._engine.dispose()
