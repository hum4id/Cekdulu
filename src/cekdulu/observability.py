"""Metrik Prometheus (diekspos di /metrics)."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

ANALYSES_TOTAL = Counter(
    "cekdulu_analyses_total", "Jumlah analisis per jenis dan level risiko", ["kind", "risk_level"]
)
LLM_FALLBACK_TOTAL = Counter(
    "cekdulu_llm_fallback_total", "Jumlah kegagalan LLM yang jatuh ke template"
)
HTTP_LATENCY = Histogram(
    "cekdulu_http_request_seconds",
    "Latensi request HTTP",
    ["method", "route", "status"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
