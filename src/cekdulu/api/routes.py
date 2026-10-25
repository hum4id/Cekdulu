from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, File, Form, HTTPException, Response, UploadFile, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool

from cekdulu import __version__
from cekdulu.api.deps import AuthDep, ContainerDep
from cekdulu.domain.schemas import (
    BotReply,
    ImageAnalysisResult,
    IncidentCreate,
    IncidentResult,
    QrisAssessment,
    QrisParseRequest,
    TextAnalysisRequest,
    TextAnalysisResult,
)
from cekdulu.vision.image_io import InvalidImageError

ops = APIRouter(tags=["ops"])
v1 = APIRouter(prefix="/v1")


@ops.get("/healthz", summary="Liveness probe")
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@ops.get("/readyz", summary="Readiness probe (model + DB)")
def readyz(container: ContainerDep) -> dict[str, Any]:
    try:
        container.repo.ping()
    except Exception as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"DB tidak siap: {exc}") from exc
    meta = container.service.classifier.metadata
    return {
        "status": "ready",
        "model_version": meta.model_version,
        "backend": container.settings.classifier_backend,
        "fraud_threshold": container.service.policy.fraud_threshold,
        "llm_enabled": container.service.llm.enabled,
        "ocr_available": container.service.ocr.available,
    }


@ops.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@v1.post("/analyze/text", response_model=TextAnalysisResult, tags=["analysis"])
def analyze_text(
    req: TextAnalysisRequest, container: ContainerDep, _: AuthDep
) -> TextAnalysisResult:
    return container.service.analyze_text(req.text, channel=req.channel)


@v1.post("/analyze/image", response_model=ImageAnalysisResult, tags=["analysis"])
async def analyze_image(
    container: ContainerDep,
    _: AuthDep,
    file: Annotated[UploadFile, File(description="JPEG/PNG/WEBP, maks 5 MB")],
    caption: Annotated[str | None, Form(max_length=2000)] = None,
) -> ImageAnalysisResult:
    limit = int(container.settings.max_upload_mb * 1024 * 1024)
    data = await file.read(limit + 1)
    try:
        return await run_in_threadpool(container.service.analyze_image, data, caption)
    except InvalidImageError as exc:
        raise HTTPException(422, str(exc)) from exc


@v1.post("/qris/parse", response_model=QrisAssessment, tags=["analysis"])
def parse_qris(req: QrisParseRequest, container: ContainerDep, _: AuthDep) -> QrisAssessment:
    return container.service.analyze_qr_payload(
        req.payload, req.claimed_merchant, req.claimed_amount, req.context_text
    )


@v1.post("/incidents", response_model=IncidentResult, status_code=201, tags=["golden-hour"])
def create_incident(req: IncidentCreate, container: ContainerDep, _: AuthDep) -> IncidentResult:
    try:
        result, _pdf = container.service.create_incident(req)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return result


@v1.get(
    "/reports/{report_id}.pdf",
    tags=["golden-hour"],
    responses={200: {"content": {"application/pdf": {}}}},
)
def get_report(report_id: str, container: ContainerDep, _: AuthDep) -> Response:
    if not report_id.isalnum() or len(report_id) != 32:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Laporan tidak ditemukan")
    rec = container.repo.get_report(report_id)
    if rec is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Laporan tidak ditemukan atau kedaluwarsa")
    return Response(
        rec.pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="laporan-cekdulu-{report_id[:8]}.pdf"',
            "Cache-Control": "no-store",
        },
    )


@v1.post(
    "/telegram/process",
    response_model=BotReply,
    tags=["integrations"],
    summary="Proses satu Telegram Update (dipanggil oleh n8n)",
)
def telegram_process(
    update: Annotated[dict[str, Any], Body()], container: ContainerDep, _: AuthDep
) -> BotReply:
    return container.telegram.handle(update)
