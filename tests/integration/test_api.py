from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cekdulu.api.app import create_app
from cekdulu.config import Settings
from tests.conftest import API_KEY, make_qr_png, requires_model
from tests.unit.test_qris import sample_qris

pytestmark = requires_model
H = {"X-API-Key": API_KEY}


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as c:
        yield c


def test_health_and_ready(client):
    assert client.get("/healthz").json()["status"] == "ok"
    ready = client.get("/readyz").json()
    assert ready["status"] == "ready" and ready["model_version"].startswith("baseline")


def test_auth_required(client):
    r = client.post("/v1/analyze/text", json={"text": "halo"})
    assert r.status_code == 401
    r = client.post("/v1/analyze/text", json={"text": "halo"}, headers={"X-API-Key": "salah"})
    assert r.status_code == 401


def test_analyze_fraud_text(client):
    text = (
        "Selamat! Nomor anda terpilih pemenang undian hadiah mobil. Transfer biaya admin "
        "pajak hadiah Rp2.500.000 ke rek 1234567890 segera"
    )
    r = client.post("/v1/analyze/text", json={"text": text}, headers=H)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["risk_level"] in {"HIGH", "CRITICAL"}
    assert body["label"] == "penipuan"
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
    assert body["entities"]["account_numbers"] == ["******7890"]
    assert body["explanation_source"] == "template"
    assert r.headers["X-Request-ID"]


def test_analyze_normal_text_low(client):
    r = client.post(
        "/v1/analyze/text",
        headers=H,
        json={"text": "Nanti jam 7 jadi kumpul di rumah Dina kan? Aku bawa martabak"},
    )
    assert r.json()["risk_level"] in {"LOW", "MEDIUM"}


def test_text_validation(client):
    assert client.post("/v1/analyze/text", json={"text": ""}, headers=H).status_code == 422
    too_long = "a" * 5001
    assert client.post("/v1/analyze/text", json={"text": too_long}, headers=H).status_code == 422


def test_image_with_qris_and_receive_caption(client):
    files = {"file": ("qr.png", make_qr_png(sample_qris()), "image/png")}
    r = client.post(
        "/v1/analyze/image",
        files=files,
        headers=H,
        data={"caption": "scan QRIS ini untuk menerima refund 450rb"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["qr_codes"][0]["raw_type"] == "qris"
    assert body["overall_risk_level"] == "CRITICAL"


def test_image_rejects_invalid_file(client):
    files = {"file": ("x.png", b"not an image", "image/png")}
    assert client.post("/v1/analyze/image", files=files, headers=H).status_code == 422


def test_qris_parse_endpoint(client):
    r = client.post(
        "/v1/qris/parse",
        headers=H,
        json={"payload": sample_qris(), "claimed_merchant": "Warung Bu Sri"},
    )
    assert r.status_code == 200
    assert r.json()["risk_level"] == "LOW"


def test_incident_and_pdf_download(client):
    r = client.post(
        "/v1/incidents",
        headers=H,
        json={
            "amount_idr": 2500000,
            "source_bank": "BRI",
            "destination_bank": "BCA",
            "destination_account": "1234567890",
            "destination_name": "Budi",
            "chronology": "Ditawari kerja paruh waktu lalu diminta deposit.",
        },
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["minutes_since_transfer"] <= 1
    assert body["urgency"].startswith("KRITIS")
    assert body["steps"][0]["order"] == 1
    pdf = client.get(f"/v1/reports/{body['report_id']}.pdf", headers=H)
    assert pdf.status_code == 200
    assert pdf.content.startswith(b"%PDF")
    assert client.get("/v1/reports/" + "0" * 32 + ".pdf", headers=H).status_code == 404
    assert client.get("/v1/reports/../../etc.pdf", headers=H).status_code == 404


def test_incident_rejects_future_time(client):
    r = client.post("/v1/incidents", headers=H, json={"transfer_time": "2999-01-01T00:00:00+07:00"})
    assert r.status_code == 422


def test_telegram_process_text_and_report(client):
    upd = {"update_id": 1, "message": {"chat": {"id": 7}, "text": "/start"}}
    assert (
        "CekDulu" in client.post("/v1/telegram/process", json=upd, headers=H).json()["reply_text"]
    )
    upd["message"]["text"] = "Kirimkan kode OTP yang masuk ke HP kamu ke saya sekarang ya"
    reply = client.post("/v1/telegram/process", json=upd, headers=H).json()
    assert "Sangat berbahaya" in reply["reply_text"]
    upd["message"]["text"] = "/lapor\nnominal: 1jt\nbank tujuan: BCA\nrekening: 9876543210"
    reply = client.post("/v1/telegram/process", json=upd, headers=H).json()
    assert reply["report_id"] and reply["schedule_reminder"] is True


def test_rate_limit(settings: Settings):
    settings.rate_limit_per_minute = 2
    with TestClient(create_app(settings)) as c:
        codes = [
            c.post("/v1/qris/parse", headers=H, json={"payload": "abc"}).status_code
            for _ in range(3)
        ]
    assert codes == [200, 200, 429]


def test_metrics_exposed(client):
    client.post("/v1/analyze/text", json={"text": "halo apa kabar"}, headers=H)
    assert "cekdulu_analyses_total" in client.get("/metrics").text


def test_production_settings_require_secrets(tmp_path):
    with pytest.raises(ValueError):
        Settings(env="production", api_keys="", database_url=f"sqlite:///{tmp_path}/x.db")
