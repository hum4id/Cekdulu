import pytest

from cekdulu.domain.schemas import Severity
from cekdulu.ml.tactics import assess_url, detect_tactics, extract_entities, parse_amount_idr


def codes(text: str) -> set[str]:
    return {t.code for t in detect_tactics(text)}


def test_otp_request_is_critical():
    hits = detect_tactics("Mohon sebutkan kode OTP yang barusan masuk ke HP kakak ya")
    assert any(h.code == "CREDENTIAL_REQUEST" and h.severity == Severity.CRITICAL for h in hits)


def test_otp_warning_from_bank_is_not_flagged():
    assert "CREDENTIAL_REQUEST" not in codes(
        "Kode OTP Anda 482913. JANGAN berikan kode ini kepada siapa pun termasuk petugas bank."
    )


@pytest.mark.parametrize(
    "text",
    [
        "Lihat undangan di file terlampir: Undangan_Pernikahan.apk",
        "Silakan download aplikasi berikut untuk cek foto paket",
    ],
)
def test_app_install_detected(text):
    assert "APP_INSTALL" in codes(text)


def test_wrong_transfer_and_advance_fee():
    found = codes("Saya salah transfer, tolong kembalikan. Biaya admin 50rb transfer dulu ya")
    assert {"WRONG_TRANSFER", "ADVANCE_FEE"} <= found


def test_qr_receive_funds_is_critical():
    assert "QR_RECEIVE_FUNDS" in codes("Silakan scan QRIS ini untuk menerima pengembalian dana")


def test_plain_chat_has_no_tactics():
    assert codes("Nanti jam 7 jadi kumpul di rumah Dina kan? Aku bawa martabak") == set()


@pytest.mark.parametrize(
    "url,expect",
    [
        ("bit.ly/robot-cuan", "pemendek"),
        ("http://192.168.10.5/login", "alamat IP"),
        ("https://bri-tarif-baru.xyz/form", "mencatut"),
        ("https://evil.example.com/app.apk", "APK"),
    ],
)
def test_assess_url_reasons(url, expect):
    assert any(expect in r for r in assess_url(url))


def test_official_looking_plain_domain_has_no_reason():
    assert assess_url("https://www.ojk.go.id") == []


@pytest.mark.parametrize(
    "raw,value",
    [
        ("Rp 2.500.000", 2_500_000),
        ("300rb", 300_000),
        ("1,5 juta", 1_500_000),
        ("Rp35 Ribu", 35_000),
        ("Rp150.000", 150_000),
        ("2jt", 2_000_000),
    ],
)
def test_parse_amount(raw, value):
    assert parse_amount_idr(raw) == value


def test_entities_are_masked():
    ents = extract_entities("Transfer Rp1.500.000 ke no rek 1234567890 a.n Siti, hub 081200001234")
    assert ents.account_numbers == ["******7890"]
    assert ents.phone_numbers and "081200001234" not in ents.phone_numbers[0]
    assert 1_500_000 in ents.amounts_idr
