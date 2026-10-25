import pytest

from cekdulu.domain.schemas import RiskLevel
from cekdulu.qris.assess import assess_qr_payload
from cekdulu.qris.parser import QrisParseError, build_qris, crc16_ccitt_false, parse_qris, tlv


def sample_qris(name: str = "WARUNG BU SRI", amount: str | None = None) -> str:
    merchant = tlv("00", "ID.CO.QRIS.WWW") + tlv("02", "ID1023123456789") + tlv("03", "UMI")
    fields = [
        ("00", "01"),
        ("01", "12" if amount else "11"),
        ("51", merchant),
        ("52", "5812"),
        ("53", "360"),
    ]
    if amount:
        fields.append(("54", amount))
    fields += [("58", "ID"), ("59", name), ("60", "PURWOKERTO"), ("61", "53147")]
    return build_qris(fields)


def test_crc16_ccitt_false_check_value():
    # Nilai cek standar CRC-16/CCITT-FALSE untuk "123456789"
    assert crc16_ccitt_false(b"123456789") == 0x29B1


def test_parse_valid_qris():
    q = parse_qris(sample_qris())
    assert q.crc_valid
    assert q.initiation == "static"
    assert q.nmid == "ID1023123456789"
    assert q.fields["59"] == "WARUNG BU SRI"


def test_tampered_payload_fails_crc():
    payload = sample_qris().replace("WARUNG BU SRI", "WARUNG BU SRO")
    result = assess_qr_payload(payload)
    assert result.info and result.info.crc_valid is False
    assert result.risk_level == RiskLevel.CRITICAL


def test_valid_qris_is_low_risk():
    result = assess_qr_payload(sample_qris(), claimed_merchant="Warung Bu Sri")
    assert result.raw_type == "qris"
    assert result.risk_level == RiskLevel.LOW


def test_merchant_mismatch_flags_quishing():
    result = assess_qr_payload(
        sample_qris(name="TOKO BUDI JAYA"), claimed_merchant="Parkir Mall Kota"
    )
    assert any(f.code == "MERCHANT_MISMATCH" for f in result.findings)
    assert result.risk_level == RiskLevel.HIGH


def test_amount_mismatch():
    result = assess_qr_payload(sample_qris(amount="150000"), claimed_amount=15000)
    assert any(f.code == "AMOUNT_MISMATCH" for f in result.findings)


def test_receive_context_is_critical_even_for_valid_qris():
    result = assess_qr_payload(sample_qris(), context_text="scan QR ini untuk menerima refund")
    assert result.risk_level == RiskLevel.CRITICAL


def test_terima_kasih_does_not_trigger_receive_rule():
    result = assess_qr_payload(sample_qris(), context_text="Terima kasih sudah belanja")
    assert result.risk_level == RiskLevel.LOW


def test_url_qr():
    result = assess_qr_payload("https://bit.ly/klaim-hadiah")
    assert result.raw_type == "url"
    assert result.risk_level == RiskLevel.HIGH


def test_malformed_tlv_raises():
    with pytest.raises(QrisParseError):
        parse_qris("000201010211269")
