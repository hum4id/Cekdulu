from cekdulu.ml.preprocessing import normalize_text
from cekdulu.security.pii import hmac_digest, mask_number, redact_text


def test_normalize_replaces_entities_with_placeholders():
    out = normalize_text(
        "BURUAN!!! Klik bit.ly/abc, hub 0812-3456-7890, transfer Rp 2.500.000 kode 123"
    )
    assert "__url__" in out
    assert "__phone__" in out
    assert "__money__" in out
    assert "__num__" in out
    assert "0812" not in out and "bit.ly" not in out
    assert out == out.lower()


def test_normalize_collapses_elongation_and_is_idempotent():
    once = normalize_text("Selamaaattt   kamuuu menang")
    assert once == "selamaatt kamuu menang"
    assert normalize_text(once) == once


def test_mask_number_keeps_last_four():
    assert mask_number("1234 5678 90") == "******7890"
    assert mask_number("") == ""


def test_hmac_is_deterministic_and_secret_dependent():
    assert hmac_digest(" ABC ", "s1") == hmac_digest("abc", "s1")
    assert hmac_digest("abc", "s1") != hmac_digest("abc", "s2")


def test_redact_text_masks_phone_and_long_digits():
    out = redact_text("hubungi 081234567890 rek 1234567890123")
    assert "081234567890" not in out
    assert "1234567890123" not in out
    assert out.endswith("0123")
