"""Parser payload QRIS (EMVCo Merchant-Presented Mode) dan validasi CRC.

Struktur TLV: ``ID(2) LEN(2) VALUE(LEN)``. Tag penting:
00 payload format ("01"), 01 metode ("11" statis, "12" dinamis),
26-51 informasi akun merchant (sub 00 GUI, 01 MPAN, 02 MID/NMID, 03 kriteria),
52 MCC, 53 mata uang ("360" = IDR), 54 nominal, 55 indikator tip, 56/57 nilai tip,
58 negara ("ID"), 59 nama merchant, 60 kota, 61 kode pos, 62 data tambahan, 63 CRC.

CRC = CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, tanpa refleksi, xorout 0)
dihitung atas seluruh payload hingga dan termasuk "6304".
"""

from __future__ import annotations

from dataclasses import dataclass, field


class QrisParseError(ValueError):
    pass


def crc16_ccitt_false(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


def parse_tlv(payload: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    i = 0
    while i < len(payload):
        if i + 4 > len(payload):
            raise QrisParseError(f"TLV terpotong pada posisi {i}")
        tag, length_str = payload[i : i + 2], payload[i + 2 : i + 4]
        if not (tag.isdigit() and length_str.isdigit()):
            raise QrisParseError(f"Tag/panjang tidak valid pada posisi {i}: {tag!r}{length_str!r}")
        length = int(length_str)
        value = payload[i + 4 : i + 4 + length]
        if len(value) != length:
            raise QrisParseError(f"Nilai tag {tag} lebih pendek dari panjang yang dinyatakan")
        items.append((tag, value))
        i += 4 + length
    return items


@dataclass
class MerchantAccount:
    tag: str
    gui: str | None = None
    pan: str | None = None
    merchant_id: str | None = None
    criteria: str | None = None
    raw: dict[str, str] = field(default_factory=dict)


@dataclass
class QrisPayload:
    fields: dict[str, str]
    accounts: list[MerchantAccount]
    additional: dict[str, str]
    crc_declared: str | None
    crc_computed: str | None

    @property
    def crc_valid(self) -> bool:
        return self.crc_declared is not None and self.crc_declared == self.crc_computed

    @property
    def initiation(self) -> str | None:
        return {"11": "static", "12": "dynamic"}.get(self.fields.get("01", ""))

    @property
    def nmid(self) -> str | None:
        for acc in self.accounts:
            if acc.tag == "51" and acc.merchant_id:
                return acc.merchant_id
        return None


def parse_qris(payload: str) -> QrisPayload:
    payload = payload.strip()
    if not payload.startswith("000201"):
        raise QrisParseError("Bukan payload EMVCo (harus diawali 000201)")
    items = parse_tlv(payload)
    fields: dict[str, str] = {}
    accounts: list[MerchantAccount] = []
    additional: dict[str, str] = {}
    for tag, value in items:
        if 26 <= int(tag) <= 51:
            sub = dict(parse_tlv(value))
            accounts.append(
                MerchantAccount(
                    tag=tag,
                    gui=sub.get("00"),
                    pan=sub.get("01"),
                    merchant_id=sub.get("02"),
                    criteria=sub.get("03"),
                    raw=sub,
                )
            )
        elif tag == "62":
            additional = dict(parse_tlv(value))
        else:
            fields[tag] = value

    crc_declared = fields.get("63")
    crc_computed = None
    idx = payload.rfind("6304")
    if crc_declared is not None and idx != -1 and idx + 8 == len(payload):
        crc_computed = f"{crc16_ccitt_false(payload[: idx + 4].encode('utf-8')):04X}"
        crc_declared = crc_declared.upper()
    return QrisPayload(fields, accounts, additional, crc_declared, crc_computed)


def build_qris(fields: list[tuple[str, str]]) -> str:
    """Bangun payload ber-CRC (untuk test/fixture). ``fields`` tanpa tag 63."""
    body = "".join(f"{tag}{len(value):02d}{value}" for tag, value in fields) + "6304"
    return body + f"{crc16_ccitt_false(body.encode('utf-8')):04X}"


def tlv(tag: str, value: str) -> str:
    return f"{tag}{len(value):02d}{value}"
