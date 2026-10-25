from datetime import UTC, datetime

import pytest

from cekdulu.domain.schemas import RiskLevel, Severity, TacticHit
from cekdulu.ml.risk import RiskPolicy
from cekdulu.services.telegram import parse_report_form


def hit(code: str, sev: Severity) -> TacticHit:
    return TacticHit(code=code, label=code, severity=sev, evidence="x")


policy = RiskPolicy(fraud_threshold=0.4, low_cutoff=0.2)


def test_low_probability_no_tactics_is_low():
    assert policy.decide(0.05, []).level == RiskLevel.LOW


def test_threshold_maps_to_high():
    assert policy.decide(0.45, []).level == RiskLevel.HIGH


def test_critical_tactic_escalates_regardless_of_model():
    d = policy.decide(0.01, [hit("APP_INSTALL", Severity.CRITICAL)])
    assert d.level == RiskLevel.CRITICAL and d.escalated_by == ["APP_INSTALL"]


def test_two_high_tactics_escalate_to_high():
    d = policy.decide(0.1, [hit("A", Severity.HIGH), hit("B", Severity.HIGH)])
    assert d.level == RiskLevel.HIGH


def test_single_high_tactic_is_at_least_medium():
    assert policy.decide(0.01, [hit("A", Severity.HIGH)]).level == RiskLevel.MEDIUM


def test_rules_never_lower_model_score():
    assert policy.decide(0.97, []).score == pytest.approx(0.97)


def test_invalid_policy_config():
    with pytest.raises(ValueError):
        RiskPolicy(fraud_threshold=0.2, low_cutoff=0.3)


def test_parse_report_form_hhmm_is_wib():
    now = datetime(2026, 10, 23, 8, 0, tzinfo=UTC)  # 15:00 WIB
    form = (
        "/lapor\nwaktu: 14:05\nnominal: 2.500.000\nbank asal: BRI\nbank tujuan: BCA\n"
        "rekening: 1234567890\nkronologi: ditawari kerja\nlalu diminta deposit"
    )
    data = parse_report_form(form, now=now)
    assert data.transfer_time == datetime(2026, 10, 23, 7, 5, tzinfo=UTC)
    assert data.amount_idr == 2_500_000
    assert data.source_bank == "BRI" and data.destination_bank == "BCA"
    assert data.chronology == "ditawari kerja\nlalu diminta deposit"


def test_parse_report_form_future_time_rolls_to_yesterday():
    now = datetime(2026, 10, 23, 1, 0, tzinfo=UTC)  # 08:00 WIB
    data = parse_report_form("/lapor waktu: 23:30", now=now)
    assert data.transfer_time == datetime(2026, 10, 22, 16, 30, tzinfo=UTC)
