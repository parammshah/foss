"""
Unit tests for canonical domain contracts, enum types, and safety defaults.
Verifies Section 4 non-negotiable rules:
- Unevaluated hard gate defaults to NOT_EVALUATED or UNKNOWN, never PASS.
- Unavailable Treasury Safety Score defaults to None, never 85.0.
- Missing protected reserve floor defaults to None, never 0.0.
"""

from __future__ import annotations
import pytest

from reserve_juggler.contracts import (
    ApprovalStatus,
    AssetPosition,
    DecisionRecord,
    GateStatus,
    HardGateOutcome,
    Obligation,
    QualityStatus,
    RiskSeverity,
    RiskSignal,
    ServiceStatus,
    TreasuryState,
    UserRole,
)


def test_hard_gate_outcome_default_status():
    gate = HardGateOutcome(gate_name="Future-Protected Capital Firewall")
    assert gate.status == GateStatus.NOT_EVALUATED
    assert gate.status != GateStatus.PASS
    assert not gate.passed


def test_treasury_state_reserve_floor_defaults_to_none():
    ts = TreasuryState(
        timestamp="2026-10-17T12:00:00+00:00",
        available_cash=100_000_000.0,
        asset_positions=[],
        obligations=[],
        authorized_credit_availability=50_000_000.0,
    )
    assert ts.protected_reserve_floor is None
    assert ts.protected_reserve_floor != 0.0


def test_decision_record_safety_score_defaults_to_none():
    dr = DecisionRecord(
        decision_id="DEC-UNIT-001",
        timestamp="2026-10-17T12:00:00+00:00",
        selected_action="INVEST_REPO",
        target_asset="REPO-01",
        amount=10_000_000.0,
    )
    assert dr.treasury_safety_score is None
    assert dr.treasury_safety_score != 85.0


def test_gate_status_values_distinct():
    assert GateStatus.PASS.value == "PASS"
    assert GateStatus.FAIL.value == "FAIL"
    assert GateStatus.UNKNOWN.value == "UNKNOWN"
    assert GateStatus.NOT_EVALUATED.value == "NOT_EVALUATED"
