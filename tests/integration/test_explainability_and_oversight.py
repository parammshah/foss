"""
Integration tests for Decision Explanation Service and Human Oversight Workflows.
Verifies grounded calculation explanations, role permissions, and simulated execution.
"""

import pytest
from reserve_juggler.explainability.service import DecisionExplanationService
from reserve_juggler.integration.contracts import ApprovalStatus, UserRole
from reserve_juggler.integration.fixtures import (
    create_mock_decision_record,
    create_mock_forecast_bundle,
    create_mock_treasury_state,
)


def test_decision_explanation_grounded_completeness():
    """Verifies all mandatory explanation attributes are present and grounded in numbers."""
    service = DecisionExplanationService()
    decision = create_mock_decision_record()
    forecast = create_mock_forecast_bundle()
    state = create_mock_treasury_state()

    exp = service.explain(decision, forecast, state)

    assert exp.decision_id == decision.decision_id
    assert exp.recommended_action == decision.selected_action
    assert exp.recommended_amount == decision.amount
    assert exp.protected_reserve_floor == decision.reserve
    assert exp.eligible_surplus == decision.eligible_surplus
    assert exp.treasury_safety_score == decision.treasury_safety_score
    assert exp.all_hard_gates_passed
    assert len(exp.hard_gates) >= 3
    assert len(exp.alternatives_comparison) >= 2
    assert exp.cost_breakdown.total_cost > 0
    # Rationale must contain specific numbers
    assert str(int(decision.amount))[:4] in exp.grounded_narrative.replace(",", "")


def test_human_workflow_role_permissions():
    """Verifies role-based authorization constraints on approvals, rejections, and overrides."""
    service = DecisionExplanationService()
    decision = create_mock_decision_record()

    # 1. Auditor attempts approval -> Must be rejected (Auditor is read-only)
    ok_aud, msg_aud, _, _, _ = service.execute_human_workflow(
        decision_record=decision,
        action_type=ApprovalStatus.APPROVED.value,
        actor_name="Internal Auditor",
        actor_role=UserRole.AUDITOR.value,
        stated_reason="Looks okay.",
    )
    assert not ok_aud
    assert "read-only" in msg_aud

    # 2. Treasury Analyst attempts rejection/override -> Must be denied
    ok_an, msg_an, _, _, _ = service.execute_human_workflow(
        decision_record=decision,
        action_type=ApprovalStatus.OVERRIDDEN.value,
        actor_name="Junior Analyst",
        actor_role=UserRole.TREASURY_ANALYST.value,
        stated_reason="Overriding recommendation.",
    )
    assert not ok_an
    assert "cannot execute override" in msg_an

    # 3. Head of Treasury approves recommendation -> Success
    ok_head, msg_head, updated_dec, outcome, learn_ev = service.execute_human_workflow(
        decision_record=decision,
        action_type=ApprovalStatus.APPROVED.value,
        actor_name="Senior Treasurer",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Approved for intraday settlement matching.",
    )
    assert ok_head
    assert updated_dec.approval_status == ApprovalStatus.APPROVED.value
    assert outcome.simulated
    assert outcome.settlement_status == "SETTLED"
    assert learn_ev.module == "HUMAN_OVERSIGHT"


def test_human_workflow_modification():
    """Verifies that an authorized officer can modify the proposed amount before execution."""
    service = DecisionExplanationService()
    decision = create_mock_decision_record()

    ok_mod, msg_mod, updated_dec, outcome, _ = service.execute_human_workflow(
        decision_record=decision,
        action_type=ApprovalStatus.MODIFIED.value,
        actor_name="Chief Risk Officer",
        actor_role=UserRole.RISK_OFFICER.value,
        stated_reason="Scaled down draw amount due to lower midday projected outflow.",
        modified_amount=10_000_000.0,
    )
    assert ok_mod
    assert updated_dec.approval_status == ApprovalStatus.MODIFIED.value
    assert updated_dec.amount == 10_000_000.0
    assert outcome.realized_amount == 10_000_000.0
