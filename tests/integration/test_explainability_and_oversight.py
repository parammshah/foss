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


def test_human_modification_revalidation_rejects_reserve_floor_breach():
    """
    CRITICAL PROMPT REQUIREMENT: A user must not be able to modify an action amount
    and have it executed without financial revalidation against live invariants.
    """
    service = DecisionExplanationService()
    decision = create_mock_decision_record()
    state = create_mock_treasury_state()

    # state available cash is $45M, protected reserve floor is $40M
    # An outflow action of $20M would leave $25M (< $40M floor)
    decision.selected_action = "INVEST_TREASURY_BILLS_3M"

    ok, msg, _, _, learn_ev = service.execute_human_workflow(
        decision_record=decision,
        action_type=ApprovalStatus.MODIFIED.value,
        actor_name="Senior Trader",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Trying to invest surplus into bills.",
        modified_amount=30_000_000.0,
        treasury_state=state,
    )

    assert not ok
    assert "breaches the Mandatory Reserve Floor" in msg
    assert learn_ev is not None
    assert learn_ev.versioned_event_type == "HUMAN_MODIFICATION_REVALIDATION_FAILED"


def test_cancellation_and_rejection_produce_zero_phantom_outcomes():
    """
    CRITICAL PROMPT REQUIREMENT: Cancellation must create an auditable cancelled record
    and must not be reported as a successful financial execution or invent phantom costs.
    """
    service = DecisionExplanationService()
    decision = create_mock_decision_record()

    # 1. Test Cancellation
    ok_c, msg_c, dec_c, out_c, ev_c = service.execute_human_workflow(
        decision_record=decision,
        action_type=ApprovalStatus.CANCELLED.value,
        actor_name="Head of Treasury",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Market conditions stabilized; cancelling proposed repo draw.",
    )

    assert ok_c
    assert dec_c.approval_status == ApprovalStatus.CANCELLED.value
    assert out_c.settlement_status == "CANCELLED"
    assert out_c.realized_amount == 0.0
    assert out_c.realized_costs["total"] == 0.0
    assert ev_c.payload["is_executed"] is False

    # 2. Test Rejection
    decision2 = create_mock_decision_record()
    ok_r, msg_r, dec_r, out_r, ev_r = service.execute_human_workflow(
        decision_record=decision2,
        action_type=ApprovalStatus.REJECTED.value,
        actor_name="Chief Risk Officer",
        actor_role=UserRole.RISK_OFFICER.value,
        stated_reason="Counterparty exposure too high.",
    )

    assert ok_r
    assert dec_r.approval_status == ApprovalStatus.REJECTED.value
    assert out_r.settlement_status == "REJECTED"
    assert out_r.realized_amount == 0.0
    assert out_r.realized_costs["total"] == 0.0


def test_truthful_explanation_missing_gates_not_marked_passed():
    """
    CRITICAL PROMPT REQUIREMENT: Never claim all mandatory gates passed if none were supplied.
    """
    service = DecisionExplanationService()
    decision = create_mock_decision_record()
    decision.hard_gate_outcomes = []  # No gates evaluated

    exp = service.explain(decision)
    assert not exp.all_hard_gates_passed
    assert any("NOT_EVALUATED" in g.gate_name for g in exp.hard_gates)
    assert any("No hard safety gate evaluations" in warning for warning in exp.disclosed_missing_information)

