"""Comprehensive regression tests for approval-state machine, freshness TTL, and ledger reconciliation."""

from datetime import datetime, timedelta, timezone
import pytest

from reserve_juggler.contracts import (
    ActionType,
    ApprovalStatus,
    AssetPosition,
    AssetTier,
    CreditFacility,
    DecisionRecord,
    GateStatus,
    ServiceStatus,
    TreasuryState,
    UserRole,
    ensure_utc,
)
from reserve_juggler.decision.models import StandardDecisionRecord
from reserve_juggler.explainability.service import DecisionExplanationService
from reserve_juggler.integration.orchestrator import (
    DecisionOrchestrator,
    IndependentConstraintValidator,
    OrchestratedDecisionResult,
)
from reserve_juggler.tax.tax_profile import (
    create_statutory_us_corp_profile,
    create_synthetic_demo_profile,
)


def _build_test_treasury_state(
    cash: float = 65_000_000.0,
    obligations: float = 12_500_000.0,
    repo_value: float = 20_000_000.0,
    now: datetime = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
) -> TreasuryState:
    t1_repo = AssetPosition(
        asset_id="T1_REPO_PORTFOLIO",
        asset_class="REPO",
        tier=AssetTier.TIER_1_REPO,
        market_value_usd=repo_value,
        unrealized_gain_usd=0.0,
        holding_period_days=1,
        is_encumbered=False,
        annual_yield_or_return=0.0515,
    )
    loc = CreditFacility(
        facility_id="EMERGENCY_LOC_01",
        committed_limit_usd=50_000_000.0,
        utilized_amount_usd=0.0,
        apr=0.075,
        fee_rate=0.0025,
        is_available=True,
    )
    return TreasuryState(
        timestamp=now,
        operational_cash_usd=cash,
        asset_positions={"T1_REPO_PORTFOLIO": t1_repo},
        upcoming_obligations_usd=obligations,
        credit_facility=loc,
        available_credit_usd=50_000_000.0,
        protected_reserve_floor=25_000_000.0,
        as_of_timestamp=now,
    )


def test_statutory_decision_requiring_approval_under_policy():
    """1. A statutory, valid-tax decision requiring approval under configured policy."""
    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    state = _build_test_treasury_state(cash=65_000_000.0, obligations=12_500_000.0, now=now)
    stat_profile = create_statutory_us_corp_profile(entity_id="US_CORP_01", tax_year=2026)

    orch = DecisionOrchestrator()
    # When auto_simulate_execution=False, policy requires officer approval before commitment
    res = orch.run_cycle(
        current_state=state,
        as_of_timestamp=now,
        tax_profile=stat_profile,
        auto_simulate_execution=False,
    )

    assert res.validation_result.passed is True
    assert res.approval_status == ApprovalStatus.PENDING_APPROVAL.value
    assert res.workflow_state == "EXECUTION_ELIGIBLE"
    assert res.simulated_outcome is None
    assert res.decision_record.approval_status == ApprovalStatus.PENDING_APPROVAL.value
    assert res.decision_record.metadata["tax_profile_status"] == "STATUTORY_APPROVED"


def test_synthetic_or_missing_tax_mandatory_human_approval():
    """2. A synthetic or missing tax profile requiring mandatory human approval."""
    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    state = _build_test_treasury_state(cash=65_000_000.0, obligations=12_500_000.0, now=now)

    orch = DecisionOrchestrator()
    # Missing tax profile
    res_none = orch.run_cycle(
        current_state=state,
        as_of_timestamp=now,
        tax_profile=None,
        auto_simulate_execution=True,
    )
    assert res_none.validation_result.requires_human_approval is True
    assert res_none.approval_status == ApprovalStatus.PENDING_APPROVAL.value
    assert res_none.workflow_state == "HUMAN_APPROVAL_PENDING"
    assert res_none.simulated_outcome is None

    # Synthetic demo profile
    synth_profile = create_synthetic_demo_profile()
    res_synth = orch.run_cycle(
        current_state=state,
        as_of_timestamp=now,
        tax_profile=synth_profile,
        auto_simulate_execution=True,
    )
    assert res_synth.validation_result.requires_human_approval is True
    assert res_synth.approval_status == ApprovalStatus.PENDING_APPROVAL.value
    assert res_synth.workflow_state == "HUMAN_APPROVAL_PENDING"
    assert res_synth.simulated_outcome is None


def test_unauthorized_actors_rejected():
    """3. An unauthorized actor cannot approve, override, or exceed limits."""
    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    state = _build_test_treasury_state(now=now)
    stat_profile = create_statutory_us_corp_profile(entity_id="US_CORP_01", tax_year=2026)

    orch = DecisionOrchestrator()
    res = orch.run_cycle(
        current_state=state,
        as_of_timestamp=now,
        tax_profile=stat_profile,
        auto_simulate_execution=False,
    )
    dec = res.decision_record

    expl_service = DecisionExplanationService()

    # Case A: Auditor role is strictly read-only
    ok_aud, msg_aud, _, out_aud, _ = expl_service.execute_human_workflow(
        decision_record=dec,
        action_type="APPROVED",
        actor_name="External Auditor",
        actor_role=UserRole.AUDITOR.value,
        stated_reason="Auditor signoff attempt",
        treasury_state=state,
        as_of_time=now + timedelta(minutes=5),
    )
    assert ok_aud is False
    assert "read-only" in msg_aud
    assert out_aud is None

    # Case B: Treasury Analyst cannot override or cancel
    ok_an_ov, msg_an_ov, _, _, _ = expl_service.execute_human_workflow(
        decision_record=dec,
        action_type="OVERRIDDEN",
        actor_name="Junior Analyst",
        actor_role=UserRole.TREASURY_ANALYST.value,
        stated_reason="Attempting override",
        treasury_state=state,
        as_of_time=now + timedelta(minutes=5),
    )
    assert ok_an_ov is False
    assert "TREASURY_ANALYST" in msg_an_ov

    # Case C: Treasury Analyst cannot approve amounts > $10M
    ok_an_hi, msg_an_hi, _, _, _ = expl_service.execute_human_workflow(
        decision_record=dec,
        action_type="MODIFIED",
        actor_name="Junior Analyst",
        actor_role=UserRole.TREASURY_ANALYST.value,
        stated_reason="Increasing to 15M",
        modified_amount=15_000_000.0,
        treasury_state=state,
        as_of_time=now + timedelta(minutes=5),
    )
    assert ok_an_hi is False
    assert "exceeding $10,000,000" in msg_an_hi


def test_denied_or_absent_approval():
    """4. A denied or absent approval produces zero financial transactions."""
    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    state = _build_test_treasury_state(now=now)
    stat_profile = create_statutory_us_corp_profile(entity_id="US_CORP_01", tax_year=2026)

    orch = DecisionOrchestrator()
    res = orch.run_cycle(
        current_state=state,
        as_of_timestamp=now,
        tax_profile=stat_profile,
        auto_simulate_execution=False,
    )
    dec = res.decision_record

    expl_service = DecisionExplanationService()

    # Case A: Explicit rejection by Risk Officer
    ok_rej, msg_rej, updated_dec, outcome_rej, _ = expl_service.execute_human_workflow(
        decision_record=dec,
        action_type="REJECTED",
        actor_name="Chief Risk Officer",
        actor_role=UserRole.RISK_OFFICER.value,
        stated_reason="Market conditions volatile; pausing allocations.",
        treasury_state=state,
        as_of_time=now + timedelta(minutes=10),
    )
    assert ok_rej is True
    assert updated_dec.approval_status == ApprovalStatus.REJECTED.value
    assert outcome_rej is not None
    assert outcome_rej.realized_amount == 0.0
    assert outcome_rej.settlement_status == "REJECTED"


def test_failed_safety_gate_cannot_be_approved():
    """5. A decision that failed safety gates cannot be approved for execution."""
    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    # Cash ($5M) < obligations ($15M) -> immediate deficit
    broken_state = _build_test_treasury_state(cash=5_000_000.0, obligations=15_000_000.0, now=now)
    stat_profile = create_statutory_us_corp_profile(entity_id="US_CORP_01", tax_year=2026)

    orch = DecisionOrchestrator()
    res = orch.run_cycle(
        current_state=broken_state,
        as_of_timestamp=now,
        tax_profile=stat_profile,
        auto_simulate_execution=False,
    )

    assert res.validation_result.passed is False
    assert res.approval_status == ApprovalStatus.REJECTED.value
    assert res.workflow_state == "SAFETY_VALIDATION_FAILED"

    expl_service = DecisionExplanationService()
    ok, msg, _, outcome, _ = expl_service.execute_human_workflow(
        decision_record=res.decision_record,
        action_type="APPROVED",
        actor_name="Head of Treasury",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Force approval attempt",
        treasury_state=broken_state,
        as_of_time=now + timedelta(minutes=5),
    )
    assert ok is False
    assert "failed mandatory independent safety validation" in msg
    assert outcome is None


def test_approval_parameter_tampering_rejected():
    """6. An approval with altered action, target, or amount without revalidation is rejected."""
    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    state = _build_test_treasury_state(cash=65_000_000.0, obligations=12_500_000.0, now=now)
    stat_profile = create_statutory_us_corp_profile(entity_id="US_CORP_01", tax_year=2026)

    orch = DecisionOrchestrator()
    res = orch.run_cycle(
        current_state=state,
        as_of_timestamp=now,
        tax_profile=stat_profile,
        auto_simulate_execution=False,
    )
    dec = res.decision_record
    orig_amt = dec.amount

    expl_service = DecisionExplanationService()

    # Tampering: Attempting standard "APPROVED" while passing modified_amount silently
    ok_tamper, msg_tamper, _, _, _ = expl_service.execute_human_workflow(
        decision_record=dec,
        action_type="APPROVED",
        actor_name="Head of Treasury",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Normal approval with silent inflation",
        modified_amount=orig_amt + 5_000_000.0,
        treasury_state=state,
        as_of_time=now + timedelta(minutes=5),
    )
    assert ok_tamper is False
    assert "Parameter tampering detected" in msg_tamper


def test_duplicate_approval_or_execution_idempotency():
    """7. Duplicate execution request cannot create double side effects."""
    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    state = _build_test_treasury_state(cash=65_000_000.0, obligations=12_500_000.0, now=now)
    state.protected_reserve_floor = 25_000_000.0
    stat_profile = create_statutory_us_corp_profile(entity_id="US_CORP_01", tax_year=2026)

    orch = DecisionOrchestrator()
    res = orch.run_cycle(
        current_state=state,
        as_of_timestamp=now,
        tax_profile=stat_profile,
        auto_simulate_execution=False,
    )
    dec = res.decision_record

    expl_service = DecisionExplanationService()

    # Execution 1: First valid approval
    ok1, msg1, dec_app, out1, _ = expl_service.execute_human_workflow(
        decision_record=dec,
        action_type="APPROVED",
        actor_name="Head of Treasury",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="First certified approval",
        treasury_state=state,
        as_of_time=now + timedelta(minutes=5),
    )
    assert ok1 is True
    assert dec_app.approval_status == ApprovalStatus.APPROVED.value
    assert dec_app.execution_timestamp is not None

    # Execution 2: Duplicate request on the same decision record
    ok2, msg2, _, out2, _ = expl_service.execute_human_workflow(
        decision_record=dec_app,
        action_type="APPROVED",
        actor_name="Head of Treasury",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Duplicate submission",
        treasury_state=state,
        as_of_time=now + timedelta(minutes=10),
    )
    assert ok2 is False
    assert "Duplicate execution request rejected" in msg2
    assert out2 is None


def test_approval_freshness_ttl_rejection():
    """Demonstrate expired approval rejection using deterministic timestamps."""
    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    state = _build_test_treasury_state(cash=65_000_000.0, obligations=12_500_000.0, now=now)
    stat_profile = create_statutory_us_corp_profile(entity_id="US_CORP_01", tax_year=2026)

    orch = DecisionOrchestrator()
    res = orch.run_cycle(
        current_state=state,
        as_of_timestamp=now,
        tax_profile=stat_profile,
        auto_simulate_execution=False,
    )
    dec = res.decision_record

    expl_service = DecisionExplanationService()

    # Attempt approval 2 hours later (TTL is 3600s = 1h)
    stale_time = now + timedelta(hours=2)
    ok_stale, msg_stale, dec_stale, _, ev_stale = expl_service.execute_human_workflow(
        decision_record=dec,
        action_type="APPROVED",
        actor_name="Head of Treasury",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Late approval",
        treasury_state=state,
        as_of_time=stale_time,
        validity_window_seconds=3600.0,
    )
    assert ok_stale is False
    assert "Approval EXPIRED" in msg_stale
    assert dec_stale.approval_status == ApprovalStatus.EXPIRED.value
    assert ev_stale is not None
    assert ev_stale.versioned_event_type == "HUMAN_APPROVAL_EXPIRED"


def test_changed_treasury_state_rejection():
    """Demonstrate that changes in treasury state invalidate pending approvals."""
    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    initial_state = _build_test_treasury_state(cash=65_000_000.0, obligations=12_500_000.0, now=now)
    stat_profile = create_statutory_us_corp_profile(entity_id="US_CORP_01", tax_year=2026)

    orch = DecisionOrchestrator()
    res = orch.run_cycle(
        current_state=initial_state,
        as_of_timestamp=now,
        tax_profile=stat_profile,
        auto_simulate_execution=False,
    )
    dec = res.decision_record
    assert dec.amount > 0

    # Before approval, cash drains drastically due to unexpected customer outflows
    degraded_state = _build_test_treasury_state(cash=20_000_000.0, obligations=12_500_000.0, now=now)
    degraded_state.protected_reserve_floor = 25_000_000.0  # Reserve floor is $25M

    expl_service = DecisionExplanationService()
    ok, msg, _, _, _ = expl_service.execute_human_workflow(
        decision_record=dec,
        action_type="APPROVED",
        actor_name="Head of Treasury",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Approval under degraded state",
        treasury_state=degraded_state,
        as_of_time=now + timedelta(minutes=15),
    )
    assert ok is False
    assert "Mandatory Reserve Floor" in msg


def test_canonical_place_repo_ledger_reconciliation():
    """Verify complete PLACE_REPO workflow: signed cash flow, asset position, reserve preservation."""
    now = datetime(2026, 1, 11, 17, 0, 0, tzinfo=timezone.utc)
    state = _build_test_treasury_state(cash=65_000_000.0, obligations=12_500_000.0, now=now)
    stat_profile = create_statutory_us_corp_profile(entity_id="US_CORP_01", tax_year=2026)

    orch = DecisionOrchestrator()
    res = orch.run_cycle(
        current_state=state,
        as_of_timestamp=now,
        tax_profile=stat_profile,
        auto_simulate_execution=False,
    )
    dec = res.decision_record
    assert any(k in str(dec.selected_action).upper() for k in ["REPO", "INVEST", "BUY", "SWEEP"])

    initial_cash = state.operational_cash_usd
    initial_investments = state.total_investments_market_value_usd
    action_amt = dec.amount
    reserve_floor = 25_000_000.0
    state.protected_reserve_floor = reserve_floor

    # Officer approves the exact recommendation
    expl_service = DecisionExplanationService()
    ok, msg, approved_dec, outcome, learn_ev = expl_service.execute_human_workflow(
        decision_record=dec,
        action_type="APPROVED",
        actor_name="Director of Treasury Operations",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Approved overnight repo investment under protected reserve floor.",
        treasury_state=state,
        as_of_time=now + timedelta(minutes=10),
    )
    assert ok is True
    assert approved_dec.approval_status == ApprovalStatus.APPROVED.value

    # Execute approved action on ledger
    if isinstance(approved_dec.selected_action, dict):
        action_name = str(approved_dec.selected_action.get("action_type", "")).upper()
    else:
        action_name = str(getattr(approved_dec, "action_type", approved_dec.selected_action)).upper()
    action_spec = {
        "type": "SWEEP_REPO" if "REPO" in action_name else action_name,
        "amount_usd": approved_dec.amount,
        "asset_id": approved_dec.target_asset,
        "decision_id": approved_dec.decision_id,
    }
    new_state, sim_out = orch.ledger.simulate_action(state, action_spec, res.forecast_bundle)

    # 1. Exact signed cash flow applied
    signed_flow = sum(sim_out.realized_cash_movements.values())
    expected_cash = initial_cash + signed_flow
    assert new_state.operational_cash_usd == pytest.approx(expected_cash, abs=1e-2)
    assert signed_flow <= -action_amt  # At least the principal amount was deducted

    # 2. Asset position increment
    assert new_state.total_investments_market_value_usd >= initial_investments + action_amt - 1e-2

    # 3. Post-execution cash remains at or above reserve floor
    assert new_state.operational_cash_usd >= reserve_floor

    # 4. Consistent linking
    assert sim_out.decision_id == approved_dec.decision_id
    assert learn_ev.correlation_id == approved_dec.decision_id


def test_provisional_tax_simulation_only_blocks_live_execution():
    """Verify that provisional/synthetic tax profile allows sandbox simulation but blocks live execution."""
    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    state = _build_test_treasury_state(cash=65_000_000.0, obligations=12_500_000.0, now=now)
    synth_profile = create_synthetic_demo_profile(entity_id="SYNTH_CORP_01")

    orch = DecisionOrchestrator()
    res = orch.run_cycle(
        current_state=state,
        as_of_timestamp=now,
        tax_profile=synth_profile,
        auto_simulate_execution=False,
    )
    dec = res.decision_record
    assert dec.metadata.get("tax_profile_status") == "SYNTHETIC_DEMO"
    assert res.approval_status == ApprovalStatus.PENDING_APPROVAL.value

    expl_service = DecisionExplanationService()

    # Attempt live operational execution -> MUST BE BLOCKED
    live_ok, live_msg, _, _, _ = expl_service.execute_human_workflow(
        decision_record=dec,
        action_type="APPROVED",
        actor_name="Director of Treasury Operations",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Attempting live commit with synthetic profile",
        treasury_state=state,
        as_of_time=now + timedelta(minutes=10),
        is_live_execution=True,
    )
    assert live_ok is False
    assert "Live execution BLOCKED" in live_msg
    assert "provisional/synthetic" in live_msg

    # Explicitly authorized sandbox simulation -> ALLOWED WITH PROMINENT DISCLAIMER
    sim_ok, sim_msg, sim_dec, sim_out, sim_ev = expl_service.execute_human_workflow(
        decision_record=dec,
        action_type="APPROVED",
        actor_name="Director of Treasury Operations",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Approved for sandbox simulation study only.",
        treasury_state=state,
        as_of_time=now + timedelta(minutes=10),
        is_live_execution=False,
    )
    assert sim_ok is True
    assert sim_dec.metadata.get("is_simulation_only") is True
    assert sim_dec.metadata.get("live_execution_eligible") is False
    assert "PROVISIONAL_TAX_ASSUMPTIONS" in sim_dec.metadata.get("provisional_tax_disclaimer", "")
    assert sim_ev.payload.get("is_simulation_only") is True
    assert sim_ev.payload.get("live_execution_eligible") is False


def test_repo_target_mapping_and_direction_aware_accounting():
    """Verify destination repo target mapping and contrast repo investment vs repo borrowing accounting."""
    now = datetime(2026, 1, 11, 17, 0, 0, tzinfo=timezone.utc)
    state = _build_test_treasury_state(cash=65_000_000.0, obligations=12_500_000.0, now=now)
    stat_profile = create_statutory_us_corp_profile(entity_id="US_CORP_01", tax_year=2026)

    orch = DecisionOrchestrator()
    res = orch.run_cycle(
        current_state=state,
        as_of_timestamp=now,
        tax_profile=stat_profile,
        auto_simulate_execution=False,
    )
    dec = res.decision_record

    # 1. Target instrument for PLACE_REPO must be specific repo instrument, not ambiguous CASH
    place_cand = next(
        (a for a in [dec.selected_action] + dec.alternatives if isinstance(a, dict) and a.get("action_type") == "PLACE_REPO"),
        None,
    )
    assert place_cand is not None, "PLACE_REPO candidate must be evaluated by optimizer"
    assert place_cand.get("target_asset") == "T1_REPO_PORTFOLIO"
    assert place_cand.get("destination_instrument") == "T1_REPO_PORTFOLIO"
    assert place_cand.get("source_asset") == "TIER_0_CASH"

    repo_rec = DecisionRecord(selected_action=place_cand)
    assert repo_rec.target_asset == "T1_REPO_PORTFOLIO"
    assert repo_rec.destination_instrument == "T1_REPO_PORTFOLIO"
    assert repo_rec.source_asset == "TIER_0_CASH"

    # 2. Simulate PLACE_REPO: Cash decreases, Repo asset increases, Liabilities remain 0
    place_spec = {
        "type": "PLACE_REPO",
        "amount_usd": 10_000_000.0,
        "asset_id": "T1_REPO_PORTFOLIO",
        "decision_id": "dec-test-place",
    }
    state_after_place, out_place = orch.ledger.simulate_action(state, place_spec, res.forecast_bundle)
    assert out_place.execution_status == "SUCCESS"
    assert out_place.realized_cash_movements.get("sweep_repo") == -10_000_000.0
    assert state_after_place.operational_cash_usd == state.operational_cash_usd - 10_000_000.0
    assert state_after_place.asset_positions["T1_REPO_PORTFOLIO"].market_value_usd == (
        state.asset_positions["T1_REPO_PORTFOLIO"].market_value_usd + 10_000_000.0
    )
    assert state_after_place.emergency_credit_drawn_usd == 0.0

    # 3. Contrast with DRAW_REPO_30D (Borrowing): Cash increases, Liabilities increase, Repo asset UNCHANGED
    borrow_spec = {
        "type": "DRAW_REPO_30D",
        "amount_usd": 5_000_000.0,
        "decision_id": "dec-test-borrow",
    }
    state_after_borrow, out_borrow = orch.ledger.simulate_action(state, borrow_spec, res.forecast_bundle)
    assert out_borrow.execution_status == "SUCCESS"
    assert out_borrow.realized_cash_movements.get("borrowing_proceeds") == 5_000_000.0
    assert state_after_borrow.operational_cash_usd == state.operational_cash_usd + 5_000_000.0
    # Liabilities increased
    assert state_after_borrow.emergency_credit_drawn_usd == 5_000_000.0
    # Repo asset did NOT increase (not masquerading as repo investment)
    assert state_after_borrow.asset_positions["T1_REPO_PORTFOLIO"].market_value_usd == (
        state.asset_positions["T1_REPO_PORTFOLIO"].market_value_usd
    )


def test_statutory_profile_entity_and_jurisdiction_matching():
    """Verify that statutory profiles require matching corporate entity and legal jurisdiction."""
    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    state = _build_test_treasury_state(cash=65_000_000.0, obligations=12_500_000.0, now=now)

    # Valid profile matching target
    valid_profile = create_statutory_us_corp_profile(entity_id="US_CORP_01", tax_year=2026)
    is_app, err = valid_profile.is_applicable_to(entity_id="US_CORP_01", jurisdiction="US_FED_CORP_STATUTORY")
    assert is_app is True
    assert err is None

    # Mismatched entity
    is_app_bad_ent, err_ent = valid_profile.is_applicable_to(entity_id="FOREIGN_CORP_99", jurisdiction="US_FED_CORP_STATUTORY")
    assert is_app_bad_ent is False
    assert "Entity mismatch" in str(err_ent)

    # Mismatched jurisdiction
    is_app_bad_jur, err_jur = valid_profile.is_applicable_to(entity_id="US_CORP_01", jurisdiction="UK_HMRC")
    assert is_app_bad_jur is False
    assert "Jurisdiction mismatch" in str(err_jur)


def test_live_execution_fails_closed_when_production_gateway_unconfigured_even_with_statutory_tax():
    """Verify that even with an approved statutory tax profile, live execution fails closed when gateways are unconfigured."""
    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    state = _build_test_treasury_state(cash=65_000_000.0, obligations=12_500_000.0, now=now)
    # Ensure default unconfigured gateway state
    assert state.production_gateway_status == "UNCONFIGURED"
    assert state.is_production_gateway_healthy is False

    stat_profile = create_statutory_us_corp_profile(entity_id="US_CORP_01", tax_year=2026)
    orch = DecisionOrchestrator()
    res = orch.run_cycle(
        current_state=state,
        as_of_timestamp=now,
        tax_profile=stat_profile,
        auto_simulate_execution=False,
    )
    dec = res.decision_record
    assert dec.metadata.get("tax_profile_status") == "STATUTORY_APPROVED"

    expl_service = DecisionExplanationService()
    live_ok, live_msg, _, live_out, _ = expl_service.execute_human_workflow(
        decision_record=dec,
        action_type="APPROVED",
        actor_name="Director of Treasury Operations",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Attempting live production dispatch",
        treasury_state=state,
        as_of_time=now + timedelta(minutes=10),
        is_live_execution=True,
    )

    # Must FAIL CLOSED: cannot execute live transactions without configured gateway
    assert live_ok is False
    assert live_out is None
    assert "Live execution BLOCKED" in live_msg
    assert "gateways are not configured or healthy" in live_msg
    assert "gateway_status='UNCONFIGURED'" in live_msg


def test_facade_and_orchestrator_live_execution_fail_closed():
    """Verify that ApplicationFacade and DecisionOrchestrator live execution entry points fail closed."""
    from reserve_juggler.integration.facade import ApplicationFacade

    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    state = _build_test_treasury_state(cash=65_000_000.0, obligations=12_500_000.0, now=now)
    stat_profile = create_statutory_us_corp_profile(entity_id="US_CORP_01", tax_year=2026)

    # 1. Orchestrator dispatch_live_execution
    orch = DecisionOrchestrator()
    res = orch.run_cycle(
        current_state=state,
        as_of_timestamp=now,
        tax_profile=stat_profile,
        auto_simulate_execution=False,
    )
    dec = res.decision_record

    orch_ok, orch_msg, orch_out = orch.dispatch_live_execution(
        decision_record=dec,
        treasury_state=state,
        actor_name="Director of Treasury Operations",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Live production dispatch attempt",
    )
    assert orch_ok is False
    assert orch_out is None
    assert "Live execution BLOCKED" in orch_msg
    assert "gateway_status='UNCONFIGURED'" in orch_msg

    # 2. ApplicationFacade submit_human_action with is_live_execution=True
    facade = ApplicationFacade(use_synthetic=True)
    facade.current_scenario = "NORMAL"
    # Seed a decision
    facade._latest_decision_record = dec
    fac_ok, fac_msg = facade.submit_human_action(
        action_type="APPROVED",
        actor_name="Director of Treasury Operations",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Live production dispatch via facade",
        is_live_execution=True,
    )
    assert fac_ok is False
    assert "Live execution BLOCKED" in fac_msg


def test_standardized_calibration_reliability_labels_and_conservative_policy():
    """Verify standardization of EMPIRICALLY_VALIDATED, CONDITIONALLY_CALIBRATED_FALLBACK, and DEGRADED."""
    from reserve_juggler.forecasting.uncertainty import ORScenarioReconciler
    from reserve_juggler.forecasting.service import ForecastService
    from reserve_juggler.risk.liquidity_shield import assess_liquidity_reserve

    reconciler = ORScenarioReconciler()
    d_true = [15_000_000.0] * 200
    d_pred = [15_000_000.0] * 200
    w_true = [16_000_000.0] * 200
    w_pred = [16_000_000.0] * 200

    # Horizon 1h with N=200 -> EMPIRICALLY_VALIDATED
    reconciler.fit_residuals(d_true, d_pred, w_true, w_pred, horizon_hours=1)
    pt1 = reconciler.reconcile_point(15_000_000.0, 16_000_000.0, horizon_hours=1)
    assert pt1["calibration_reliability_status"] == "EMPIRICALLY_VALIDATED"

    # Horizon 6h with N=200 -> CONDITIONALLY_CALIBRATED_FALLBACK (raw undercoverage requires 1.55x buffer)
    reconciler.fit_residuals(d_true, d_pred, w_true, w_pred, horizon_hours=6)
    pt6 = reconciler.reconcile_point(15_000_000.0, 16_000_000.0, horizon_hours=6)
    assert pt6["calibration_reliability_status"] == "CONDITIONALLY_CALIBRATED_FALLBACK"

    # Limited sample size N=50 -> DEGRADED
    reconciler.fit_residuals(d_true[:50], d_pred[:50], w_true[:50], w_pred[:50], horizon_hours=1)
    pt_deg = reconciler.reconcile_point(15_000_000.0, 16_000_000.0, horizon_hours=1)
    assert pt_deg["calibration_reliability_status"] == "DEGRADED"

    # Degraded forecast point triggers conservative policy in liquidity shield
    now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    state = _build_test_treasury_state(cash=65_000_000.0, obligations=10_000_000.0, now=now)
    from reserve_juggler.contracts import ForecastBundle, ForecastPoint

    deg_fp = ForecastPoint(
        as_of_timestamp=now,
        target_timestamp=now + timedelta(hours=6),
        horizon_hours=6,
        deposit_point_forecast=10_000_000.0,
        withdrawal_point_forecast=12_000_000.0,
        deposit_p10=8_000_000.0,
        withdrawal_p90=15_000_000.0,
        net_outflow_p95=5_000_000.0,
        net_outflow_marginal_proxy=7_000_000.0,
        calibration_is_reliable=False,
        calibration_reliability_status="DEGRADED",
    )
    bundle = ForecastBundle(forecast_timestamp=now, points=[deg_fp])
    res_deg = assess_liquidity_reserve(state, bundle, use_calibrated_net_quantile=True)
    # The degraded point triggers 1.15x buffer on marginal proxy (7M * 1.15 = 8.05M deficit)
    assert res_deg.conservative_net_deficit >= 7_000_000.0 * 1.15
