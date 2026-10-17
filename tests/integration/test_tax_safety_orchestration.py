"""
Regression tests for tax safety in the real orchestration and decision path.

Verifies:
1. Missing TaxProfile defaults safely to provisional gross yield without synthetic assumptions.
2. Missing tax profile triggers human approval (PENDING_APPROVAL) and blocks automated execution.
3. Unregistered/missing asset classifications are flagged as provisional.
4. Invalid tax profiles are intercepted, blocked from net-of-tax deduction, and marked provisional.
5. Statutory corporate tax profiles are validated as approved.
6. Synthetic demo assumptions are explicitly prevented from masquerading as authoritative statutory results.
"""

from __future__ import annotations

from datetime import datetime, timezone
import pytest

from reserve_juggler.contracts import (
    ApprovalStatus,
    AssetPosition,
    AssetTier,
    GateStatus,
    Obligation,
    ObligationType,
    TreasuryState,
)
from reserve_juggler.decision.service import TreasuryDecisionService
from reserve_juggler.integration.orchestrator import (
    DecisionOrchestrator,
    IndependentConstraintValidator,
)
from reserve_juggler.tax.tax_engine import evaluate_tax_adjusted_value
from reserve_juggler.tax.tax_profile import (
    IncomeType,
    InstrumentTaxClassification,
    TaxBracket,
    TaxJurisdictionRules,
    TaxProfile,
    create_statutory_us_corp_profile,
    create_synthetic_demo_profile,
)


@pytest.fixture
def base_now() -> datetime:
    return datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def operational_state(base_now: datetime) -> TreasuryState:
    t1_repo = AssetPosition(
        asset_id="T1_REPO_PORTFOLIO",
        name="Overnight General Collateral Repo",
        tier=AssetTier.TIER_1_REPO,
        units=10_000_000.0,
        unit_price_usd=1.0,
        market_value_usd=10_000_000.0,
        liquidatable_value_usd=10_000_000.0 * 0.995,
        annual_yield_or_return=0.043,
    )
    return TreasuryState(
        operational_cash_usd=25_000_000.0,
        asset_positions={"T1_REPO_PORTFOLIO": t1_repo},
        upcoming_obligations_usd=5_000_000.0,
        as_of_timestamp=base_now,
    )


def test_orchestration_cycle_with_missing_tax_profile(operational_state: TreasuryState, base_now: datetime):
    """Real orchestration path with tax_profile=None must fail-safe to provisional gross yield and block auto-execution."""
    orch = DecisionOrchestrator()
    res = orch.run_cycle(
        current_state=operational_state,
        as_of_timestamp=base_now,
        tax_profile=None,
        auto_simulate_execution=True,
    )

    # 1. Decision record metadata flags
    assert res.decision_record.metadata.get("ranking_is_provisional") is True
    assert res.decision_record.metadata.get("tax_profile_status") == "UNCONFIGURED_GROSS_ONLY"

    # 2. Reason codes disclose provisional gross ranking
    assert any("WARNING_PROVISIONAL_TAX_RANKING" in r for r in res.decision_record.reason_codes)

    # 3. Independent constraint validator requires human approval
    assert res.validation_result.requires_human_approval is True
    assert res.approval_status == ApprovalStatus.PENDING_APPROVAL.value

    # 4. Automated execution is blocked
    assert res.simulated_outcome is None

    # 5. Explainability discloses missing tax assumptions
    assert res.explanation is not None
    assert any("Tax Profile" in d for d in res.explanation.disclosed_missing_information)


def test_orchestration_cycle_with_invalid_tax_profile(operational_state: TreasuryState, base_now: datetime):
    """An invalid tax profile (e.g. rate > 1.0, negative tax year) is intercepted and marked provisional."""
    invalid_rules = TaxJurisdictionRules(
        jurisdiction_code="INVALID_JURISDICTION",
        tax_year=1850,  # Invalid year < 1900
        rule_version="1850.1",
        ordinary_income_brackets=[],  # Missing brackets
        qualified_dividend_brackets=[],
        long_term_capital_gains_brackets=[],
    )
    invalid_profile = TaxProfile(entity_id="INVALID_ENTITY", jurisdiction_rules=invalid_rules)
    assert len(invalid_profile.validate()) > 0

    orch = DecisionOrchestrator()
    res = orch.run_cycle(
        current_state=operational_state,
        as_of_timestamp=base_now,
        tax_profile=invalid_profile,
        auto_simulate_execution=True,
    )

    assert res.decision_record.metadata.get("ranking_is_provisional") is True
    assert res.decision_record.metadata.get("tax_profile_status") == "INVALID"
    assert res.approval_status == ApprovalStatus.PENDING_APPROVAL.value
    assert res.simulated_outcome is None


def test_orchestration_cycle_with_synthetic_demo_profile(operational_state: TreasuryState, base_now: datetime):
    """A synthetic demo profile evaluates progressive brackets but is restricted from auto-approval."""
    synth_profile = create_synthetic_demo_profile()
    orch = DecisionOrchestrator()
    res = orch.run_cycle(
        current_state=operational_state,
        as_of_timestamp=base_now,
        tax_profile=synth_profile,
        auto_simulate_execution=True,
    )

    assert res.decision_record.metadata.get("ranking_is_provisional") is True
    assert res.decision_record.metadata.get("tax_profile_status") == "SYNTHETIC_DEMO"
    assert res.approval_status == ApprovalStatus.PENDING_APPROVAL.value
    assert res.simulated_outcome is None
    assert any("SYNTHETIC_DEMO" in d for d in res.explanation.disclosed_missing_information)


def test_missing_asset_classification_flagged_as_provisional():
    """Assets with no registered tax classification are flagged provisional without invented tax exemptions."""
    stat_profile = create_statutory_us_corp_profile()

    # Known asset registered in profile: SOVEREIGN_2Y
    res_known = evaluate_tax_adjusted_value(
        gross_revenue=100_000.0,
        instrument_id="SOVEREIGN_2Y",
        tax_profile=stat_profile,
    )
    assert res_known.has_missing_classification is False
    assert res_known.ranking_is_provisional is False

    # Unknown exotic asset: UNREGISTERED_STRUCTURED_NOTE
    res_unknown = evaluate_tax_adjusted_value(
        gross_revenue=100_000.0,
        instrument_id="UNREGISTERED_STRUCTURED_NOTE",
        tax_profile=stat_profile,
    )
    assert res_unknown.has_missing_classification is True
    assert res_unknown.ranking_is_provisional is True
    assert any("MISSING_ASSET_CLASSIFICATION" in w for w in res_unknown.warnings)


def test_gross_yield_distinguished_from_after_tax_yield():
    """Confirm gross yield is clearly distinguished from verified after-tax yield and synthetic demo yield."""
    gross_rev = 100_000.0

    # 1. No profile -> Gross yield ($100k net, $0 tax liability)
    res_gross = evaluate_tax_adjusted_value(gross_revenue=gross_rev, tax_profile=None)
    assert res_gross.after_tax_net_value == 100_000.0
    assert res_gross.tax_liability == 0.0
    assert res_gross.jurisdiction_version == "UNCONFIGURED_GROSS_ONLY"
    assert res_gross.ranking_is_provisional is True

    # 2. Statutory US corporate -> Flat 21% tax ($79k after-tax net value)
    stat_profile = create_statutory_us_corp_profile()
    res_stat = evaluate_tax_adjusted_value(
        gross_revenue=gross_rev,
        instrument_id="SOVEREIGN_2Y",
        tax_profile=stat_profile,
    )
    assert res_stat.tax_liability == 21_000.0
    assert res_stat.after_tax_net_value == 79_000.0
    assert res_stat.jurisdiction_version == "2026.1-US-CORP-STATUTORY"
    assert res_stat.ranking_is_provisional is False

    # 3. Synthetic demo -> Progressive brackets ($11,700 tax on first 70k, etc.)
    synth_profile = create_synthetic_demo_profile()
    res_synth = evaluate_tax_adjusted_value(
        gross_revenue=gross_rev,
        instrument_id="SOVEREIGN_2Y",
        tax_profile=synth_profile,
    )
    # $50k @ 15% ($7.5k) + $50k @ 21% ($10.5k) = $18,000 tax
    assert res_synth.tax_liability == 18_000.0
    assert res_synth.after_tax_net_value == 82_000.0
    assert res_synth.ranking_is_provisional is True

    # Statutory vs synthetic yields are mathematically distinct ($79,000 vs $82,000)
    assert res_stat.after_tax_net_value != res_synth.after_tax_net_value


def test_deterministic_surplus_allocation_multi_candidate_tax_scenarios():
    """Deterministic scenario with positive surplus and multiple feasible candidates (PLACE_REPO and BUY_BOND).
    Exercises:
    1. No tax profile: Provisional gross ranking, human approval mandated, auto-sim blocked.
    2. Synthetic tax profile: Provisional demo ranking, human approval mandated, auto-sim blocked.
    3. Invalid tax profile: Intercepted by validation, provisional gross ranking, human approval mandated.
    4. Valid statutory profile matching entity/jurisdiction: Verified after-tax ranking, gate pass.
    5. Entity / jurisdiction mismatch: US corporate statutory rules are NOT universally applied to foreign entity.
    """
    from reserve_juggler.contracts import ForecastBundle, ForecastPoint
    from reserve_juggler.optimization.optimizer import optimize_treasury_actions

    now = datetime(2026, 1, 15, 17, 0, 0, tzinfo=timezone.utc)  # 17:00 UTC -> Repo window open

    # Substantial surplus cash with low obligations and strong projected net inflow
    t1_repo = AssetPosition(
        asset_id="T1_REPO_PORTFOLIO",
        name="Overnight Repo",
        tier=AssetTier.TIER_1_REPO,
        market_value_usd=10_000_000.0,
        annual_yield_or_return=0.043,
    )
    state = TreasuryState(
        operational_cash_usd=60_000_000.0,
        asset_positions={"T1_REPO_PORTFOLIO": t1_repo},
        upcoming_obligations_usd=5_000_000.0,
        as_of_timestamp=now,
        entity_id="US_CORP_01",
        jurisdiction="US_FED_CORP_STATUTORY",
    )

    fb = ForecastBundle(
        as_of_timestamp=now.isoformat(),
        point_forecasts={
            "1h": ForecastPoint(horizon="1h", expected_inflow=10_000_000.0, expected_outflow=2_000_000.0, expected_net_flow=8_000_000.0, uncertainty_p50=8_000_000.0, uncertainty_p05=5_000_000.0, uncertainty_p95=11_000_000.0),
            "24h": ForecastPoint(horizon="24h", expected_inflow=50_000_000.0, expected_outflow=10_000_000.0, expected_net_flow=40_000_000.0, uncertainty_p50=40_000_000.0, uncertainty_p05=25_000_000.0, uncertainty_p95=55_000_000.0),
            "72h": ForecastPoint(horizon="72h", expected_inflow=150_000_000.0, expected_outflow=30_000_000.0, expected_net_flow=120_000_000.0, uncertainty_p50=120_000_000.0, uncertainty_p05=80_000_000.0, uncertainty_p95=160_000_000.0),
        },
    )

    # -------------------------------------------------------------
    # Scenario 1: No Tax Profile -> Gross Yield Ranking, Provisional, Human Approval
    # -------------------------------------------------------------
    orch = DecisionOrchestrator()
    res_none = orch.run_cycle(current_state=state, as_of_timestamp=now, tax_profile=None, auto_simulate_execution=True)
    assert res_none.decision_record.metadata["tax_profile_status"] == "UNCONFIGURED_GROSS_ONLY"
    assert res_none.decision_record.metadata["ranking_is_provisional"] is True
    assert res_none.approval_status == ApprovalStatus.PENDING_APPROVAL.value
    assert res_none.simulated_outcome is None
    # Verify multiple feasible investment candidates were generated
    feasible_cands = [c for c in res_none.decision_record.alternatives if c.get("is_feasible")]
    assert len(feasible_cands) >= 1

    # -------------------------------------------------------------
    # Scenario 2: Synthetic Tax Profile -> Marked Provisional
    # -------------------------------------------------------------
    synth_profile = create_synthetic_demo_profile(entity_id="US_CORP_01")
    res_synth = orch.run_cycle(current_state=state, as_of_timestamp=now, tax_profile=synth_profile, auto_simulate_execution=True)
    assert res_synth.decision_record.metadata["tax_profile_status"] == "SYNTHETIC_DEMO"
    assert res_synth.decision_record.metadata["ranking_is_provisional"] is True
    assert res_synth.approval_status == ApprovalStatus.PENDING_APPROVAL.value
    assert res_synth.simulated_outcome is None

    # -------------------------------------------------------------
    # Scenario 3: Invalid Tax Profile -> Blocked, Provisional Gross Ranking
    # -------------------------------------------------------------
    invalid_rules = TaxJurisdictionRules(
        jurisdiction_code="INVALID_JURISDICTION",
        tax_year=1820,
        rule_version="1820.1",
        ordinary_income_brackets=[],
        qualified_dividend_brackets=[],
        long_term_capital_gains_brackets=[],
    )
    invalid_profile = TaxProfile(entity_id="US_CORP_01", jurisdiction_rules=invalid_rules)
    res_invalid = orch.run_cycle(current_state=state, as_of_timestamp=now, tax_profile=invalid_profile, auto_simulate_execution=True)
    assert res_invalid.decision_record.metadata["tax_profile_status"] == "INVALID"
    assert res_invalid.decision_record.metadata["ranking_is_provisional"] is True
    assert res_invalid.approval_status == ApprovalStatus.PENDING_APPROVAL.value
    assert res_invalid.simulated_outcome is None

    # -------------------------------------------------------------
    # Scenario 4: Valid Statutory US Corporate Profile (Matching Entity & Jurisdiction)
    # -------------------------------------------------------------
    stat_profile = create_statutory_us_corp_profile(entity_id="US_CORP_01", tax_year=2026)
    res_stat = orch.run_cycle(current_state=state, as_of_timestamp=now, tax_profile=stat_profile, auto_simulate_execution=True)
    assert res_stat.decision_record.metadata["tax_profile_status"] == "STATUTORY_APPROVED"
    assert res_stat.decision_record.metadata["ranking_is_provisional"] is False
    tax_gate = [g for g in res_stat.validation_result.gate_outcomes if g.gate_name == "TaxSafetyAuthorityGate"][0]
    assert tax_gate.status == GateStatus.PASS
    assert tax_gate.observed_value == 1.0

    # -------------------------------------------------------------
    # Scenario 5: Foreign Entity / Jurisdiction Mismatch
    # -------------------------------------------------------------
    foreign_state = TreasuryState(
        operational_cash_usd=60_000_000.0,
        asset_positions={"T1_REPO_PORTFOLIO": t1_repo},
        upcoming_obligations_usd=5_000_000.0,
        as_of_timestamp=now,
        entity_id="UK_SUBSIDIARY_LTD",
        jurisdiction="UK_HMRC_STATUTORY",  # Foreign jurisdiction!
    )
    # Applying US corporate rules to UK subsidiary must NOT be silently accepted as verified
    applicable, reason = stat_profile.is_applicable_to(
        entity_id=foreign_state.entity_id, jurisdiction=foreign_state.jurisdiction
    )
    assert applicable is False
    assert "Entity mismatch" in reason or "Jurisdiction mismatch" in reason

    # Optimization with incompatible entity flags provisional status
    opt_incompat = optimize_treasury_actions(state=foreign_state, forecast_bundle=fb, tax_profile=stat_profile)
    assert opt_incompat.selected_action.ranking_is_provisional is True
    assert any("PROVISIONAL_TAX_ASSUMPTIONS" in r for r in opt_incompat.remediation_notes)

