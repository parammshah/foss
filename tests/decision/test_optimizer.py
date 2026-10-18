"""
Tests for Operations Research optimizer, benchmark validation, and tier limits.
"""

from datetime import datetime, timedelta, timezone

import pytest

from reserve_juggler.optimization.models import (
    ActionType,
    get_tier_limits_for_tss,
)
from reserve_juggler.optimization.optimizer import (
    calculate_bridge_financing_cost,
    optimize_treasury_actions,
)
from reserve_juggler.risk.models import SafetyMode, SafetyScoreResult, TSSComponents
from tests.decision.fixtures import (
    ConcreteAssetPosition,
    ConcreteObligation,
    ConcreteTreasuryState,
    make_benchmark_repo_bridge_state,
    make_healthy_forecast_bundle,
)


def test_section_7_required_benchmark_example():
    """Verify Section 7 Benchmark Example:

    - $10M repo unlocking in 2 hours has $50,000 early exit fee (0.5%).
    - 2-hour bridge at 12% APR costs $273.97 using 8,760-hour annual convention.
    - Expected cost math: 10,000,000 * 0.12 * (2 / 8760) = 273.9726...
    - Optimizer must choose DRAW_BRIDGE_CREDIT over BREAK_REPO_EARLY.
    """
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    # 1. Verify exact mathematical formula for bridge interest
    bridge_cost = calculate_bridge_financing_cost(
        principal=10000000.0, apr=0.12, duration_hours=2.0
    )
    assert round(bridge_cost, 2) == 273.97

    # 2. Benchmark state with available bridge credit
    state = make_benchmark_repo_bridge_state(now, credit_available=15000000.0)

    opt_result = optimize_treasury_actions(state, bundle)
    selected = opt_result.selected_action

    # Selected action must be DRAW_BRIDGE_CREDIT
    assert selected.action_type == ActionType.DRAW_BRIDGE_CREDIT
    assert selected.amount == 10000000.0
    assert round(selected.total_estimated_cost, 2) == 273.97

    # Check candidate breakdown contains BREAK_REPO_EARLY with $50,000 penalty
    break_candidates = [
        c for c in opt_result.candidate_actions if c.action_type == ActionType.BREAK_REPO_EARLY
    ]
    assert len(break_candidates) >= 1
    assert round(break_candidates[0].total_estimated_cost, 2) == 50000.0


def test_benchmark_when_credit_unavailable_breaks_repo():
    """Verify that when credit is unavailable, optimizer breaks repo early to protect payouts:

    Must NOT delay a mandatory customer payout simply to save the fee!
    """
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    # State with $0 credit availability
    state = make_benchmark_repo_bridge_state(now, credit_available=0.0)

    opt_result = optimize_treasury_actions(state, bundle)
    selected = opt_result.selected_action

    # Selected action must be BREAK_REPO_EARLY (cannot draw bridge, cannot wait because payout would fail)
    assert selected.action_type == ActionType.BREAK_REPO_EARLY
    assert selected.amount == 10000000.0
    assert round(selected.total_estimated_cost, 2) == 50000.0

    # Verify WAIT_REPO_MATURITY was rejected because it would delay customer payout
    wait_candidates = [
        c for c in opt_result.candidate_actions if c.action_type == ActionType.WAIT_REPO_MATURITY
    ]
    assert len(wait_candidates) >= 1
    assert wait_candidates[0].is_feasible is False
    assert "Delaying payout is prohibited" in wait_candidates[0].reason_code


def test_tier_exposure_limits_by_tss():
    """Verify prototype tier exposure limits match Section 9."""
    # TSS 0–39: Tier 2 cap 0%, Tier 3 cap 0%
    lim_crit = get_tier_limits_for_tss(25.0)
    assert lim_crit.tier2_max_pct == 0.0
    assert lim_crit.tier3_max_pct == 0.0

    # TSS 40–59: Tier 2 cap 10%, Tier 3 cap 0%
    lim_def = get_tier_limits_for_tss(45.0)
    assert lim_def.tier2_max_pct == 0.10
    assert lim_def.tier3_max_pct == 0.0

    # TSS 60–79: Tier 2 cap 25%, Tier 3 cap 5%
    lim_bal = get_tier_limits_for_tss(72.0)
    assert lim_bal.tier2_max_pct == 0.25
    assert lim_bal.tier3_max_pct == 0.05

    # TSS 80–100: Tier 2 cap 35%, Tier 3 cap 10%
    lim_opt = get_tier_limits_for_tss(88.0)
    assert lim_opt.tier2_max_pct == 0.35
    assert lim_opt.tier3_max_pct == 0.10


def test_remediation_policy_for_existing_breaches():
    """Verify existing positions exceeding cap report breach without forced automatic fire-sale."""
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    # State with 50% Tier 2 bonds (exceeding 35% cap)
    bonds = ConcreteAssetPosition(
        asset_id="BONDS_50M",
        tier=2,
        market_value=50000000.0,
        liquidatable_value=49000000.0,
        availability=now,
    )
    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=50000000.0,
        asset_positions=[bonds],
    )

    opt_result = optimize_treasury_actions(state, bundle)

    assert len(opt_result.tier_breaches_detected) > 0
    assert any("Tier 2 exposure" in b for b in opt_result.tier_breaches_detected)
    assert any("Freeze new Tier 2 allocations; do NOT force fire-sale" in r for r in opt_result.remediation_notes)
