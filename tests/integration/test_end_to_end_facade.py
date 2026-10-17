"""
End-to-End integration tests for ApplicationFacade.
Verifies the complete lifecycle: event ingestion -> risk signal -> decision -> explanation -> human approval -> simulated execution -> learning loop.
"""

import pytest
from reserve_juggler.integration.facade import ApplicationFacade
from reserve_juggler.integration.contracts import UserRole


def test_end_to_end_event_to_decision_to_learning_trace():
    """Traces full lifecycle through the application facade."""
    facade = ApplicationFacade(use_synthetic=True)

    # 1. Fetch initial snapshot
    snapshot1 = facade.get_dashboard_snapshot()
    assert snapshot1.snapshot_version == "1.0.0"
    assert snapshot1.treasury_state.available_cash > 0
    assert len(snapshot1.sentinel_result.active_signals) > 0
    assert snapshot1.latest_decision is not None
    assert snapshot1.decision_explanation is not None

    initial_events_count = len(facade.learning_service.store.get_all())

    # 2. Submit human approval
    ok, msg = facade.submit_human_action(
        action_type="APPROVED",
        actor_name="Director of Treasury",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="End-to-end integration test execution.",
    )
    assert ok
    assert "successfully recorded" in msg

    # 3. Check that learning events were generated and persisted
    events_after = facade.learning_service.store.get_all()
    assert len(events_after) > initial_events_count

    # 4. Check that outcome matcher has pairs
    pairs = facade.learning_service.matcher.match_all()
    assert len(pairs) >= 1

    # 5. Evaluate performance report
    perf = facade.learning_service.evaluate_system_performance()
    assert perf.total_learning_events == len(events_after)


def test_facade_scenario_switching():
    """Verifies operational scenario transitions (Normal -> Banking Stress)."""
    facade = ApplicationFacade(use_synthetic=True)

    # Baseline
    snap_normal = facade.get_dashboard_snapshot()
    assert snap_normal.active_scenario == "NORMAL"

    # Switch to Banking Stress
    facade.switch_scenario("BANKING_STRESS")
    snap_stress = facade.get_dashboard_snapshot()
    assert snap_stress.active_scenario == "BANKING_STRESS"
    # Macro stress index should rise under banking stress
    assert snap_stress.sentinel_result.aggregate_macro_stress_score > 30.0


def test_stress_lab_waterfall_and_shortfall_calculation():
    """Verifies that stress lab correctly identifies liquidity shortfalls under severe shocks."""
    facade = ApplicationFacade(use_synthetic=True)

    # Mild shock -> no shortfall
    res_mild = facade.run_stress_lab_scenario(
        deposit_shock_pct=10.0,
        rate_shock_bps=20.0,
        credit_unavailable=False,
    )
    assert not res_mild["is_shortfall"]
    assert res_mild["shortfall_amount"] == 0.0

    # Severe deposit run (200% shock) -> triggers reserve shortfall
    res_severe = facade.run_stress_lab_scenario(
        deposit_shock_pct=200.0,
        rate_shock_bps=150.0,
        credit_unavailable=True,
    )
    assert res_severe["is_shortfall"]
    assert res_severe["shortfall_amount"] > 0.0
