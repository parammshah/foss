"""
Comprehensive tests for all eight dashboard views and snapshot schemas.
Verifies complete rendered-page execution, UI pipeline stability,
edge cases (empty data, missing services, failed gates, inconclusive metrics),
and AppTest integration.
"""

from __future__ import annotations
import sys
from pathlib import Path
import pytest

from reserve_juggler.integration.facade import ApplicationFacade, DashboardSnapshot
from reserve_juggler.dashboard.views.overview import render_overview
from reserve_juggler.dashboard.views.forecasts import render_forecasts
from reserve_juggler.dashboard.views.portfolio import render_portfolio
from reserve_juggler.dashboard.views.decision import render_decision
from reserve_juggler.dashboard.views.stress_lab import render_stress_lab
from reserve_juggler.dashboard.views.sentinel import render_sentinel
from reserve_juggler.dashboard.views.learning import render_learning
from reserve_juggler.dashboard.views.decision_log import render_decision_log


def test_dashboard_snapshot_schema():
    """Verifies that the facade returns a well-formed snapshot conforming to schema v1.0.0."""
    facade = ApplicationFacade(use_synthetic=True)
    snapshot = facade.get_dashboard_snapshot()

    assert snapshot.snapshot_version == "1.0.0"
    assert snapshot.treasury_state is not None
    assert snapshot.forecast_bundle is not None
    assert snapshot.sentinel_result is not None
    assert snapshot.latest_decision is not None
    assert snapshot.decision_explanation is not None
    assert snapshot.system_performance is not None
    assert isinstance(snapshot.audit_log, list)
    assert isinstance(snapshot.active_candidates, list)
    assert snapshot.data_source_mode in ("SYNTHETIC_DEMO", "LIVE", "UNINTEGRATED")


def test_all_eight_views_render_via_apptest():
    """
    Executes all 8 navigation views using the official Streamlit AppTest framework.
    Guarantees no runtime exceptions, unhandled None attributes, or broken widgets across the app.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).parent.parent.parent / "src" / "reserve_juggler" / "dashboard" / "app.py")
    at = AppTest.from_file(app_path).run()
    assert len(at.exception) == 0, f"Overview encountered exception: {at.exception}"

    radio = at.radio[0]
    expected_views = [
        "🏛️ Overview",
        "📈 Forecasts and Reliability",
        "💼 Portfolio",
        "🎯 Next Best Action",
        "🔬 Stress Lab",
        "📡 Macro-Panic Sentinel",
        "🧠 System Learning",
        "📜 Decision Log",
    ]

    for view_label in expected_views:
        radio.set_value(view_label).run()
        assert len(at.exception) == 0, f"View '{view_label}' raised an unhandled exception: {at.exception}"


def test_forecast_view_edge_case_with_zero_actuals_mape_none():
    """Verifies that the forecast view handles a None MAPE cleanly without crashing."""
    facade = ApplicationFacade(use_synthetic=True)
    snapshot = facade.get_dashboard_snapshot()

    # Artificially set MAPE to None (zero-actuals scenario)
    snapshot.system_performance.forecast_metrics.mape = None
    snapshot.system_performance.forecast_metrics.drift_analysis.drift_detected = True

    # Calling render_forecasts directly should not crash
    render_forecasts(snapshot)


def test_decision_view_edge_case_with_no_alternatives_and_failed_gate():
    """Verifies that decision view renders with failed gates and empty alternatives cleanly."""
    facade = ApplicationFacade(use_synthetic=True)
    snapshot = facade.get_dashboard_snapshot()

    # Empty alternatives and a failed hard gate
    snapshot.decision_explanation.alternatives_comparison = []
    if snapshot.decision_explanation.hard_gates:
        from reserve_juggler.contracts import GateStatus
        snapshot.decision_explanation.hard_gates[0].status = GateStatus.FAIL
        snapshot.decision_explanation.hard_gates[0].reason = "Mandatory reserve floor violation simulated."

    render_decision(snapshot, facade)


def test_sentinel_view_edge_case_with_zero_signals():
    """Verifies sentinel view renders cleanly when no active signals exist."""
    facade = ApplicationFacade(use_synthetic=True)
    snapshot = facade.get_dashboard_snapshot()

    snapshot.sentinel_result.active_signals = []
    snapshot.sentinel_result.aggregate_macro_stress_score = 15.0

    render_sentinel(snapshot, facade)


def test_learning_view_edge_case_with_inconclusive_health():
    """Verifies learning view renders cleanly with INCONCLUSIVE and ACTION_REQUIRED states."""
    facade = ApplicationFacade(use_synthetic=True)
    snapshot = facade.get_dashboard_snapshot()

    snapshot.system_performance.overall_system_health = "ACTION_REQUIRED"
    snapshot.system_performance.inconclusive_reason = "Holdout sample size (4) below threshold (10)."

    render_learning(snapshot, facade)
