"""
Comprehensive tests for all eight dashboard views and snapshot schemas.
Verifies complete rendered-page execution, UI pipeline stability,
edge cases (empty data, missing services, failed gates, inconclusive metrics),
and Streamlit AppTest integration.
"""

from __future__ import annotations
import sys
from pathlib import Path
import pytest

_root = str(Path(__file__).resolve().parent.parent.parent)
_src = str(Path(__file__).resolve().parent.parent.parent / "src")
_frontend = str(Path(__file__).resolve().parent.parent.parent / "frontend" / "legacy_streamlit")
for p in (_root, _src, _frontend):
    if p not in sys.path:
        sys.path.insert(0, p)

from reserve_juggler.integration.facade import ApplicationFacade, DashboardSnapshot
from views.overview import render_overview
from views.forecasts import render_forecasts
from views.portfolio import render_portfolio
from views.decision import render_decision
from views.stress_lab import render_stress_lab
from views.sentinel import render_sentinel
from views.learning import render_learning
from views.decision_log import render_decision_log


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
    facade.close()


def test_all_eight_views_render_via_apptest():
    """
    Executes all 8 navigation views using the official Streamlit AppTest framework.
    Guarantees no runtime exceptions, unhandled None attributes, or broken widgets across the app.
    """
    from streamlit.testing.v1 import AppTest

    app_path = str(Path(__file__).resolve().parent.parent.parent / "frontend" / "legacy_streamlit" / "app.py")
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
    facade.close()


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
    facade.close()


def test_sentinel_view_edge_case_with_zero_signals():
    """Verifies sentinel view renders cleanly when no active signals exist."""
    facade = ApplicationFacade(use_synthetic=True)
    snapshot = facade.get_dashboard_snapshot()

    snapshot.sentinel_result.active_signals = []
    snapshot.sentinel_result.total_events_processed = 0
    snapshot.sentinel_result.aggregate_macro_stress_score = 0.0

    render_sentinel(snapshot, facade)
    facade.close()


def test_learning_view_edge_case_with_inconclusive_health():
    """Verifies learning view handles inconclusive evaluation and empty candidates without crashing."""
    facade = ApplicationFacade(use_synthetic=True)
    snapshot = facade.get_dashboard_snapshot()

    snapshot.system_performance.overall_system_health = "INCONCLUSIVE"
    snapshot.active_candidates = []

    render_learning(snapshot, facade)
    facade.close()


def test_all_eight_pages_render_under_populated_and_missing_data_conditions():
    """
    Verifies that all 8 views render without exceptions even when snapshot fields are None or empty.
    """
    facade = ApplicationFacade(use_synthetic=True)
    snapshot = facade.get_dashboard_snapshot()

    # Populate with None/empty edge cases
    snapshot.treasury_state.protected_reserve_floor = None
    snapshot.latest_decision.treasury_safety_score = None
    snapshot.active_candidates = []
    snapshot.audit_log = []

    # Direct rendering execution of all 8 view functions
    render_overview(snapshot)
    render_forecasts(snapshot)
    render_portfolio(snapshot)
    render_decision(snapshot, facade)
    render_stress_lab(snapshot, facade)
    render_sentinel(snapshot, facade)
    render_learning(snapshot, facade)
    render_decision_log(snapshot)
    facade.close()
