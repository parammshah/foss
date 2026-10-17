"""
Smoke tests for dashboard view rendering functions to guarantee no unhandled exceptions in the UI pipeline.
"""

import pytest
from reserve_juggler.integration.facade import ApplicationFacade


def test_dashboard_snapshot_schema():
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
