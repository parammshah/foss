"""
Comprehensive End-to-End Test Suite for Reserve Juggler API & Facade.
Tests all GET/POST routes, scenario registry validation & aliases, decision cycle timestamp normalization,
audit logs store interface, sentinel signals schema, candidate promotion/rollback, and static asset delivery.
"""

import pytest
from fastapi.testclient import TestClient

from reserve_juggler.api_server import create_app
from reserve_juggler.integration.facade import ApplicationFacade


@pytest.fixture
def client():
    facade = ApplicationFacade(use_synthetic=True)
    app = create_app(facade=facade)
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert data["mode"] == "SYNTHETIC_DEMO"
    assert data["scenario"] == "NORMAL"


def test_status_endpoint(client):
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["system_status"] == "ONLINE"
    assert data["data_source_mode"] == "SYNTHETIC_DEMO"
    assert "subsystems" in data
    assert "team1_forecasting_ledger" in data["subsystems"]
    assert "team2_decision_optimizer" in data["subsystems"]
    assert "team3_sentinel_risk" in data["subsystems"]


def test_dashboard_snapshot(client):
    res = client.get("/api/dashboard/snapshot")
    assert res.status_code == 200
    data = res.json()
    assert "treasury_state" in data
    assert "forecast_bundle" in data
    assert "sentinel_result" in data
    assert "latest_decision" in data
    assert data["active_scenario"] == "NORMAL"


def test_audit_logs_real_store_interface(client):
    """Verifies defect repair 2.A: /api/audit/logs calls real EventStore.get_all() with bounds."""
    res = client.get("/api/audit/logs?limit=25")
    assert res.status_code == 200
    data = res.json()
    assert "audit_trail" in data
    assert "learning_events" in data
    assert "count" in data
    assert isinstance(data["audit_trail"], list)
    assert isinstance(data["learning_events"], list)
    assert len(data["learning_events"]) <= 25


def test_sentinel_signals_schema(client):
    """Verifies defect repair 2.B: /api/sentinel/signals contains recommended_stress_scenarios and warnings."""
    res = client.get("/api/sentinel/signals")
    assert res.status_code == 200
    data = res.json()
    assert "sentinel_status" in data
    assert "aggregate_stress_score" in data
    assert "active_signals" in data
    assert "recommended_stress_scenarios" in data
    assert "recommended_scenarios" in data  # compatibility alias
    assert "data_quality_warnings" in data
    assert "signal_categories" in data
    assert "severities" in data
    assert "confidences" in data


def test_decision_cycle_completes_without_timestamp_error(client):
    """Verifies defect repair 2.C: /api/decision/cycle timestamp normalization."""
    res = client.post("/api/decision/cycle")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "cycle_id" in data
    assert "cycle_result" in data
    assert "snapshot" in data


def test_scenario_switching_and_alias_validation(client):
    """Verifies defect repair 2.E: canonical scenario registry with aliases and HTTP 400 on invalid."""
    # 1. Valid canonical scenario
    res = client.post("/api/scenario/switch", json={"scenario": "BANKING_STRESS"})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["active_scenario"] == "BANKING_STRESS"

    # Verify snapshot reflects changed state (reduced cash, stress obligations)
    snap_res = client.get("/api/dashboard/snapshot")
    assert snap_res.status_code == 200
    snap_data = snap_res.json()
    assert snap_data["active_scenario"] == "BANKING_STRESS"
    assert snap_data["treasury_state"]["operational_cash_usd"] == 18_000_000.0

    # 2. Alias mapping: SVB_COLLAPSE -> BANKING_STRESS
    res_alias = client.post("/api/scenario/switch", json={"scenario": "SVB_COLLAPSE"})
    assert res_alias.status_code == 200
    assert res_alias.json()["active_scenario"] == "BANKING_STRESS"

    # 3. Alias mapping: FLASH_CRASH -> RATE_HIKE_SHOCK
    res_flash = client.post("/api/scenario/switch", json={"scenario": "FLASH_CRASH"})
    assert res_flash.status_code == 200
    assert res_flash.json()["active_scenario"] == "RATE_HIKE_SHOCK"

    # 4. Invalid scenario returns HTTP 400
    res_bad = client.post("/api/scenario/switch", json={"scenario": "UNSUPPORTED_APOCALYPSE"})
    assert res_bad.status_code == 400
    assert "Invalid scenario" in res_bad.json()["detail"]


def test_stress_presets_and_simulation(client):
    # Presets
    res_presets = client.get("/api/stress/presets")
    assert res_presets.status_code == 200
    presets = res_presets.json()["presets"]
    assert len(presets) >= 4
    preset_ids = [p["id"] for p in presets]
    assert "svb_run" in preset_ids
    assert "baseline" in preset_ids

    # Simulation execution
    sim_payload = {
        "deposit_shock_pct": 25.0,
        "rate_shock_bps": 200.0,
        "repo_delay_days": 1,
        "credit_unavailable": False,
        "equity_drawdown_pct": 10.0,
    }
    res_sim = client.post("/api/stress/simulate", json=sim_payload)
    assert res_sim.status_code == 200
    sim_data = res_sim.json()
    assert "strategies" in sim_data
    assert "all_cash" in sim_data["strategies"]
    assert "greedy_yield" in sim_data["strategies"]
    assert "reserve_juggler" in sim_data["strategies"]


def test_decision_action_approval_and_duplicate_prevention(client):
    # First action: Approve
    action_payload = {
        "action_type": "APPROVED",
        "actor_name": "Chief Treasurer",
        "actor_role": "HEAD_OF_TREASURY",
        "stated_reason": "Approved under liquidity policy.",
        "is_live_execution": False,
    }
    res_act = client.post("/api/decision/action", json=action_payload)
    assert res_act.status_code == 200
    act_data = res_act.json()
    assert act_data["success"] is True

    # Duplicate execution attempt must fail
    res_dup = client.post("/api/decision/action", json=action_payload)
    assert res_dup.status_code == 200
    dup_data = res_dup.json()
    assert dup_data["success"] is False
    assert "Duplicate execution request rejected" in dup_data["message"]


def test_learning_candidate_promotion_and_rollback(client):
    """Verifies self-learning demo (Section 8): pre-seeded candidate can be promoted and rolled back."""
    # List candidates
    res_cands = client.get("/api/learning/candidates")
    assert res_cands.status_code == 200
    cands = res_cands.json()
    assert len(cands) >= 1
    target_cand = cands[0]
    cand_id = target_cand["candidate_id"]

    # Promote candidate
    promote_payload = {
        "candidate_id": cand_id,
        "actor_role": "HEAD_OF_TREASURY",
    }
    res_prom = client.post("/api/learning/promote", json=promote_payload)
    assert res_prom.status_code == 200
    prom_data = res_prom.json()
    assert prom_data["success"] is True
    assert "successfully promoted" in prom_data["message"]

    # Rollback candidate
    rollback_payload = {
        "candidate_id": cand_id,
        "actor_role": "HEAD_OF_TREASURY",
    }
    res_roll = client.post("/api/learning/rollback", json=rollback_payload)
    assert res_roll.status_code == 200
    roll_data = res_roll.json()
    assert roll_data["success"] is True
    assert "rolled back" in roll_data["message"]


def test_static_asset_serving(client):
    """Verifies that all primary platform and landing assets are served with HTTP 200."""
    routes_to_test = [
        ("/", "text/html"),
        ("/index.html", "text/html"),
        ("/platform.html", "text/html"),
        ("/reserve_juggler_treasury_platform.html", "text/html"),
        ("/css/platform.css", "text/css"),
        ("/css/landing.css", "text/css"),
        ("/css/components.css", "text/css"),
        ("/css/base.css", "text/css"),
        ("/js/platform.js", "javascript"),
        ("/js/api-client.js", "javascript"),
        ("/js/sim-theatre.js", "javascript"),
    ]
    for route, expected_content_type in routes_to_test:
        res = client.get(route)
        assert res.status_code == 200, f"Failed on route {route}"
        assert expected_content_type in res.headers.get("content-type", "").lower()
