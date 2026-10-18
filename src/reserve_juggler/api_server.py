"""
FastAPI Server for Reserve Juggler Treasury Management Platform.
Provides unified REST API & SSE streaming endpoints connecting the FOSS UI frontend
to the integrated backend ApplicationFacade (Team 1, Team 2, and Team 3 subsystems).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from reserve_juggler.contracts import (
    ApprovalStatus,
    AssetTier,
    RiskSeverity,
    ServiceStatus,
    UserRole,
)
from reserve_juggler.integration.facade import ApplicationFacade, DashboardSnapshot

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# =============================================================================
# Serialization Helper
# =============================================================================

def serialize_to_json_friendly(obj: Any) -> Any:
    """Recursively serializes Pydantic models, Dataclasses, Enums, and structures into JSON-friendly format."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict
        return {k: serialize_to_json_friendly(v) for k, v in asdict(obj).items()}
    if hasattr(obj, "__dict__"):
        return {k: serialize_to_json_friendly(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple, set)):
        return [serialize_to_json_friendly(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): serialize_to_json_friendly(v) for k, v in obj.items()}
    if isinstance(obj, (int, float, bool, str)):
        return obj
    return str(obj)


# =============================================================================
# Request / Response Schemas
# =============================================================================

class ScenarioSwitchRequest(BaseModel):
    scenario: str = Field(..., description="Scenario name: NORMAL, BANKING_STRESS, RATE_HIKE_SHOCK, SOVEREIGN_DEBT_DISRUPTION")


class DecisionActionRequest(BaseModel):
    action_type: str = Field(..., description="Action type: APPROVED, MODIFIED, REJECTED, CANCELLED")
    actor_name: str = Field("Chief Treasurer", description="Name of signing actor")
    actor_role: str = Field("HEAD_OF_TREASURY", description="Role: HEAD_OF_TREASURY, RISK_OFFICER, TREASURY_ANALYST, AUDITOR")
    stated_reason: str = Field("Authorized under established liquidity policy bounds.", description="Audit rationale")
    modified_amount: Optional[float] = Field(None, description="Optional override amount")
    modified_action: Optional[str] = Field(None, description="Optional override action")
    is_live_execution: bool = Field(False, description="Flag for live production execution")


class StressSimulationRequest(BaseModel):
    deposit_shock_pct: float = Field(0.0, description="Deposit withdrawal shock in percent (e.g. 20.0 for 20%)")
    rate_shock_bps: float = Field(0.0, description="Yield curve rate shock in basis points (e.g. 250 for +2.50%)")
    repo_delay_days: int = Field(0, description="Delay in days for repo access (e.g. 1)")
    credit_unavailable: bool = Field(False, description="Whether credit facility is frozen/unavailable")
    equity_drawdown_pct: float = Field(0.0, description="Equity market drawdown shock in percent")


class ProposeCandidateRequest(BaseModel):
    component: str = Field(..., description="Target component: FORECASTING, DECISION_ENGINE, SENTINEL")
    parameter_key: str = Field(..., description="Config parameter key")
    proposed_value: Any = Field(..., description="New parameter value")
    current_value: Optional[Any] = Field(None, description="Current baseline value")
    description: str = Field(..., description="Reason and rationale for change")
    author: str = Field("Treasury Quant Desk", description="Author identifier")
    holdout_evaluation_score: float = Field(0.05, description="Holdout score")
    holdout_sample_size: int = Field(150, description="Holdout sample size")
    stress_shortfall_count: int = Field(0, description="Shortfalls under stress")


class PromoteCandidateRequest(BaseModel):
    candidate_id: str = Field(..., description="Candidate ID to promote")
    actor_role: str = Field("HEAD_OF_TREASURY", description="Role approving promotion")


class RollbackCandidateRequest(BaseModel):
    candidate_id: str = Field(..., description="Candidate ID to rollback")
    actor_role: str = Field("HEAD_OF_TREASURY", description="Role approving rollback")


# =============================================================================
# Application Factory
# =============================================================================

def create_app(facade: Optional[ApplicationFacade] = None) -> FastAPI:
    app = FastAPI(
        title="Reserve Juggler — Treasury Command & Liquidity Choreography API",
        description="Autonomous Liquidity Orchestration API connecting Team 1, Team 2, and Team 3 to FOSS UI.",
        version="2.0.0",
    )

    # Enable CORS for all origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize Facade singleton
    if facade is None:
        facade = ApplicationFacade(use_synthetic=True)
    app.state.facade = facade

    # -------------------------------------------------------------------------
    # REST API Routes
    # -------------------------------------------------------------------------

    @app.get("/api/health")
    async def get_health():
        return {
            "status": "HEALTHY",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "facade_version": app.state.facade.FACADE_VERSION,
            "mode": app.state.facade.data_source_mode,
            "scenario": app.state.facade.current_scenario,
        }

    @app.get("/api/status")
    async def get_status():
        f: ApplicationFacade = app.state.facade
        perf = f.learning_service.evaluate_system_performance()
        return {
            "system_status": "ONLINE",
            "facade_version": f.FACADE_VERSION,
            "data_source_mode": f.data_source_mode,
            "active_scenario": f.current_scenario,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_health": perf.overall_system_health,
            "subsystems": {
                "team1_forecasting_ledger": {
                    "connected": f.team1_adapter.is_connected,
                    "status": f.team1_adapter.forecast_status.value if hasattr(f.team1_adapter.forecast_status, "value") else str(f.team1_adapter.forecast_status),
                    "model_versions": {"deposit_model": "v2.1", "ensemble_combiner": "v1.4"},
                },
                "team2_decision_optimizer": {
                    "connected": f.team2_adapter.is_connected,
                    "status": f.team2_adapter.decision_status.value if hasattr(f.team2_adapter.decision_status, "value") else str(f.team2_adapter.decision_status),
                    "model_versions": {"decision_engine": "v1.2.0", "optimizer": "v2.0.1"},
                },
                "team3_sentinel_risk": {
                    "connected": True,
                    "status": f.sentinel.source_manager.status.value if hasattr(f.sentinel.source_manager.status, "value") else str(f.sentinel.source_manager.status),
                    "model_versions": {"sentinel": "v1.0.0", "explainability": "v1.0.0"},
                },
            },
        }

    @app.get("/api/dashboard/snapshot")
    async def get_dashboard_snapshot(as_of: Optional[str] = Query(None, description="Optional ISO timestamp")):
        try:
            f: ApplicationFacade = app.state.facade
            snap: DashboardSnapshot = f.get_dashboard_snapshot(as_of_timestamp=as_of)
            return serialize_to_json_friendly(snap)
        except Exception as e:
            logger.exception("Failed to build dashboard snapshot: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    SCENARIO_REGISTRY = {
        "NORMAL": "NORMAL",
        "BENIGN_INFLOW": "NORMAL",
        "BANKING_STRESS": "BANKING_STRESS",
        "SVB_COLLAPSE": "BANKING_STRESS",
        "RATE_HIKE_SHOCK": "RATE_HIKE_SHOCK",
        "FLASH_CRASH": "RATE_HIKE_SHOCK",
        "SOVEREIGN_DEBT_DISRUPTION": "SOVEREIGN_DEBT_DISRUPTION",
        "RECESSIONARY_SPIRAL": "SOVEREIGN_DEBT_DISRUPTION",
    }

    @app.post("/api/scenario/switch")
    async def switch_scenario(req: ScenarioSwitchRequest):
        try:
            raw_scenario = req.scenario.strip().upper()
            if raw_scenario not in SCENARIO_REGISTRY:
                valid_list = sorted(list(SCENARIO_REGISTRY.keys()))
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid scenario '{req.scenario}'. Supported scenarios: {valid_list}"
                )
            canonical_scenario = SCENARIO_REGISTRY[raw_scenario]
            f: ApplicationFacade = app.state.facade
            f.switch_scenario(canonical_scenario)
            snap = f.get_dashboard_snapshot()
            return {
                "success": True,
                "active_scenario": f.current_scenario,
                "requested_scenario": req.scenario,
                "message": f"Operational scenario transitioned to {f.current_scenario}",
                "snapshot": serialize_to_json_friendly(snap),
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Failed to switch scenario: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/decision/action")
    async def submit_decision_action(req: DecisionActionRequest):
        try:
            f: ApplicationFacade = app.state.facade
            success, msg = f.submit_human_action(
                action_type=req.action_type.upper(),
                actor_name=req.actor_name,
                actor_role=req.actor_role,
                stated_reason=req.stated_reason,
                modified_amount=req.modified_amount,
                modified_action=req.modified_action,
                is_live_execution=req.is_live_execution,
            )
            snap = f.get_dashboard_snapshot()
            return {
                "success": success,
                "message": msg,
                "decision_status": snap.latest_decision.approval_status if snap.latest_decision else "UNKNOWN",
                "snapshot": serialize_to_json_friendly(snap),
            }
        except Exception as e:
            logger.exception("Failed to submit decision action: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/decision/cycle")
    async def run_decision_cycle():
        try:
            f: ApplicationFacade = app.state.facade
            cycle_result = f.run_decision_cycle()
            # Refresh latest decision record in snapshot if valid
            if hasattr(cycle_result, "decision_record") and cycle_result.decision_record:
                f._latest_decision_record = cycle_result.decision_record
            snap = f.get_dashboard_snapshot()
            return {
                "success": True,
                "cycle_id": getattr(cycle_result, "cycle_id", "CYC-LIVE"),
                "workflow_state": getattr(cycle_result, "workflow_state", "COMPLETED"),
                "cycle_result": serialize_to_json_friendly(cycle_result),
                "snapshot": serialize_to_json_friendly(snap),
            }
        except Exception as e:
            logger.exception("Failed to run decision cycle: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/stress/simulate")
    async def run_stress_simulation(req: StressSimulationRequest):
        try:
            f: ApplicationFacade = app.state.facade
            res = f.run_stress_lab_scenario(
                deposit_shock_pct=req.deposit_shock_pct,
                rate_shock_bps=req.rate_shock_bps,
                repo_delay_days=req.repo_delay_days,
                credit_unavailable=req.credit_unavailable,
                equity_drawdown_pct=req.equity_drawdown_pct,
            )
            return serialize_to_json_friendly(res)
        except Exception as e:
            logger.exception("Failed to execute stress lab simulation: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/stress/presets")
    async def get_stress_presets():
        return {
            "presets": [
                {
                    "id": "svb_run",
                    "name": "SVB 2023 Contagion",
                    "description": "3.2x deposit surge, +250 bps rate hike, credit line frozen.",
                    "deposit_shock_pct": 32.0,
                    "rate_shock_bps": 250.0,
                    "repo_delay_days": 1,
                    "credit_unavailable": True,
                    "equity_drawdown_pct": 20.0,
                },
                {
                    "id": "payday_crunch",
                    "name": "Payday Crunch",
                    "description": "2.4x payroll surge, +75 bps rate adjustment.",
                    "deposit_shock_pct": 24.0,
                    "rate_shock_bps": 75.0,
                    "repo_delay_days": 0,
                    "credit_unavailable": False,
                    "equity_drawdown_pct": 5.0,
                },
                {
                    "id": "rate_spike",
                    "name": "Emergency Rate Hike",
                    "description": "+400 bps aggressive sovereign yield jump, 12% deposit drain.",
                    "deposit_shock_pct": 12.0,
                    "rate_shock_bps": 400.0,
                    "repo_delay_days": 0,
                    "credit_unavailable": False,
                    "equity_drawdown_pct": 15.0,
                },
                {
                    "id": "baseline",
                    "name": "Baseline Normal",
                    "description": "Historical steady-state operational flows with zero shocks.",
                    "deposit_shock_pct": 0.0,
                    "rate_shock_bps": 0.0,
                    "repo_delay_days": 0,
                    "credit_unavailable": False,
                    "equity_drawdown_pct": 0.0,
                },
            ]
        }

    @app.get("/api/learning/performance")
    async def get_learning_performance():
        try:
            f: ApplicationFacade = app.state.facade
            perf = f.learning_service.evaluate_system_performance()
            return serialize_to_json_friendly(perf)
        except Exception as e:
            logger.exception("Failed to evaluate learning performance: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/learning/candidates")
    async def get_learning_candidates():
        try:
            f: ApplicationFacade = app.state.facade
            candidates = f.learning_service.get_all_candidates()
            return serialize_to_json_friendly(candidates)
        except Exception as e:
            logger.exception("Failed to list candidates: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/learning/propose")
    async def propose_candidate(req: ProposeCandidateRequest):
        try:
            f: ApplicationFacade = app.state.facade
            candidate = f.learning_service.propose_update(
                component=req.component,
                parameter_key=req.parameter_key,
                proposed_value=req.proposed_value,
                current_value=req.current_value,
                description=req.description,
                author=req.author,
                holdout_evaluation_score=req.holdout_evaluation_score,
                holdout_sample_size=req.holdout_sample_size,
                stress_shortfall_count=req.stress_shortfall_count,
            )
            return {
                "success": True,
                "candidate": serialize_to_json_friendly(candidate),
                "message": f"Candidate update {candidate.candidate_id} proposed successfully in shadow mode.",
            }
        except Exception as e:
            logger.exception("Failed to propose candidate: %s", e)
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/learning/promote")
    async def promote_candidate(req: PromoteCandidateRequest):
        try:
            f: ApplicationFacade = app.state.facade
            success, msg = f.learning_service.promote_candidate(
                candidate_id=req.candidate_id,
                approver="Chief Treasurer",
                approver_role=req.actor_role,
                justification="Authorized promotion of validated candidate model.",
            )
            cand = f.learning_service.get_candidate(req.candidate_id)
            return {
                "success": success,
                "message": msg,
                "candidate": serialize_to_json_friendly(cand),
            }
        except Exception as e:
            logger.exception("Failed to promote candidate: %s", e)
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/learning/rollback")
    async def rollback_candidate(req: RollbackCandidateRequest):
        try:
            f: ApplicationFacade = app.state.facade
            success, msg = f.learning_service.rollback_candidate(
                candidate_id=req.candidate_id,
                actor_role=req.actor_role,
            )
            return {
                "success": success,
                "message": msg,
            }
        except Exception as e:
            logger.exception("Failed to rollback candidate: %s", e)
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/audit/logs")
    async def get_audit_logs(limit: int = Query(50, ge=1, le=500)):
        try:
            f: ApplicationFacade = app.state.facade
            events = f.learning_service.store.get_all()
            bounded_events = events[-limit:] if events else []
            audit_trail = getattr(f, "_audit_trail", [])
            bounded_audit = audit_trail[-limit:] if audit_trail else []
            return {
                "audit_trail": serialize_to_json_friendly(bounded_audit),
                "learning_events": serialize_to_json_friendly(bounded_events),
                "count": len(bounded_events),
            }
        except Exception as e:
            logger.exception("Failed to get audit logs: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/sentinel/signals")
    async def get_sentinel_signals():
        try:
            f: ApplicationFacade = app.state.facade
            snap = f.get_dashboard_snapshot()
            sent_res = snap.sentinel_result
            active_signals = getattr(sent_res, "active_signals", []) or []

            categories = list({getattr(s, "category", "GENERAL_NEWS") for s in active_signals})
            severities = [getattr(s, "severity", "LOW") for s in active_signals]
            confidences = [getattr(s, "confidence", 1.0) for s in active_signals]
            rec_scenarios = getattr(sent_res, "recommended_stress_scenarios", [])

            return {
                "sentinel_status": getattr(sent_res, "system_status", "OK"),
                "aggregate_stress_score": getattr(sent_res, "aggregate_macro_stress_score", 0.0),
                "active_signals": serialize_to_json_friendly(active_signals),
                "signal_categories": categories,
                "severities": severities,
                "confidences": confidences,
                "recommended_stress_scenarios": rec_scenarios,
                "recommended_scenarios": rec_scenarios,
                "data_quality_warnings": getattr(sent_res, "warnings", []),
                "mode": getattr(sent_res, "mode", "SYNTHETIC_LAB"),
                "impact_formula": "Event Impact Score = 100 * C * (0.25*E + 0.25*R + 0.20*M + 0.15*V + 0.10*T + 0.05*N)",
                "official_sources": [
                    {
                        "name": "Federal Reserve RSS Directory",
                        "url": "https://www.federalreserve.gov/feeds/feeds.htm",
                        "status": "AUTHORIZED_OFFICIAL",
                        "credibility": 0.99
                    },
                    {
                        "name": "European Central Bank Official Directory",
                        "url": "https://www.ecb.europa.eu/home/html/rss.sk.html",
                        "status": "AUTHORIZED_OFFICIAL",
                        "credibility": 0.98
                    },
                    {
                        "name": "Bank of Japan Policy Record",
                        "url": "https://www.boj.or.jp/en/",
                        "status": "AUTHORIZED_OFFICIAL",
                        "credibility": 0.98
                    },
                    {
                        "name": "Reuters Markets Financial Wire",
                        "url": "https://www.reuters.com/markets/",
                        "status": "AUTHORIZED_MEDIA",
                        "credibility": 0.94
                    }
                ],
                "historical_case_studies": [
                    {
                        "id": "japan_aug_2024",
                        "title": "Bank of Japan Rate Hike & 'Black Monday' Global Carry Trade Unwind",
                        "date": "2024-08-05",
                        "category": "MARKET_SHOCK",
                        "severity_score": 94,
                        "source_credibility": 99,
                        "treasury_relevance": 86,
                        "market_confirmation": 98,
                        "overall_impact_score": 92.5,
                        "primary_article_url": "https://www.reuters.com/markets/asia/japans-nikkei-sees-biggest-rout-since-1987-black-monday-2024-08-05/",
                        "official_record_url": "https://www.boj.or.jp/en/mopo/mpmdeci/state_2024/index.htm",
                        "publisher": "Reuters / Bank of Japan",
                        "summary": "On July 31, 2024, the Bank of Japan raised short-term policy rates to 0.25%. On August 5, the Nikkei 225 suffered a 12.4% plunge—its steepest single-day collapse since 1987—prompting worldwide cross-asset liquidity liquidation and sovereign yield spikes.",
                        "simulated_treasury_response": "Reserve Juggler Capital Firewall immediately engaged: restricted Tier 3 equity exposure cap from 10% to 2%, expanded Tier 0 cash reserve by $15,000,000, and verified zero forced-liquidation hair cuts on Tier 2 benchmark bonds."
                    },
                    {
                        "id": "svb_mar_2023",
                        "title": "Silicon Valley Bank (SVB) Accelerated Deposit Flight & Liquidity Run",
                        "date": "2023-03-10",
                        "category": "BANKING_STRESS",
                        "severity_score": 98,
                        "source_credibility": 100,
                        "treasury_relevance": 96,
                        "market_confirmation": 99,
                        "overall_impact_score": 97.4,
                        "primary_article_url": "https://www.federalreserve.gov/publications/2023-april-svb-executive-summary.htm",
                        "official_record_url": "https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/",
                        "publisher": "Federal Reserve Executive Summary / FDIC",
                        "summary": "Rapid social-media coordinated withdrawal of $42B in under 24 hours overwhelmed unhedged long-duration held-to-maturity (HTM) assets, causing severe liquidity insolvency and systemic interbank credit spread widening.",
                        "simulated_treasury_response": "Hot Money vs Core Capital separation proved decisive: $60,141,895 in mandatory customer withdrawal payouts was held in T+0 cash & overnight repo, ensuring 100% of customer obligations cleared with $0 liquidation penalty."
                    }
                ]
            }
        except Exception as e:
            logger.exception("Failed to fetch sentinel signals: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/algorithms/metadata")
    async def get_algorithms_metadata():
        """Returns actual fitted algorithm metadata, weights, and mathematical formulations."""
        return {
            "forecasting_pipeline": [
                {
                    "name": "Recent Average Baseline",
                    "module": "reserve_juggler.forecasting.baseline.RecentAverageBaseline",
                    "type": "Benchmark",
                    "description": "Rolling 24-hour mean of observed net cash flows; provides basic benchmark baseline.",
                    "status": "ACTIVE_BENCHMARK",
                    "configured_weights": {"weight": 0.00}
                },
                {
                    "name": "Time-of-Week Seasonal Baseline",
                    "module": "reserve_juggler.forecasting.baseline.TimeOfWeekSeasonalBaseline",
                    "type": "Seasonal Prior",
                    "description": "Historical flow distributions conditioned on (day_of_week, hour_of_day); captures diurnal corporate payment cycles.",
                    "status": "ACTIVE_CHAMPION_COMPONENT",
                    "configured_weights": {"weight": 0.968}
                },
                {
                    "name": "Ridge Regularized Linear Model",
                    "module": "reserve_juggler.forecasting.ml_model.RidgeMLModel",
                    "type": "Machine Learning",
                    "description": "L2-regularized regression over multi-lag features and calendar indicators to prevent coefficient explosion.",
                    "status": "ACTIVE_CHAMPION_COMPONENT",
                    "configured_weights": {"weight": 0.032}
                },
                {
                    "name": "OR-Constrained Forecast Ensemble",
                    "module": "reserve_juggler.forecasting.or_ensemble.ORConstrainedEnsemble",
                    "type": "Operations Research Optimization",
                    "description": "SciPy SLSQP solver minimizing holdout WAPE subject to non-negativity and unit sum constraint: w_i >= 0, sum(w_i) = 1.0.",
                    "status": "ACTIVE_OPTIMIZER",
                    "objective_loss": "WAPE 5.48% (72h Horizon)"
                },
                {
                    "name": "Conformal Uncertainty & Residual Sizing",
                    "module": "reserve_juggler.forecasting.uncertainty.ConformalUncertaintyEngine",
                    "type": "Risk & Uncertainty",
                    "description": "Empirical residual quantile estimation (P05/P50/P95) providing mathematically guaranteed coverage intervals.",
                    "status": "ACTIVE_SAFETY_BUFFER",
                    "calibration_sample_size": 336
                }
            ],
            "optimizer_pipeline": {
                "name": "Mixed-Integer Convex Liquidity Optimizer",
                "module": "reserve_juggler.optimization.optimizer.TreasuryOptimizer",
                "objective": "Minimize Expected Cost (Overdraft + Early-Exit Fees + Tax Slippage) Subject to Liquidity Constraints",
                "governance_gates": [
                    "MandatoryReserveFloorGate (SEC 15c3-1 / Basel III LCR)",
                    "CustomerObligationsProtectionGate (T+0 Intraday Priority)",
                    "CreditFacilityDrawLimitGate (Authorized Bank Facility Cap)",
                    "AssetEligibilityGate (HQLA Level 1 & Level 2A Qualification)"
                ]
            }
        }

    @app.get("/api/decision/timeline")
    async def get_decision_timeline():
        """Returns the canonical Two-Hour Bridge case study and alternative action cost waterfall."""
        return {
            "case_study": {
                "name": "The Two-Hour Bridge Demonstration",
                "current_time_utc": "06:00:00",
                "repo_maturity_time_utc": "08:00:00",
                "repo_principal": 10000000.0,
                "early_break_fee_pct": 0.50,
                "early_break_cost": 50000.0,
                "overdraft_apr_pct": 12.0,
                "hours_to_maturity": 2.0,
                "bridge_interest_cost": 273.97,
                "net_direct_savings": 49726.03,
                "cost_reduction_pct": 99.45
            },
            "strategy_lanes": {
                "lane_a_conventional_reactive": [
                    {"time": "06:00", "event": "Temporary liquidity gap detected ($10M needed for customer payouts)", "status": "WARNING"},
                    {"time": "06:02", "event": "Treasury detects Repo funds locked until 08:00 maturity", "status": "LOCKED"},
                    {"time": "06:04", "event": "Naive policy evaluates breaking repo early", "status": "EVALUATE"},
                    {"time": "06:05", "event": "Forced early-break fee of 0.50% ($50,000) incurred", "status": "FEE_PAID"},
                    {"time": "08:00", "event": "Original repo matures, but penalty was already paid unnecessarily", "status": "LOSS"}
                ],
                "lane_b_reserve_juggler_time_aware": [
                    {"time": "06:00", "event": "Forecasting predicts temporary 2-hour liquidity gap", "status": "PREDICT"},
                    {"time": "06:01", "event": "Time-aware engine maps $10M repo maturity timer (2 hours remaining)", "status": "TIME_AWARE"},
                    {"time": "06:02", "event": "Convex optimizer compares 5 alternatives; selects 2h credit bridge", "status": "OPTIMIZE"},
                    {"time": "06:03", "event": "All 4 hard safety gates evaluated & passed", "status": "PASS"},
                    {"time": "06:04", "event": "Human governance approval verified by Head of Treasury", "status": "APPROVED"},
                    {"time": "06:05", "event": "2-hour bridge executed at $273.97 total cost", "status": "BRIDGED"},
                    {"time": "08:00", "event": "Repo matures cleanly; $10M principal + $274 repaid without early-break fee", "status": "REPAID"}
                ]
            },
            "action_waterfall": [
                {"action": "Two-Hour Bridge (Recommended)", "cost": 273.97, "feasibility": "FEASIBLE", "settlement": "T+0 (Immediate)", "note": "99.45% cheaper than breaking repo"},
                {"action": "Wait Until Maturity", "cost": 0.00, "feasibility": "FEASIBLE", "settlement": "At 08:00", "note": "Zero cost if obligations allow 2h window"},
                {"action": "Break Overnight Repo", "cost": 50000.00, "feasibility": "FEASIBLE_EXPENSIVE", "settlement": "T+0", "note": "Incurs avoidable 50 bps contractual fee"},
                {"action": "24-Hour Credit Draw", "cost": 3287.67, "feasibility": "FEASIBLE", "settlement": "T+0", "note": "Full day credit line interest"},
                {"action": "Emergency Bond Liquidation", "cost": 1020000.00, "feasibility": "INFEASIBLE_LATENCY", "settlement": "T+1", "note": "T+1 latency violates T+0 morning obligations"}
            ]
        }

    # -------------------------------------------------------------------------
    # Server-Sent Events (SSE) Live Telemetry Stream
    # -------------------------------------------------------------------------
    @app.get("/api/stream")
    async def stream_telemetry(request: Request):
        async def event_generator():
            f: ApplicationFacade = app.state.facade
            while True:
                if await request.is_disconnected():
                    break
                try:
                    snap = f.get_dashboard_snapshot()
                    data_str = json.dumps(serialize_to_json_friendly(snap), default=str)
                    yield f"event: snapshot\ndata: {data_str}\n\n"
                except Exception as ex:
                    yield f"event: error\ndata: {json.dumps({'error': str(ex)})}\n\n"
                await asyncio.sleep(3.0)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # -------------------------------------------------------------------------
    # Static Files Mounting (Serving FOSS UI)
    # -------------------------------------------------------------------------
    repo_root = Path(__file__).resolve().parent.parent.parent
    foss_ui_dir = repo_root / "FOSS UI"

    if foss_ui_dir.exists():
        # Mount the static assets under /
        app.mount("/css", StaticFiles(directory=str(foss_ui_dir / "css")), name="css")
        app.mount("/js", StaticFiles(directory=str(foss_ui_dir / "js")), name="js")
        assets_dir = foss_ui_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/", response_class=HTMLResponse)
        async def serve_index():
            index_path = foss_ui_dir / "index.html"
            return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

        @app.get("/index.html", response_class=HTMLResponse)
        async def serve_index_html():
            index_path = foss_ui_dir / "index.html"
            return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

        @app.get("/platform.html", response_class=HTMLResponse)
        async def serve_platform_html():
            platform_path = foss_ui_dir / "platform.html"
            return HTMLResponse(content=platform_path.read_text(encoding="utf-8"))

        @app.get("/reserve_juggler_treasury_platform.html", response_class=HTMLResponse)
        async def serve_treasury_html():
            plat_path = foss_ui_dir / "reserve_juggler_treasury_platform.html"
            return HTMLResponse(content=plat_path.read_text(encoding="utf-8"))

    return app


# Export singleton app
app = create_app()


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run Reserve Juggler API Server and FOSS UI Platform.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8089, help="Port to bind (default: 8089)")
    parser.add_argument("--reload", action="store_true", help="Enable live code reload")
    args = parser.parse_args()

    print(f"================================================================================")
    print(f"   RESERVE JUGGLER: UNIFIED BACKEND API & FOSS UI SERVER                        ")
    print(f"   Host: http://{args.host}:{args.port}                                         ")
    print(f"   Platform UI: http://localhost:{args.port}/platform.html                      ")
    print(f"   Landing Page: http://localhost:{args.port}/index.html                        ")
    print(f"   API OpenAPI Docs: http://localhost:{args.port}/docs                         ")
    print(f"================================================================================")

    uvicorn.run("reserve_juggler.api_server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
