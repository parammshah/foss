"""
Integration adapters for Team 1 (Data/Forecasting/Ledger) and Team 2 (Decision/Risk/Optimization).
Gracefully binds to real implementations if installed, or falls back to local structural fixtures
while attaching cross-module learning telemetry.
"""

from __future__ import annotations
import importlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from reserve_juggler.integration.contracts import (
    DecisionRecord,
    ForecastBundle,
    LearningEvent,
    RiskSignal,
    TreasuryState,
)
from reserve_juggler.integration.fixtures import (
    create_mock_decision_record,
    create_mock_forecast_bundle,
    create_mock_treasury_state,
)

logger = logging.getLogger(__name__)


class Team1DataForecastAdapter:
    """
    Adapter wrapper for Team 1's data ingestion, ledger, and forecasting services.
    Captures telemetry for the self-learning feedback loop without modifying Team 1's files.
    """

    def __init__(self):
        self._real_service = None
        # Attempt dynamic discovery
        try:
            mod = importlib.import_module("reserve_juggler.forecasting")
            if hasattr(mod, "ForecastService"):
                self._real_service = mod.ForecastService()
                logger.info("Connected to canonical Team 1 ForecastService.")
        except ImportError:
            logger.info("Team 1 modules not installed; using local integration adapter.")

    def get_treasury_state(self, as_of_iso: Optional[str] = None, scenario: str = "NORMAL") -> TreasuryState:
        if self._real_service and hasattr(self._real_service, "get_state"):
            return self._real_service.get_state(as_of_iso)
        return create_mock_treasury_state(as_of_iso, scenario=scenario)

    def get_forecast_bundle(
        self, as_of_iso: Optional[str] = None, scenario: str = "NORMAL"
    ) -> Tuple[ForecastBundle, LearningEvent]:
        now_iso = as_of_iso or datetime.now(timezone.utc).isoformat()
        
        if self._real_service and hasattr(self._real_service, "predict"):
            bundle = self._real_service.predict(as_of_iso)
        else:
            bundle = create_mock_forecast_bundle(now_iso, scenario=scenario)

        # Emit LearningEvent for Team 1 telemetry
        corr_id = f"FC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        learning_event = LearningEvent(
            event_id=f"EV-T1-{corr_id}",
            versioned_event_type="FORECAST_INFERENCE_RECORDED",
            module="FORECASTING",
            timestamp=now_iso,
            correlation_id=corr_id,
            payload={
                "horizons": list(bundle.point_forecasts.keys()),
                "quality_status": bundle.quality_status,
                "net_flows": {h: f.expected_net_flow for h, f in bundle.point_forecasts.items()},
                "confidence_avg": sum(f.confidence_score for f in bundle.point_forecasts.values()) / len(bundle.point_forecasts) if bundle.point_forecasts else 0.0,
            },
            associated_versions=bundle.model_versions,
        )

        return (bundle, learning_event)


class Team2DecisionRiskAdapter:
    """
    Adapter wrapper for Team 2's risk scoring, safety constraints, and decision optimization.
    Captures decision rationale and gates for the self-learning feedback loop.
    """

    def __init__(self):
        self._real_service = None
        try:
            mod = importlib.import_module("reserve_juggler.decision")
            if hasattr(mod, "DecisionService"):
                self._real_service = mod.DecisionService()
                logger.info("Connected to canonical Team 2 DecisionService.")
        except ImportError:
            logger.info("Team 2 modules not installed; using local integration adapter.")

    def compute_next_best_action(
        self,
        treasury_state: TreasuryState,
        forecast_bundle: ForecastBundle,
        risk_signals: List[RiskSignal],
        as_of_iso: Optional[str] = None,
        scenario: str = "NORMAL",
    ) -> Tuple[DecisionRecord, LearningEvent]:
        now_iso = as_of_iso or datetime.now(timezone.utc).isoformat()

        if self._real_service and hasattr(self._real_service, "decide"):
            record = self._real_service.decide(treasury_state, forecast_bundle, risk_signals)
        else:
            record = create_mock_decision_record(now_iso, scenario=scenario)

        # Emit LearningEvent for Team 2 telemetry
        learning_event = LearningEvent(
            event_id=f"EV-T2-{record.decision_id}",
            versioned_event_type="DECISION_GENERATED",
            module="DECISION_OPTIMIZER",
            timestamp=now_iso,
            correlation_id=record.decision_id,
            payload={
                "decision_id": record.decision_id,
                "selected_action": record.selected_action,
                "target_asset": record.target_asset,
                "amount": record.amount,
                "estimated_costs": record.estimated_costs,
                "hard_gates_passed": all(g.passed for g in record.hard_gate_outcomes),
                "tss_score": record.treasury_safety_score,
                "reason_codes": record.reason_codes,
            },
            associated_versions=record.model_versions,
        )

        return (record, learning_event)
