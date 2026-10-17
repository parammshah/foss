"""
Integration adapters for Team 1 (Data/Forecasting/Ledger) and Team 2 (Decision/Risk/Optimization).
Gracefully binds to real implementations if installed, or falls back to explicitly labeled
structural fixtures in SYNTHETIC_DEMO mode while attaching cross-module learning telemetry.
Never silently disguises synthetic fixtures as authoritative live integration.
"""

from __future__ import annotations
import importlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from reserve_juggler.contracts import (
    DecisionRecord,
    ForecastBundle,
    LearningEvent,
    RiskSignal,
    ServiceStatus,
    TreasuryState,
)
from reserve_juggler.integration.fixtures import (
    create_mock_decision_record,
    create_mock_forecast_bundle,
    create_mock_treasury_state,
)

logger = logging.getLogger(__name__)


class UnintegratedServiceError(Exception):
    """Raised when live mode is requested for an unintegrated external subsystem."""
    pass


class Team1DataForecastAdapter:
    """
    Adapter wrapper for Team 1's data ingestion, ledger, and forecasting services.
    Captures telemetry for the self-learning feedback loop without modifying Team 1's files.
    """

    def __init__(self, mode: str = "SYNTHETIC_DEMO"):
        self.mode = mode
        self._real_ledger_service = None
        self._real_forecast_service = None
        self.ledger_status = ServiceStatus.UNINTEGRATED
        self.forecast_status = ServiceStatus.UNINTEGRATED
        self.unintegrated_reason: Optional[str] = None

        self._discover_services()

    def _discover_services(self) -> None:
        """Dynamically discovers canonical Team 1 service implementations."""
        # 1. Discover Ledger
        try:
            ledger_mod = importlib.import_module("reserve_juggler.ledger")
            if hasattr(ledger_mod, "TreasuryLedger"):
                self._real_ledger_service = ledger_mod.TreasuryLedger()
                self.ledger_status = ServiceStatus.CONNECTED
                logger.info("Connected to canonical Team 1 TreasuryLedger.")
            elif hasattr(ledger_mod, "TreasuryLedgerService"):
                self._real_ledger_service = ledger_mod.TreasuryLedgerService()
                self.ledger_status = ServiceStatus.CONNECTED
        except ImportError:
            pass

        # 2. Discover Forecasting
        try:
            forecast_mod = importlib.import_module("reserve_juggler.forecasting")
            if hasattr(forecast_mod, "ForecastService"):
                self._real_forecast_service = forecast_mod.ForecastService()
                self.forecast_status = ServiceStatus.CONNECTED
                logger.info("Connected to canonical Team 1 ForecastService.")
                # Check if ForecastService also provides get_state
                if self._real_ledger_service is None and hasattr(self._real_forecast_service, "get_state"):
                    self._real_ledger_service = self._real_forecast_service
                    self.ledger_status = ServiceStatus.CONNECTED
        except ImportError:
            pass

        if self.ledger_status != ServiceStatus.CONNECTED or self.forecast_status != ServiceStatus.CONNECTED:
            missing = []
            if self.ledger_status != ServiceStatus.CONNECTED:
                missing.append("TreasuryLedger")
            if self.forecast_status != ServiceStatus.CONNECTED:
                missing.append("ForecastService")
            self.unintegrated_reason = f"Team 1 components [{', '.join(missing)}] not found in PYTHONPATH."
            if self.mode == "SYNTHETIC_DEMO":
                self.ledger_status = ServiceStatus.SYNTHETIC_DEMO
                self.forecast_status = ServiceStatus.SYNTHETIC_DEMO

    @property
    def is_connected(self) -> bool:
        return self.forecast_status == ServiceStatus.CONNECTED and self.ledger_status == ServiceStatus.CONNECTED

    def get_treasury_state(self, as_of_iso: Optional[str] = None, scenario: str = "NORMAL") -> TreasuryState:
        now_iso = as_of_iso or datetime.now(timezone.utc).isoformat()

        if self._real_ledger_service:
            if hasattr(self._real_ledger_service, "get_state"):
                state = self._real_ledger_service.get_state(now_iso)
                state.source_status = ServiceStatus.CONNECTED
                state.is_synthetic = False
                return state

        if self.mode != "SYNTHETIC_DEMO":
            raise UnintegratedServiceError(
                f"Cannot retrieve live TreasuryState: {self.unintegrated_reason or 'Ledger service unintegrated.'}"
            )

        state = create_mock_treasury_state(now_iso, scenario=scenario)
        state.source_status = ServiceStatus.SYNTHETIC_DEMO
        state.is_synthetic = True
        return state

    def get_forecast_bundle(
        self, as_of_iso: Optional[str] = None, scenario: str = "NORMAL"
    ) -> Tuple[ForecastBundle, LearningEvent]:
        now_iso = as_of_iso or datetime.now(timezone.utc).isoformat()

        if self._real_forecast_service:
            if hasattr(self._real_forecast_service, "forecast"):
                bundle = self._real_forecast_service.forecast(now_iso)
            elif hasattr(self._real_forecast_service, "predict"):
                bundle = self._real_forecast_service.predict(now_iso)
            else:
                bundle = None

            if bundle:
                bundle.source_status = ServiceStatus.CONNECTED
                bundle.is_synthetic = False
                corr_id = f"FC-{datetime.now().strftime('%Y%m%d%H%M%S%f')}-{uuid.uuid4().hex[:6]}"
                learning_event = LearningEvent(
                    event_id=f"EV-T1-{corr_id}",
                    versioned_event_type="FORECAST_INFERENCE_RECORDED",
                    module="FORECASTING",
                    timestamp=now_iso,
                    correlation_id=corr_id,
                    payload={
                        "horizons": list(bundle.point_forecasts.keys()),
                        "quality_status": bundle.quality_status,
                        "point_forecasts": {
                            h: {
                                "expected_net_flow": f.expected_net_flow,
                                "expected_inflow": f.expected_inflow,
                                "expected_outflow": f.expected_outflow,
                                "uncertainty_p05": f.uncertainty_p05,
                                "uncertainty_p95": f.uncertainty_p95,
                            }
                            for h, f in bundle.point_forecasts.items()
                        },
                        "confidence_avg": sum(f.confidence_score for f in bundle.point_forecasts.values()) / len(bundle.point_forecasts) if bundle.point_forecasts else 0.0,
                    },
                    associated_versions=bundle.model_versions,
                )
                return (bundle, learning_event)

        if self.mode != "SYNTHETIC_DEMO":
            raise UnintegratedServiceError(
                f"Cannot retrieve live ForecastBundle: {self.unintegrated_reason or 'Forecast service unintegrated.'}"
            )

        bundle = create_mock_forecast_bundle(now_iso, scenario=scenario)
        bundle.source_status = ServiceStatus.SYNTHETIC_DEMO
        bundle.is_synthetic = True

        corr_id = f"FC-{datetime.now().strftime('%Y%m%d%H%M%S%f')}-{uuid.uuid4().hex[:6]}"
        learning_event = LearningEvent(
            event_id=f"EV-T1-{corr_id}",
            versioned_event_type="FORECAST_INFERENCE_RECORDED",
            module="FORECASTING",
            timestamp=now_iso,
            correlation_id=corr_id,
            payload={
                "horizons": list(bundle.point_forecasts.keys()),
                "quality_status": bundle.quality_status,
                "point_forecasts": {
                    h: {
                        "expected_net_flow": f.expected_net_flow,
                        "expected_inflow": f.expected_inflow,
                        "expected_outflow": f.expected_outflow,
                        "uncertainty_p05": f.uncertainty_p05,
                        "uncertainty_p95": f.uncertainty_p95,
                    }
                    for h, f in bundle.point_forecasts.items()
                },
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

    def __init__(self, mode: str = "SYNTHETIC_DEMO"):
        self.mode = mode
        self._real_decision_service = None
        self.decision_status = ServiceStatus.UNINTEGRATED
        self.unintegrated_reason: Optional[str] = None

        self._discover_services()

    def _discover_services(self) -> None:
        """Dynamically discovers canonical Team 2 service implementations."""
        try:
            decision_mod = importlib.import_module("reserve_juggler.decision")
            if hasattr(decision_mod, "TreasuryDecisionService"):
                self._real_decision_service = decision_mod.TreasuryDecisionService()
                self.decision_status = ServiceStatus.CONNECTED
                logger.info("Connected to canonical Team 2 TreasuryDecisionService.")
            elif hasattr(decision_mod, "DecisionService"):
                self._real_decision_service = decision_mod.DecisionService()
                self.decision_status = ServiceStatus.CONNECTED
                logger.info("Connected to canonical Team 2 DecisionService.")
        except ImportError:
            self.unintegrated_reason = "Team 2 module 'reserve_juggler.decision' not found in PYTHONPATH."
            if self.mode == "SYNTHETIC_DEMO":
                self.decision_status = ServiceStatus.SYNTHETIC_DEMO

    @property
    def is_connected(self) -> bool:
        return self.decision_status == ServiceStatus.CONNECTED

    def compute_next_best_action(
        self,
        treasury_state: TreasuryState,
        forecast_bundle: ForecastBundle,
        risk_signals: List[RiskSignal],
        as_of_iso: Optional[str] = None,
        scenario: str = "NORMAL",
    ) -> Tuple[DecisionRecord, LearningEvent]:
        now_iso = as_of_iso or datetime.now(timezone.utc).isoformat()

        if self._real_decision_service and hasattr(self._real_decision_service, "decide"):
            record = self._real_decision_service.decide(treasury_state, forecast_bundle, risk_signals)
            record.source_status = ServiceStatus.CONNECTED
            record.is_synthetic = False
        else:
            if self.mode != "SYNTHETIC_DEMO":
                raise UnintegratedServiceError(
                    f"Cannot compute live Next Best Action: {self.unintegrated_reason or 'Decision service unintegrated.'}"
                )
            record = create_mock_decision_record(now_iso, scenario=scenario)
            record.source_status = ServiceStatus.SYNTHETIC_DEMO
            record.is_synthetic = True

        # Emit LearningEvent for Team 2 telemetry
        learning_event = LearningEvent(
            event_id=f"EV-T2-{record.decision_id}-{uuid.uuid4().hex[:6]}",
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
                "gate_outcomes": [
                    {"gate_name": g.gate_name, "status": g.status.value, "reason": g.reason}
                    for g in record.hard_gate_outcomes
                ],
                "tss_score": record.treasury_safety_score,
                "reason_codes": record.reason_codes,
            },
            associated_versions=record.model_versions,
        )

        return (record, learning_event)
