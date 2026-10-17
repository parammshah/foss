"""
Integration Package for Reserve Juggler.
"""

from reserve_juggler.integration.facade import ApplicationFacade, DashboardSnapshot
from reserve_juggler.integration.contracts import (
    TreasuryState,
    AssetPosition,
    Obligation,
    ForecastBundle,
    HorizonForecast,
    RiskSignal,
    DecisionRecord,
    HardGateOutcome,
    AlternativeAction,
    ExecutionOutcome,
    LearningEvent,
    UserRole,
    ApprovalStatus,
)
from reserve_juggler.integration.adapters import Team1DataForecastAdapter, Team2DecisionRiskAdapter
from reserve_juggler.integration.fixtures import (
    create_mock_treasury_state,
    create_mock_forecast_bundle,
    create_mock_decision_record,
)

__all__ = [
    "ApplicationFacade",
    "DashboardSnapshot",
    "TreasuryState",
    "AssetPosition",
    "Obligation",
    "ForecastBundle",
    "HorizonForecast",
    "RiskSignal",
    "DecisionRecord",
    "HardGateOutcome",
    "AlternativeAction",
    "ExecutionOutcome",
    "LearningEvent",
    "UserRole",
    "ApprovalStatus",
    "Team1DataForecastAdapter",
    "Team2DecisionRiskAdapter",
    "create_mock_treasury_state",
    "create_mock_forecast_bundle",
    "create_mock_decision_record",
]
