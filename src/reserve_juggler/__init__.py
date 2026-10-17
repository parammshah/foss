"""Reserve Juggler: Quantitative Treasury Platform for Digital Banking.

Unified Integration of:
- Team 1: Data Foundation, Ledger, Simulation & Hybrid Forecasting
- Team 2: Treasury Safety Score, Capital Firewall, Tax-Aware OR & Next Best Action
- Team 3: Macro-Panic Sentinel, System-Wide Self-Learning & Explainable Oversight
"""

__version__ = "1.0.0"

# Team 1 Contracts & Utilities
from reserve_juggler.contracts import (
    AssetPosition,
    AssetTier,
    CreditFacility,
    DataQualityStatus,
    DecisionRecord,
    EventPublisher,
    ExecutionOutcome,
    ForecastBundle,
    ForecastPoint,
    InMemoryEventPublisher,
    LearningEvent,
    NoOpEventPublisher,
    Obligation,
    ObligationType,
    RiskSignal,
    TreasuryState,
    ensure_utc,
)

# Team 2 Decision, Risk, Tax & Optimization
from reserve_juggler.decision.service import TreasuryDecisionService
from reserve_juggler.optimization.optimizer import optimize_treasury_actions
from reserve_juggler.risk.capital_firewall import (
    calculate_holding_time_gate,
    classify_hot_core_capital,
)
from reserve_juggler.risk.liquidity_shield import (
    assess_liquidity_reserve,
    evaluate_hard_safety_gates,
)
from reserve_juggler.risk.safety_score import calculate_treasury_safety_score
from reserve_juggler.tax.tax_engine import evaluate_tax_adjusted_value

__all__ = [
    # Team 1
    "AssetPosition",
    "AssetTier",
    "CreditFacility",
    "DataQualityStatus",
    "DecisionRecord",
    "EventPublisher",
    "ExecutionOutcome",
    "ForecastBundle",
    "ForecastPoint",
    "InMemoryEventPublisher",
    "LearningEvent",
    "NoOpEventPublisher",
    "Obligation",
    "ObligationType",
    "RiskSignal",
    "TreasuryState",
    "ensure_utc",
    # Team 2
    "TreasuryDecisionService",
    "calculate_treasury_safety_score",
    "assess_liquidity_reserve",
    "evaluate_hard_safety_gates",
    "classify_hot_core_capital",
    "calculate_holding_time_gate",
    "optimize_treasury_actions",
    "evaluate_tax_adjusted_value",
]
