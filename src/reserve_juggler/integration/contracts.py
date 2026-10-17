"""
Cross-team contracts and structural protocols for Reserve Juggler.

Team 3 exclusively owns `src/reserve_juggler/integration/`.
This module dynamically imports Team 1's canonical `contracts.py` if present,
falling back to fully compatible structural protocols and dataclasses.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum


class EventCategory(str, Enum):
    MONETARY_POLICY = "MONETARY_POLICY"
    MARKET_SHOCK = "MARKET_SHOCK"
    BANKING_STRESS = "BANKING_STRESS"
    WITHDRAWAL_ACCELERATION = "WITHDRAWAL_ACCELERATION"
    MARKET_DISRUPTION = "MARKET_DISRUPTION"
    GEO_POLITICAL = "GEO_POLITICAL"
    GENERAL_NEWS = "GENERAL_NEWS"


class RiskSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class QualityStatus(str, Enum):
    HIGH = "HIGH"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    FAILED = "FAILED"


class UserRole(str, Enum):
    TREASURY_ANALYST = "TREASURY_ANALYST"
    RISK_OFFICER = "RISK_OFFICER"
    HEAD_OF_TREASURY = "HEAD_OF_TREASURY"
    AUDITOR = "AUDITOR"


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MODIFIED = "MODIFIED"
    OVERRIDDEN = "OVERRIDDEN"


# Structural Dataclasses matching Canonical cross-team specifications:

@dataclass
class AssetPosition:
    asset_id: str
    name: str
    asset_class: str  # "CASH", "REPO", "SOVEREIGN_BONDS", "EQUITIES"
    market_value: float
    accessible_liquidity: float  # After haircuts and settlement rules
    settlement_days: int  # T+0, T+1, T+2
    yield_pct: float
    maturity_date: Optional[str] = None
    duration_years: float = 0.0
    haircut_pct: float = 0.0
    is_eligible: bool = True
    allocation_cap: float = 0.0  # Max permitted allocation in portfolio


@dataclass
class Obligation:
    obligation_id: str
    due_timestamp: str
    amount: float
    description: str
    priority: str = "MANDATORY"  # "MANDATORY", "OPERATIONAL", "DISCRETIONARY"
    beneficiary: str = ""


@dataclass
class TreasuryState:
    timestamp: str
    available_cash: float
    asset_positions: List[AssetPosition]
    obligations: List[Obligation]
    authorized_credit_availability: float
    data_quality_warnings: List[str] = field(default_factory=list)
    protected_reserve_floor: float = 0.0
    core_capital_pct: float = 0.70  # Core capital percentage
    hot_capital_pct: float = 0.30   # Hot capital percentage


@dataclass
class HorizonForecast:
    horizon: str  # "1h", "6h", "24h", "48h", "72h"
    expected_inflow: float
    expected_outflow: float
    expected_net_flow: float
    uncertainty_p05: float
    uncertainty_p50: float
    uncertainty_p95: float
    confidence_score: float  # 0.0 - 1.0


@dataclass
class ForecastBundle:
    timestamp: str
    point_forecasts: Dict[str, HorizonForecast]  # Keyed by horizon string
    scenarios: Dict[str, Dict[str, float]]  # e.g. {"STRESS_RUN": {"1h": -5000000, ...}}
    quality_status: str  # "HIGH", "DEGRADED", "STALE"
    model_versions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskSignal:
    signal_id: str
    source: str
    category: str
    publication_time: str
    ingestion_time: str
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    confidence: float  # 0.0 to 1.0
    evidence: str
    evidence_url: Optional[str] = None
    affected_markets: List[str] = field(default_factory=list)
    scenario_tags: List[str] = field(default_factory=list)
    corroboration_status: str = "UNVERIFIED"  # "UNVERIFIED", "CORROBORATED", "OFFICIAL"
    sentiment_score: float = 0.0  # -1.0 (very negative) to +1.0 (very positive)
    is_duplicate: bool = False
    relevance_to_treasury: float = 0.0  # 0.0 to 1.0


@dataclass
class HardGateOutcome:
    gate_name: str
    passed: bool
    threshold_value: float
    observed_value: float
    reason: str


@dataclass
class AlternativeAction:
    action_type: str  # "HOLD_CASH", "DRAW_REPO", "LIQUIDATE_BOND", "DRAW_CREDIT", "EQUITY_TRIM"
    target_asset: str
    amount: float
    estimated_total_cost: float
    settlement_horizon_hours: int
    rejection_reason: str
    net_yield_impact: float


@dataclass
class DecisionRecord:
    decision_id: str
    timestamp: str
    selected_action: str
    target_asset: str
    amount: float
    reserve: float
    eligible_surplus: float
    estimated_costs: Dict[str, float]  # interest, tax, penalty, slippage, total
    hard_gate_outcomes: List[HardGateOutcome]
    model_versions: Dict[str, str]
    reason_codes: List[str]
    alternatives: List[AlternativeAction] = field(default_factory=list)
    treasury_safety_score: float = 85.0
    tss_factors: Dict[str, float] = field(default_factory=dict)
    approval_status: str = ApprovalStatus.PENDING.value
    approved_by: Optional[str] = None
    approval_role: Optional[str] = None
    approval_timestamp: Optional[str] = None
    override_reason: Optional[str] = None


@dataclass
class ExecutionOutcome:
    outcome_id: str
    decision_id: str
    correlation_id: str
    execution_timestamp: str
    simulated: bool
    realized_amount: float
    realized_costs: Dict[str, float]  # interest, tax, penalty, slippage, total
    errors: List[str] = field(default_factory=list)
    liquidity_shortfall: float = 0.0  # Dollar amount of deficit below required reserve
    settlement_status: str = "SETTLED"  # "PENDING", "SETTLED", "DELAYED", "FAILED"
    slippage_bps: float = 0.0


@dataclass
class LearningEvent:
    event_id: str
    versioned_event_type: str
    module: str  # "DATA_INGESTION", "FORECASTING", "TSS_RESERVE", "SENTINEL", "DECISION_ENGINE", "HUMAN_OVERSIGHT", "EXECUTION"
    timestamp: str
    correlation_id: str
    payload: Dict[str, Any]
    associated_versions: Dict[str, str] = field(default_factory=dict)
