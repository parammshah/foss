"""
Canonical Domain Contracts and Shared Interfaces for Reserve Juggler.

This module defines the single source of truth for cross-team data contracts:
- TreasuryState (Team 1 / Ledger)
- ForecastBundle & ForecastPoint (Team 1 / Forecasting)
- RiskSignal (Team 3 / Macro-Panic Sentinel)
- DecisionRecord (Team 2 / Treasury Decision Service)
- ExecutionOutcome (Team 2 / Execution Simulator)
- LearningEvent (Cross-System Telemetry)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ServiceStatus(str, Enum):
    CONNECTED = "CONNECTED"
    UNINTEGRATED = "UNINTEGRATED"
    UNAVAILABLE = "UNAVAILABLE"
    SYNTHETIC_DEMO = "SYNTHETIC_DEMO"
    FALLBACK = "FALLBACK"
    LIVE = "LIVE"
    PARTIAL = "PARTIAL"


class EventCategory(str, Enum):
    MONETARY_POLICY = "MONETARY_POLICY"
    MARKET_SHOCK = "MARKET_SHOCK"
    BANKING_STRESS = "BANKING_STRESS"
    WITHDRAWAL_ACCELERATION = "WITHDRAWAL_ACCELERATION"
    MARKET_DISRUPTION = "MARKET_DISRUPTION"
    GEO_POLITICAL = "GEO_POLITICAL"
    GENERAL_NEWS = "GENERAL_NEWS"
    UNKNOWN = "UNKNOWN"


class RiskSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class QualityStatus(str, Enum):
    HIGH = "HIGH"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"


class UserRole(str, Enum):
    AUDITOR = "AUDITOR"
    TREASURY_ANALYST = "TREASURY_ANALYST"
    RISK_OFFICER = "RISK_OFFICER"
    HEAD_OF_TREASURY = "HEAD_OF_TREASURY"


class ApprovalStatus(str, Enum):
    PROPOSED = "PROPOSED"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MODIFIED = "MODIFIED"
    CANCELLED = "CANCELLED"
    OVERRIDDEN = "OVERRIDDEN"
    EXECUTION_PENDING = "EXECUTION_PENDING"
    EXECUTED = "EXECUTED"
    EXECUTION_FAILED = "EXECUTION_FAILED"


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    NOT_EVALUATED = "NOT_EVALUATED"


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
    core_capital_pct: float = 0.70
    hot_capital_pct: float = 0.30
    is_synthetic: bool = False
    source_status: ServiceStatus = ServiceStatus.SYNTHETIC_DEMO


@dataclass
class ForecastPoint:
    horizon: str  # "1h", "6h", "24h", "48h", "72h"
    target_timestamp: str
    expected_inflow: float
    expected_outflow: float
    expected_net_flow: float
    uncertainty_p05: float
    uncertainty_p50: float
    uncertainty_p90: float
    uncertainty_p95: float
    uncertainty_p99: float
    confidence_score: float  # 0.0 - 1.0
    model_version: str = "v1.4.0"


# HorizonForecast alias for backward compatibility
HorizonForecast = ForecastPoint


@dataclass
class ForecastBundle:
    timestamp: str
    point_forecasts: Dict[str, ForecastPoint]  # Keyed by horizon string e.g. "1h", "6h"
    scenarios: Dict[str, Dict[str, float]]
    quality_status: str  # "HIGH", "DEGRADED", "STALE", "UNAVAILABLE"
    model_versions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_synthetic: bool = False
    source_status: ServiceStatus = ServiceStatus.SYNTHETIC_DEMO


@dataclass
class RiskSignal:
    signal_id: str
    source: str
    category: str
    publication_time: str
    ingestion_time: str
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL", "LOW_CONFIDENCE"
    confidence: float  # 0.0 to 1.0
    evidence: str
    evidence_url: Optional[str] = None
    affected_markets: List[str] = field(default_factory=list)
    scenario_tags: List[str] = field(default_factory=list)
    corroboration_status: str = "UNVERIFIED"  # "UNVERIFIED", "CORROBORATED", "OFFICIAL"
    sentiment_score: float = 0.0  # -1.0 to +1.0
    is_duplicate: bool = False
    relevance_to_treasury: float = 0.0  # 0.0 to 1.0


@dataclass
class HardGateOutcome:
    gate_name: str
    status: GateStatus = GateStatus.PASS
    threshold_value: float = 0.0
    observed_value: float = 0.0
    reason: str = ""

    @property
    def passed(self) -> bool:
        return self.status == GateStatus.PASS


@dataclass
class AlternativeAction:
    action_type: str  # "HOLD_CASH", "DRAW_REPO", "LIQUIDATE_BOND", "DRAW_CREDIT", "EQUITY_TRIM"
    target_asset: str
    amount: float
    estimated_total_cost: float
    settlement_horizon_hours: int
    rejection_reason: str
    net_yield_impact: float
    is_feasible: bool = False


@dataclass
class DecisionRecord:
    decision_id: str
    timestamp: str
    selected_action: str
    target_asset: str
    amount: float
    reserve: float
    eligible_surplus: float
    estimated_costs: Dict[str, float]  # interest, tax, liquidation_penalty, slippage, total
    hard_gate_outcomes: List[HardGateOutcome]
    model_versions: Dict[str, str]
    reason_codes: List[str]
    alternatives: List[AlternativeAction] = field(default_factory=list)
    treasury_safety_score: float = 85.0
    tss_factors: Dict[str, float] = field(default_factory=dict)
    approval_status: str = ApprovalStatus.PENDING_APPROVAL.value
    approved_by: Optional[str] = None
    approval_role: Optional[str] = None
    approval_timestamp: Optional[str] = None
    override_reason: Optional[str] = None
    is_synthetic: bool = False
    source_status: ServiceStatus = ServiceStatus.SYNTHETIC_DEMO


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
    liquidity_shortfall: float = 0.0
    settlement_status: str = "SETTLED"  # "PENDING", "SETTLED", "DELAYED", "FAILED", "CANCELLED"
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
