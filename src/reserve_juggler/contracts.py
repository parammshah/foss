"""Canonical integration contracts and shared domain models for Reserve Juggler.

This module defines typed, JSON-serializable Pydantic schemas shared across all
three teams (Data & Forecasting, Treasury Optimization, and Continuous Learning).

Financial Standards:
- Currency: USD (represented as float).
- Rates: Normalized decimal values (e.g., 4.8% is represented as 0.048, 0.5% is 0.005).
- Timestamps: UTC-aware datetime objects (`datetime.timezone.utc`).
- Tiers:
    - Tier 0: Operational cash, 0% APY.
    - Tier 1: Overnight repo, ~4.3% APY, 17:00 entry, 08:00 maturity, 0.5% early break penalty.
    - Tier 2: Two-year sovereign bonds, ~4.8% yield, 2.0% emergency liquidation haircut.
    - Tier 3: Equity index, ~10.0% expected return, 18% volatility, 15% emergency slippage.
"""

from __future__ import annotations

import abc
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AssetTier(IntEnum):
    """Asset hierarchy tiers defined in the challenge brief."""
    TIER_0_CASH = 0
    TIER_1_REPO = 1
    TIER_2_BONDS = 2
    TIER_3_EQUITIES = 3


class ObligationType(str, Enum):
    """Categories of scheduled cash obligations."""
    MERCHANT_SETTLEMENT = "merchant_settlement"
    CUSTOMER_PAYOUT = "customer_payout"
    OPERATIONAL_EXPENSE = "operational_expense"
    CREDIT_FACILITY_REPAYMENT = "credit_facility_repayment"
    TAX_LIABILITY = "tax_liability"
    OTHER = "other"


class DataQualityStatus(str, Enum):
    """Health status of datasets and generated features."""
    OK = "OK"
    IMPUTED = "IMPUTED"
    DEGRADED = "DEGRADED"
    WARNING = "WARNING"
    CRITICAL_ERROR = "CRITICAL_ERROR"


def ensure_utc(dt: datetime) -> datetime:
    """Ensure a datetime object is timezone-aware and set to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class AssetPosition(BaseModel):
    """Marked-to-market and liquidatable holding position in a specific asset tier.

    Separates accounting marked-to-market value from immediately realizable
    liquidation proceeds (accounting for early exit penalties, haircuts, and slippage).
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str = Field(description="Unique asset identifier (e.g., 'T0_CASH', 'T1_REPO_OVERNIGHT', 'T2_UST_2Y', 'T3_SP500')")
    tier: AssetTier = Field(description="Asset tier (0: Cash, 1: Repo, 2: Bonds, 3: Equities)")
    units: float = Field(default=0.0, ge=0.0, description="Quantity or face value of units held")
    unit_price_usd: float = Field(default=1.0, ge=0.0, description="Current marked-to-market unit price in USD")
    market_value_usd: float = Field(ge=0.0, description="Marked-to-market value = units * unit_price_usd")
    liquidatable_value_usd: float = Field(ge=0.0, description="Estimated realizable liquidation proceeds after penalties/haircuts")

    availability_timestamp: Optional[datetime] = Field(default=None, description="UTC timestamp when position can be liquidated/accessed")
    maturity_timestamp: Optional[datetime] = Field(default=None, description="UTC timestamp of maturity (e.g., 08:00 for overnight repo)")
    settlement_delay_hours: int = Field(default=0, ge=0, description="Settlement lag in hours before proceeds reach operational cash")

    # Financial returns & penalties normalized to decimal fractions (e.g. 0.048 = 4.8%)
    annual_yield_or_return: float = Field(default=0.0, description="Normalized annual yield (bonds/repo APY) or expected return (equities)")
    early_exit_fee_rate: float = Field(default=0.0, ge=0.0, description="Penalty decimal applied if broken before maturity (e.g., 0.005 for repo)")
    liquidation_haircut_rate: float = Field(default=0.0, ge=0.0, description="Haircut or slippage decimal applied upon liquidation (e.g. 0.02 or 0.15)")

    annualized_volatility: Optional[float] = Field(default=None, ge=0.0, description="Annualized volatility decimal where relevant (Tier 3)")
    duration_years: Optional[float] = Field(default=None, ge=0.0, description="Effective duration in years where relevant (Tier 2)")
    accrued_income_usd: float = Field(default=0.0, description="Accrued interest/dividends not yet paid out into cash")
    cost_basis_usd: float = Field(default=0.0, ge=0.0, description="Total purchase cost basis in USD")

    @field_validator("availability_timestamp", "maturity_timestamp", mode="before")
    @classmethod
    def validate_utc(cls, v: Optional[datetime]) -> Optional[datetime]:
        if v is None:
            return None
        return ensure_utc(v) if isinstance(v, datetime) else v


class Obligation(BaseModel):
    """Upcoming payout or settlement liability that must be funded."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    obligation_id: str = Field(description="Unique identifier for the scheduled obligation")
    timestamp: datetime = Field(description="UTC timestamp when the obligation must be settled")
    amount_usd: float = Field(gt=0.0, description="Obligation amount in USD")
    obligation_type: ObligationType = Field(default=ObligationType.OTHER, description="Category of obligation")
    is_settled: bool = Field(default=False, description="Whether this obligation has been executed/settled")
    priority: int = Field(default=1, ge=1, le=10, description="Priority rank (1 = mandatory legal/regulatory settlement)")

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_utc(cls, v: datetime) -> datetime:
        return ensure_utc(v) if isinstance(v, datetime) else v


class CreditFacility(BaseModel):
    """Committed or uncommitted emergency credit facility terms."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    facility_id: str = Field(description="Credit facility identifier")
    is_available: bool = Field(default=True, description="Whether facility is active and accessible")
    committed_limit_usd: float = Field(ge=0.0, description="Maximum borrowing capacity in USD")
    utilized_amount_usd: float = Field(default=0.0, ge=0.0, description="Currently drawn/outstanding balance in USD")
    apr: float = Field(ge=0.0, description="Annual percentage rate in decimal format (e.g. 0.075 for 7.5%)")
    fee_rate: float = Field(default=0.0, ge=0.0, description="Undrawn commitment fee rate or draw fee decimal")
    min_draw_usd: float = Field(default=0.0, ge=0.0, description="Minimum single drawdown amount in USD")
    max_draw_usd: float = Field(default=0.0, ge=0.0, description="Maximum single drawdown amount in USD (0 = limit)")
    permitted_draw_rules: Dict[str, Any] = Field(default_factory=dict, description="Constraints/rules governing drawdowns")

    @property
    def available_capacity_usd(self) -> float:
        """Unutilized credit capacity available for immediate draw."""
        if not self.is_available:
            return 0.0
        return max(0.0, self.committed_limit_usd - self.utilized_amount_usd)


class TreasuryState(BaseModel):
    """Reconciled point-in-time financial state of the digital bank's treasury.

    Exposes immediately available operational cash, marked-to-market positions,
    liquidatable values, upcoming obligations, and credit capacity.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(description="UTC as-of timestamp for this snapshot")
    operational_cash_usd: float = Field(description="Immediately available cash in USD (Tier 0, 0% APY)")
    asset_positions: Dict[str, AssetPosition] = Field(default_factory=dict, description="Asset positions keyed by asset_id")
    obligations: List[Obligation] = Field(default_factory=list, description="Pending obligations due in the planning window")
    credit_facility: Optional[CreditFacility] = Field(default=None, description="Available emergency credit facility terms")
    warnings: List[str] = Field(default_factory=list, description="Data quality, liquidity, or accounting warnings")

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_utc(cls, v: datetime) -> datetime:
        return ensure_utc(v) if isinstance(v, datetime) else v

    @property
    def operational_cash(self) -> float:
        """Convenience alias for operational_cash_usd."""
        return self.operational_cash_usd

    @property
    def total_investments_market_value_usd(self) -> float:
        """Total marked-to-market value of non-cash asset positions (Tiers 1, 2, 3)."""
        return sum(pos.market_value_usd for pos in self.asset_positions.values() if pos.tier != AssetTier.TIER_0_CASH)

    @property
    def total_balance_sheet_assets_usd(self) -> float:
        """Total marked-to-market assets: operational cash + all asset positions."""
        return self.operational_cash_usd + sum(pos.market_value_usd for pos in self.asset_positions.values())

    @property
    def total_liquidatable_value_usd(self) -> float:
        """Total immediately realizable liquidity: operational cash + post-haircut liquidatable investments."""
        return self.operational_cash_usd + sum(pos.liquidatable_value_usd for pos in self.asset_positions.values())

    @property
    def upcoming_obligations_usd(self) -> float:
        """Sum of unsettled obligations in USD."""
        return sum(ob.amount_usd for ob in self.obligations if not ob.is_settled)

    @property
    def available_credit_usd(self) -> float:
        """Emergency credit capacity available."""
        if self.credit_facility is None:
            return 0.0
        return self.credit_facility.available_capacity_usd

    @property
    def net_liquidity_usd(self) -> float:
        """Conservative net liquidity = liquidatable value + available credit - unsettled obligations."""
        return self.total_liquidatable_value_usd + self.available_credit_usd - self.upcoming_obligations_usd

    def get_tier_market_value(self, tier: AssetTier) -> float:
        """Sum of marked-to-market value in a specific tier."""
        if tier == AssetTier.TIER_0_CASH:
            return self.operational_cash_usd
        return sum(pos.market_value_usd for pos in self.asset_positions.values() if pos.tier == tier)


class ForecastPoint(BaseModel):
    """Multi-horizon probabilistic forecast for a specific future target timestamp.

    Contains separate deposit and withdrawal quantiles (P10, P50, P90) and conservative
    cumulative net-outflow shock buffers (P95, P99) for treasury liquidity planning.
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    as_of_timestamp: datetime = Field(description="UTC timestamp when the forecast was generated")
    target_timestamp: datetime = Field(description="UTC timestamp being predicted")
    horizon_hours: int = Field(gt=0, description="Forecast lead time in hours (e.g., 1, 6, 24, 48, 72)")

    # Deposits quantiles (USD)
    deposit_p10: float = Field(ge=0.0, description="Conservative 10th percentile deposit volume in USD")
    deposit_p50: float = Field(ge=0.0, description="Median expected deposit volume in USD")
    deposit_p90: float = Field(ge=0.0, description="Optimistic 90th percentile deposit volume in USD")

    # Withdrawals quantiles (USD)
    withdrawal_p10: float = Field(ge=0.0, description="Optimistic 10th percentile withdrawal volume in USD")
    withdrawal_p50: float = Field(ge=0.0, description="Median expected withdrawal volume in USD")
    withdrawal_p90: float = Field(ge=0.0, description="Adverse 90th percentile withdrawal volume in USD")

    # Point & cumulative net-outflow shock estimates: Net Outflow = Withdrawals - Deposits
    net_outflow_p50: float = Field(description="Median expected net cash outflow in USD (withdrawals - deposits)")
    net_outflow_p95: float = Field(description="Adverse 95th percentile single-hour net outflow shock in USD")
    net_outflow_p99: float = Field(description="Extreme stress 99th percentile single-hour net outflow shock in USD")

    cumulative_net_outflow_p95: float = Field(default=0.0, description="Conservative cumulative net outflow from t=0 to target at P95")
    cumulative_net_outflow_p99: float = Field(default=0.0, description="Extreme stress cumulative net outflow from t=0 to target at P99")

    reliability_score: float = Field(ge=0.0, le=1.0, default=1.0, description="Model confidence score [0.0 to 1.0]")
    model_version: str = Field(description="Version string identifying the forecasting ensemble/models used")

    @field_validator("as_of_timestamp", "target_timestamp", mode="before")
    @classmethod
    def validate_utc(cls, v: datetime) -> datetime:
        return ensure_utc(v) if isinstance(v, datetime) else v


class ForecastBundle(BaseModel):
    """Complete multi-horizon forecast package delivered to the treasury optimizer."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    forecast_timestamp: datetime = Field(description="UTC as-of timestamp of forecast generation")
    points: List[ForecastPoint] = Field(description="List of forecast points across requested horizons")
    scenario_references: Dict[str, Any] = Field(default_factory=dict, description="OR-reconciled scenarios or stress paths")
    data_quality_status: DataQualityStatus = Field(default=DataQualityStatus.OK, description="Data quality health status")
    model_versions: Dict[str, str] = Field(default_factory=dict, description="Map of submodel names to their version strings")
    evaluation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Validation losses, weights, calibration info")

    @field_validator("forecast_timestamp", mode="before")
    @classmethod
    def validate_utc(cls, v: datetime) -> datetime:
        return ensure_utc(v) if isinstance(v, datetime) else v

    def get_point(self, horizon_hours: int) -> Optional[ForecastPoint]:
        """Retrieve the forecast point corresponding to a given horizon in hours."""
        for pt in self.points:
            if pt.horizon_hours == horizon_hours:
                return pt
        return None


class RiskSignal(BaseModel):
    """External or internal telemetry event alerting to liquidity or market stress."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_id: str = Field(description="Unique risk signal identifier")
    source: str = Field(description="Source module or provider (e.g., 'news_sentinel', 'market_feed')")
    category: str = Field(description="Category (e.g., 'monetary_policy', 'bank_run', 'market_crash')")
    event_timestamp: datetime = Field(description="UTC timestamp when the external event occurred")
    observation_timestamp: datetime = Field(description="UTC timestamp when the system ingested the signal")
    severity: float = Field(ge=0.0, le=1.0, description="Normalized severity score [0.0 = benign, 1.0 = catastrophic]")
    confidence: float = Field(ge=0.0, le=1.0, description="Normalized signal confidence [0.0 = low, 1.0 = certain]")
    evidence_references: List[str] = Field(default_factory=list, description="IDs or URIs of articles, logs, or metrics")
    explanation: str = Field(description="Human-readable explanation of the risk signal")
    stress_scenario_tags: List[str] = Field(default_factory=list, description="Scenario labels mapped to this risk")

    @field_validator("event_timestamp", "observation_timestamp", mode="before")
    @classmethod
    def validate_utc(cls, v: datetime) -> datetime:
        return ensure_utc(v) if isinstance(v, datetime) else v


class DecisionRecord(BaseModel):
    """Structured record of a treasury allocation decision produced by Team 2."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: str = Field(description="Unique decision record identifier")
    timestamp: datetime = Field(description="UTC timestamp when the decision was computed")
    status: str = Field(description="Decision status (e.g., 'EXECUTED', 'PENDING', 'REJECTED')")
    selected_action: Dict[str, Any] = Field(description="The chosen allocation action or rebalance instruction")
    alternatives_considered: List[Dict[str, Any]] = Field(default_factory=list, description="Alternative actions evaluated")
    protected_reserve_usd: float = Field(ge=0.0, description="Mandatory liquidity reserve locked for payout protection")
    eligible_surplus_usd: float = Field(description="Surplus capital eligible for yield-bearing investment")
    score: Optional[float] = Field(default=None, description="Optimizer objective value or utility score")
    cost_breakdown: Dict[str, float] = Field(default_factory=dict, description="Itemized costs (transaction, spread, fees)")
    applied_constraints: List[str] = Field(default_factory=list, description="Constraints active during optimization")
    reason_codes: List[str] = Field(default_factory=list, description="Explaining codes for why this action was selected")
    model_versions: Dict[str, str] = Field(default_factory=dict, description="Versions of optimizer and forecast models used")

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_utc(cls, v: datetime) -> datetime:
        return ensure_utc(v) if isinstance(v, datetime) else v


class ExecutionOutcome(BaseModel):
    """Simulated or realized financial outcome from executing a treasury decision."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome_id: str = Field(description="Unique identifier for this execution outcome")
    decision_id: str = Field(description="ID of the DecisionRecord that triggered this execution")
    execution_status: str = Field(description="Execution result (e.g., 'SUCCESS', 'PARTIAL', 'FAILED', 'SHORTFALL')")
    realized_cash_movements: Dict[str, float] = Field(default_factory=dict, description="Realized inflows/outflows by category")
    costs_usd: float = Field(default=0.0, ge=0.0, description="Direct execution fees and transaction costs in USD")
    penalties_usd: float = Field(default=0.0, ge=0.0, description="Penalties paid (e.g. repo early-break fee)")
    losses_usd: float = Field(default=0.0, ge=0.0, description="Realized capital loss or slippage incurred upon liquidation")
    shortfall_usd: float = Field(default=0.0, ge=0.0, description="Liquidity deficit / unfulfilled payout amount in USD")
    timestamp: datetime = Field(description="UTC timestamp when execution finished")
    reconciliation_status: str = Field(default="RECONCILED", description="Ledger double-entry reconciliation status")

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_utc(cls, v: datetime) -> datetime:
        return ensure_utc(v) if isinstance(v, datetime) else v


class LearningEvent(BaseModel):
    """Standardized event published to Team 3's continuous learning service."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(description="Unique UUID for this learning event")
    event_type: str = Field(description="Event name (e.g. 'forecast_generated', 'forecast_evaluated', 'data_quality_warning', 'ledger_reconciled')")
    source_module: str = Field(description="Originating module name (e.g., 'reserve_juggler.forecasting')")
    event_timestamp: datetime = Field(description="UTC timestamp of the underlying business event")
    recording_timestamp: datetime = Field(description="UTC timestamp when the event was emitted into telemetry")
    correlation_id: Optional[str] = Field(default=None, description="Correlation identifier for tracing workflows")
    decision_id: Optional[str] = Field(default=None, description="Associated decision ID if applicable")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Structured payload data specific to event_type")
    model_version: Optional[str] = Field(default=None, description="Version of active model")
    policy_version: Optional[str] = Field(default=None, description="Version of active policy")
    schema_version: str = Field(default="1.0", description="Learning event schema version")

    @field_validator("event_timestamp", "recording_timestamp", mode="before")
    @classmethod
    def validate_utc(cls, v: datetime) -> datetime:
        return ensure_utc(v) if isinstance(v, datetime) else v


# ---------------------------------------------------------------------------
# Injectable Event Publisher Interface
# ---------------------------------------------------------------------------

class EventPublisher(abc.ABC):
    """Abstract interface for emitting telemetry and domain events.

    Modules accept an optional EventPublisher. When None or NoOpEventPublisher is
    configured, modules operate completely independently without network or logging dependencies.
    """

    @abc.abstractmethod
    def publish(self, event: LearningEvent) -> None:
        """Publish a structured learning/telemetry event."""
        pass


class NoOpEventPublisher(EventPublisher):
    """Default publisher that silently ignores all emitted events."""

    def publish(self, event: LearningEvent) -> None:
        pass


class InMemoryEventPublisher(EventPublisher):
    """Thread-safe in-memory publisher for testing and local integration."""

    def __init__(self) -> None:
        self.events: List[LearningEvent] = []
        self._subscribers: List[Callable[[LearningEvent], None]] = []

    def subscribe(self, callback: Callable[[LearningEvent], None]) -> None:
        """Register a callback to receive events synchronously."""
        self._subscribers.append(callback)

    def publish(self, event: LearningEvent) -> None:
        """Store event in memory and notify subscribers."""
        self.events.append(event)
        for sub in self._subscribers:
            try:
                sub(event)
            except Exception:
                pass

    def clear(self) -> None:
        """Clear recorded events."""
        self.events.clear()

    def get_events_by_type(self, event_type: str) -> List[LearningEvent]:
        """Filter published events by event_type."""
        return [e for e in self.events if e.event_type == event_type]
