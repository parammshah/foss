"""Canonical integration contracts and shared domain models for Reserve Juggler.

Unified Single Source of Truth shared across:
- Team 1: Data Foundation, Ledger, Simulation & Hybrid Forecasting
- Team 2: Treasury Safety Score, Capital Firewall, Tax-Aware OR & Next Best Action
- Team 3: Macro-Panic Sentinel, System-Wide Self-Learning & Explainable Oversight

Financial Standards:
- Currency: USD (represented as float).
- Rates: Normalized decimal values (e.g., 4.8% is 0.048, 0.5% is 0.005).
- Timestamps: UTC-aware datetime or ISO 8601 UTC strings.
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator


# =============================================================================
# 1. Canonical Enumerations
# =============================================================================

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


class DataQualityStatus(str, Enum):
    OK = "OK"
    IMPUTED = "IMPUTED"
    DEGRADED = "DEGRADED"
    WARNING = "WARNING"
    CRITICAL_ERROR = "CRITICAL_ERROR"


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


class ActionType(str, Enum):
    HOLD_CASH = "HOLD_CASH"
    INVEST_REPO = "INVEST_REPO"
    INVEST_BONDS = "INVEST_BONDS"
    INVEST_EQUITIES = "INVEST_EQUITIES"
    LIQUIDATE_REPO = "LIQUIDATE_REPO"
    LIQUIDATE_BONDS = "LIQUIDATE_BONDS"
    LIQUIDATE_EQUITIES = "LIQUIDATE_EQUITIES"
    DRAW_REPO_30D = "DRAW_REPO_30D"
    DRAW_CREDIT = "DRAW_CREDIT"
    REPAY_CREDIT = "REPAY_CREDIT"
    UNKNOWN = "UNKNOWN"


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


# =============================================================================
# 2. Helpers
# =============================================================================

def ensure_utc(dt: Union[datetime, str]) -> datetime:
    """Ensure a datetime or ISO string is a timezone-aware UTC datetime."""
    if isinstance(dt, str):
        cleaned = dt.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(cleaned)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def format_iso_utc(dt: Union[datetime, str, None]) -> str:
    """Ensure output is formatted as an ISO 8601 UTC string."""
    if dt is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(dt, str):
        return dt
    return ensure_utc(dt).isoformat()


# =============================================================================
# 3. Base Canonical Contract Model
# =============================================================================

class CanonicalBaseModel(BaseModel):
    """Base model with arbitrary types, extra field retention, and JSON convenience."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        populate_by_name=True,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for JSON serialization."""
        return self.model_dump()


# =============================================================================
# 4. Domain Models
# =============================================================================

class HardGateOutcome(CanonicalBaseModel):
    gate_name: str
    status: GateStatus = GateStatus.NOT_EVALUATED
    threshold_value: float = 0.0
    observed_value: float = 0.0
    reason: str = ""

    @property
    def passed(self) -> bool:
        return self.status == GateStatus.PASS


class AlternativeAction(CanonicalBaseModel):
    action_type: str = "HOLD_CASH"
    target_asset: str = "CASH"
    amount: float = 0.0
    estimated_total_cost: float = 0.0
    settlement_horizon_hours: int = 0
    rejection_reason: str = ""
    net_yield_impact: float = 0.0
    is_feasible: bool = False


class Obligation(CanonicalBaseModel):
    obligation_id: str = Field(default_factory=lambda: f"obl-{uuid.uuid4().hex[:8]}")
    due_timestamp: Union[datetime, str] = Field(default_factory=lambda: datetime.now(timezone.utc))
    amount: float = 0.0
    amount_usd: float = 0.0
    description: str = ""
    priority: str = "MANDATORY"
    is_mandatory: bool = True
    beneficiary: str = ""
    counterparty: str = ""
    obligation_type: ObligationType = ObligationType.OTHER

    @model_validator(mode="before")
    @classmethod
    def sync_obligation_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Sync amount and amount_usd
            amt = data.get("amount") or data.get("amount_usd", 0.0)
            data["amount"] = float(amt)
            data["amount_usd"] = float(amt)
            # Sync due_time
            if "due_time" in data and "due_timestamp" not in data:
                data["due_timestamp"] = data["due_time"]
            # Sync mandatory priority
            if "is_mandatory" in data:
                data["priority"] = "MANDATORY" if data["is_mandatory"] else "OPERATIONAL"
            elif data.get("priority") == "MANDATORY":
                data["is_mandatory"] = True
        return data

    @property
    def due_time(self) -> datetime:
        return ensure_utc(self.due_timestamp)


class AssetPosition(CanonicalBaseModel):
    asset_id: str
    name: str = ""
    tier: Union[AssetTier, int] = AssetTier.TIER_0_CASH
    asset_class: str = "CASH"
    units: float = 0.0
    unit_price_usd: float = 1.0
    market_value_usd: float = 0.0
    liquidatable_value_usd: float = 0.0
    cost_basis_usd: float = 0.0
    unrealized_gain_usd: float = 0.0
    availability_timestamp: Optional[Union[datetime, str]] = None
    maturity_timestamp: Optional[Union[datetime, str]] = None
    settlement_delay_hours: float = 0.0
    settlement_days: int = 0
    annual_yield_or_return: float = 0.0
    liquidation_haircut_rate: float = 0.0
    early_exit_penalty_pct: float = 0.0
    duration_years: float = 0.0
    volatility_annual_pct: float = 0.0
    instrument_type: str = "cash"
    is_eligible: bool = True
    allocation_cap: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def sync_asset_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            mv = data.get("market_value_usd") if data.get("market_value_usd") is not None else data.get("market_value", 0.0)
            data["market_value_usd"] = float(mv)
            liq = data.get("liquidatable_value_usd") if data.get("liquidatable_value_usd") is not None else data.get("accessible_liquidity", mv)
            data["liquidatable_value_usd"] = float(liq)

            # Haircut
            hc = data.get("liquidation_haircut_rate") if data.get("liquidation_haircut_rate") is not None else data.get("haircut_pct", data.get("haircut", 0.0))
            data["liquidation_haircut_rate"] = float(hc)

            # Yield
            yd = data.get("annual_yield_or_return") if data.get("annual_yield_or_return") is not None else data.get("yield_pct", data.get("annual_yield", 0.0))
            data["annual_yield_or_return"] = float(yd)

            # Exit fees
            ef = data.get("early_exit_penalty_pct") if data.get("early_exit_penalty_pct") is not None else data.get("exit_fees", 0.0)
            data["early_exit_penalty_pct"] = float(ef)

            # Duration
            dur = data.get("duration_years") if data.get("duration_years") is not None else data.get("duration", 0.0)
            data["duration_years"] = float(dur)

            # Volatility
            vol = data.get("volatility_annual_pct") if data.get("volatility_annual_pct") is not None else data.get("volatility", 0.0)
            data["volatility_annual_pct"] = float(vol)

            # Maturity date / timestamp
            mat = data.get("maturity_timestamp") or data.get("maturity_date") or data.get("maturity")
            data["maturity_timestamp"] = mat
            if mat:
                data["maturity_date"] = str(mat)

            # Availability
            avail = data.get("availability_timestamp") or data.get("availability")
            data["availability_timestamp"] = avail
        return data

    @property
    def market_value(self) -> float:
        return self.market_value_usd

    @property
    def accessible_liquidity(self) -> float:
        return self.liquidatable_value_usd

    @property
    def liquidatable_value(self) -> float:
        return self.liquidatable_value_usd

    @property
    def annual_yield(self) -> float:
        return self.annual_yield_or_return

    @property
    def yield_pct(self) -> float:
        return self.annual_yield_or_return

    @property
    def haircut(self) -> float:
        return self.liquidation_haircut_rate

    @property
    def haircut_pct(self) -> float:
        return self.liquidation_haircut_rate

    @property
    def exit_fees(self) -> float:
        return self.early_exit_penalty_pct

    @property
    def duration(self) -> float:
        return self.duration_years

    @property
    def volatility(self) -> float:
        return self.volatility_annual_pct

    @property
    def cost_basis(self) -> float:
        return self.cost_basis_usd

    @property
    def availability(self) -> datetime:
        return ensure_utc(self.availability_timestamp) if self.availability_timestamp else datetime.now(timezone.utc)

    @property
    def maturity(self) -> Optional[datetime]:
        return ensure_utc(self.maturity_timestamp) if self.maturity_timestamp else None


class CreditFacility(CanonicalBaseModel):
    committed_limit_usd: float = 50_000_000.0
    drawn_amount_usd: float = 0.0
    apr: float = 0.12
    facility_name: str = "Standard Emergency Overdraft"
    is_active: bool = True

    @property
    def available_credit_usd(self) -> float:
        if not self.is_active:
            return 0.0
        return max(0.0, self.committed_limit_usd - self.drawn_amount_usd)


class TreasuryState(CanonicalBaseModel):
    timestamp: Union[datetime, str] = Field(default_factory=lambda: datetime.now(timezone.utc))
    tier0_cash_usd: float = 0.0
    available_cash: float = 0.0
    tier1_repo_usd: float = 0.0
    tier2_bonds_usd: float = 0.0
    tier3_stocks_usd: float = 0.0
    emergency_credit_available_usd: float = 50_000_000.0
    emergency_credit_drawn_usd: float = 0.0
    authorized_credit_availability: float = 50_000_000.0
    credit_facility: Optional[CreditFacility] = None
    asset_positions: List[AssetPosition] = Field(default_factory=list)
    obligations: List[Obligation] = Field(default_factory=list)
    future_obligations: List[Obligation] = Field(default_factory=list)
    data_quality_warnings: List[str] = Field(default_factory=list)
    protected_reserve_floor: Optional[float] = None
    core_capital_pct: float = 0.70
    hot_capital_pct: float = 0.30
    is_synthetic: bool = False
    source_status: ServiceStatus = ServiceStatus.SYNTHETIC_DEMO

    @model_validator(mode="before")
    @classmethod
    def sync_treasury_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            cash = data.get("tier0_cash_usd") if data.get("tier0_cash_usd") is not None else data.get("available_cash", 0.0)
            data["tier0_cash_usd"] = float(cash)
            data["available_cash"] = float(cash)

            cred = data.get("emergency_credit_available_usd") if data.get("emergency_credit_available_usd") is not None else data.get("authorized_credit_availability", 50_000_000.0)
            data["emergency_credit_available_usd"] = float(cred)
            data["authorized_credit_availability"] = float(cred)

            # Obligations
            obls = data.get("obligations") or data.get("future_obligations") or []
            data["obligations"] = obls
            data["future_obligations"] = obls
        return data

    @property
    def credit_availability(self) -> float:
        return self.emergency_credit_available_usd

    @property
    def credit_limit(self) -> float:
        return self.emergency_credit_available_usd + self.emergency_credit_drawn_usd

    @property
    def credit_apr(self) -> float:
        return self.credit_facility.apr if self.credit_facility else 0.12

    @property
    def positions_by_tier(self) -> Dict[int, List[AssetPosition]]:
        res: Dict[int, List[AssetPosition]] = {0: [], 1: [], 2: [], 3: []}
        for pos in self.asset_positions:
            tier_val = int(pos.tier)
            res.setdefault(tier_val, []).append(pos)
        return res

    @property
    def total_portfolio_market_value_usd(self) -> float:
        return self.tier0_cash_usd + sum(p.market_value_usd for p in self.asset_positions)

    @property
    def total_liquidatable_value_usd(self) -> float:
        return self.tier0_cash_usd + sum(p.liquidatable_value_usd for p in self.asset_positions)


class ForecastPoint(CanonicalBaseModel):
    as_of_timestamp: Union[datetime, str] = Field(default_factory=lambda: datetime.now(timezone.utc))
    target_timestamp: Union[datetime, str] = Field(default_factory=lambda: datetime.now(timezone.utc))
    horizon_hours: int = 1
    horizon: str = "1h"
    expected_inflow: float = 0.0
    expected_outflow: float = 0.0
    expected_net_flow: float = 0.0
    deposit_point_forecast: float = 0.0
    withdrawal_point_forecast: float = 0.0
    deposit_p025: float = 0.0
    deposit_p10: float = 0.0
    deposit_p25: float = 0.0
    deposit_p50: float = 0.0
    deposit_p75: float = 0.0
    deposit_p90: float = 0.0
    deposit_p975: float = 0.0
    withdrawal_p025: float = 0.0
    withdrawal_p10: float = 0.0
    withdrawal_p25: float = 0.0
    withdrawal_p50: float = 0.0
    withdrawal_p75: float = 0.0
    withdrawal_p90: float = 0.0
    withdrawal_p975: float = 0.0
    net_outflow_p50: float = 0.0
    net_outflow_p95: float = 0.0
    net_outflow_p99: float = 0.0
    cumulative_net_outflow_p95: Optional[float] = None
    cumulative_net_outflow_p99: Optional[float] = None
    uncertainty_p05: float = 0.0
    uncertainty_p50: float = 0.0
    uncertainty_p90: float = 0.0
    uncertainty_p95: float = 0.0
    uncertainty_p99: float = 0.0
    calibration_sample_size: int = 0
    training_cutoff_timestamp: Optional[Union[datetime, str]] = None
    reliability_tier: str = "HIGH"
    reliability_rationale: str = ""
    is_fallback: bool = False
    reliability_score: float = 0.90
    confidence_score: float = 0.90
    model_version: str = "v1.0"

    @model_validator(mode="before")
    @classmethod
    def sync_point_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Horizon
            if "horizon_hours" in data and "horizon" not in data:
                data["horizon"] = f"{data['horizon_hours']}h"
            elif "horizon" in data and "horizon_hours" not in data:
                try:
                    data["horizon_hours"] = int(str(data["horizon"]).replace("h", "").strip())
                except ValueError:
                    data["horizon_hours"] = 1

            # Point expectations
            dep = data.get("deposit_point_forecast") if data.get("deposit_point_forecast") is not None else data.get("expected_inflow", 0.0)
            wid = data.get("withdrawal_point_forecast") if data.get("withdrawal_point_forecast") is not None else data.get("expected_outflow", 0.0)
            data["deposit_point_forecast"] = float(dep)
            data["withdrawal_point_forecast"] = float(wid)
            data["expected_inflow"] = float(dep)
            data["expected_outflow"] = float(wid)
            data["expected_net_flow"] = float(dep - wid)

            # Confidence
            conf = data.get("reliability_score") if data.get("reliability_score") is not None else data.get("confidence_score", 0.90)
            data["reliability_score"] = float(conf)
            data["confidence_score"] = float(conf)

            # Net outflows & uncertainties
            w90 = data.get("withdrawal_p90", wid * 1.1)
            w50 = data.get("withdrawal_p50", wid)
            d10 = data.get("deposit_p10", dep * 0.9)
            d50 = data.get("deposit_p50", dep)

            net_p50 = data.get("net_outflow_p50", max(0.0, w50 - d50))
            net_p95 = data.get("net_outflow_p95", max(0.0, w90 - d10))
            net_p99 = data.get("net_outflow_p99", net_p95 * 1.25)

            data["net_outflow_p50"] = float(net_p50)
            data["net_outflow_p95"] = float(net_p95)
            data["net_outflow_p99"] = float(net_p99)

            u05 = data.get("uncertainty_p05", dep * 0.8 - wid * 1.2)
            u50 = data.get("uncertainty_p50", dep - wid)
            u90 = data.get("uncertainty_p90", dep * 1.1 - wid * 0.9)
            u95 = data.get("uncertainty_p95", dep * 1.2 - wid * 0.8)
            u99 = data.get("uncertainty_p99", dep * 1.3 - wid * 0.7)

            data["uncertainty_p05"] = float(u05)
            data["uncertainty_p50"] = float(u50)
            data["uncertainty_p90"] = float(u90)
            data["uncertainty_p95"] = float(u95)
            data["uncertainty_p99"] = float(u99)

        return data

    @property
    def forecast_horizon(self) -> float:
        return float(self.horizon_hours)

    @property
    def conservative_outflow(self) -> float:
        return self.withdrawal_p90

    @property
    def reliability_metadata(self) -> Dict[str, Any]:
        return {
            "tier": self.reliability_tier,
            "rationale": self.reliability_rationale,
            "calibration_sample_size": self.calibration_sample_size,
            "is_fallback": self.is_fallback,
            "score": self.reliability_score,
        }


# Alias for backward compatibility
HorizonForecast = ForecastPoint


class ForecastBundle(CanonicalBaseModel):
    timestamp: Union[datetime, str] = Field(default_factory=lambda: datetime.now(timezone.utc))
    forecast_timestamp: Optional[datetime] = None
    points: List[ForecastPoint] = Field(default_factory=list)
    point_forecasts: Dict[str, ForecastPoint] = Field(default_factory=dict)
    scenarios: Dict[str, Any] = Field(default_factory=dict)
    scenario_references: Dict[str, Any] = Field(default_factory=dict)
    quality_status: str = "HIGH"
    data_quality_status: DataQualityStatus = DataQualityStatus.OK
    data_quality_warnings: List[str] = Field(default_factory=list)
    is_fallback_bundle: bool = False
    model_versions: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    evaluation_metadata: Dict[str, Any] = Field(default_factory=dict)
    is_synthetic: bool = False
    source_status: ServiceStatus = ServiceStatus.SYNTHETIC_DEMO

    @model_validator(mode="before")
    @classmethod
    def sync_bundle_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            pts = data.get("points")
            pt_dict = data.get("point_forecasts")

            if pts and not pt_dict:
                built_dict: Dict[str, Any] = {}
                for p in pts:
                    h_key = p.horizon if hasattr(p, "horizon") else (p.get("horizon") if isinstance(p, dict) else f"{getattr(p, 'horizon_hours', 1)}h")
                    built_dict[str(h_key)] = p
                data["point_forecasts"] = built_dict
            elif pt_dict and not pts:
                data["points"] = list(pt_dict.values())

            # Scenarios
            sc = data.get("scenarios") or data.get("scenario_references") or {}
            data["scenarios"] = sc
            data["scenario_references"] = sc

            # Quality
            qs = data.get("quality_status") or getattr(data.get("data_quality_status"), "value", None) or "OK"
            data["quality_status"] = str(qs)

            # Metadata
            meta = data.get("metadata") or data.get("evaluation_metadata") or {}
            data["metadata"] = meta
            data["evaluation_metadata"] = meta

            # Timestamp
            ts = data.get("timestamp") or data.get("forecast_timestamp") or datetime.now(timezone.utc)
            data["timestamp"] = ts
            if isinstance(ts, datetime):
                data["forecast_timestamp"] = ts
            elif isinstance(ts, str):
                data["forecast_timestamp"] = ensure_utc(ts)
        return data

    def get_point(self, horizon_hours: int) -> Optional[ForecastPoint]:
        for pt in self.points:
            if pt.horizon_hours == horizon_hours:
                return pt
        h_str = f"{horizon_hours}h"
        return self.point_forecasts.get(h_str)


class RiskSignal(CanonicalBaseModel):
    signal_id: str = Field(default_factory=lambda: f"sig-{uuid.uuid4().hex[:8]}")
    source: str = ""
    category: str = "GENERAL_NEWS"
    publication_time: str = ""
    ingestion_time: str = ""
    severity: str = "LOW"
    confidence: float = 1.0
    evidence: str = ""
    evidence_url: Optional[str] = None
    affected_markets: List[str] = Field(default_factory=list)
    scenario_tags: List[str] = Field(default_factory=list)
    corroboration_status: str = "UNVERIFIED"
    sentiment_score: float = 0.0
    is_duplicate: bool = False
    relevance_to_treasury: float = 0.0
    warnings: List[str] = Field(default_factory=list)
    timestamp: Union[datetime, str] = Field(default_factory=lambda: datetime.now(timezone.utc))
    supporting_evidence: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def sync_risk_signal_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Evidence
            ev = data.get("evidence") or str(data.get("supporting_evidence", ""))
            data["evidence"] = ev
            if "supporting_evidence" not in data or not data["supporting_evidence"]:
                data["supporting_evidence"] = {"evidence_text": ev}
            # Timestamps
            if "publication_time" not in data or not data["publication_time"]:
                data["publication_time"] = format_iso_utc(data.get("timestamp"))
            if "ingestion_time" not in data or not data["ingestion_time"]:
                data["ingestion_time"] = format_iso_utc(data.get("timestamp"))
        return data


class DecisionRecord(CanonicalBaseModel):
    decision_id: str = Field(default_factory=lambda: f"dec-{uuid.uuid4().hex[:12]}")
    timestamp: Union[datetime, str] = Field(default_factory=lambda: datetime.now(timezone.utc))
    selected_action: Union[str, Dict[str, Any]] = "HOLD_CASH"
    target_asset: str = "CASH"
    amount: float = 0.0
    reserve: Union[float, Dict[str, Any]] = 0.0
    eligible_surplus: float = 0.0
    estimated_costs: Dict[str, float] = Field(default_factory=dict)
    cost_breakdown: Dict[str, float] = Field(default_factory=dict)
    hard_gate_outcomes: List[HardGateOutcome] = Field(default_factory=list)
    hard_gate_results: Dict[str, Any] = Field(default_factory=dict)
    model_versions: Dict[str, str] = Field(default_factory=dict)
    reason_codes: List[str] = Field(default_factory=list)
    alternatives: List[Any] = Field(default_factory=list)
    treasury_safety_score: Optional[float] = None
    tss_factors: Dict[str, float] = Field(default_factory=dict)
    score: Dict[str, Any] = Field(default_factory=dict)
    approval_status: str = ApprovalStatus.PENDING_APPROVAL.value
    approved_by: Optional[str] = None
    approval_role: Optional[str] = None
    approval_timestamp: Optional[str] = None
    override_reason: Optional[str] = None
    audit_trail_id: str = ""
    is_synthetic: bool = False
    source_status: ServiceStatus = ServiceStatus.SYNTHETIC_DEMO

    @model_validator(mode="before")
    @classmethod
    def sync_decision_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Costs
            c = data.get("estimated_costs") or data.get("cost_breakdown") or {}
            data["estimated_costs"] = c
            data["cost_breakdown"] = c
            # Gates
            hg = data.get("hard_gate_outcomes") or []
            hgr = data.get("hard_gate_results") or {}
            data["hard_gate_outcomes"] = hg
            data["hard_gate_results"] = hgr
            # TSS
            tss = data.get("treasury_safety_score")
            if tss is not None and "score" in data and isinstance(data["score"], dict):
                data["score"]["treasury_safety_score"] = tss
        return data


class ExecutionOutcome(CanonicalBaseModel):
    outcome_id: str = Field(default_factory=lambda: f"out-{uuid.uuid4().hex[:12]}")
    decision_id: str = ""
    correlation_id: str = ""
    execution_timestamp: Union[datetime, str] = Field(default_factory=lambda: datetime.now(timezone.utc))
    simulated: bool = True
    realized_amount: float = 0.0
    realized_costs: Dict[str, float] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    liquidity_shortfall: float = 0.0
    settlement_status: str = "PENDING"
    slippage_bps: float = 0.0
    realized_cash_movements: float = 0.0
    costs: float = 0.0
    penalties: float = 0.0
    losses: float = 0.0
    liquidity_shortfalls: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LearningEvent(CanonicalBaseModel):
    event_id: str = Field(default_factory=lambda: f"evt-{uuid.uuid4().hex[:12]}")
    versioned_event_type: str = ""
    event_type: str = ""
    module: str = "CORE"
    source_module: str = ""
    timestamp: Union[datetime, str] = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)
    associated_versions: Dict[str, str] = Field(default_factory=dict)
    model_version: str = "v1.0"

    @model_validator(mode="before")
    @classmethod
    def sync_learning_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Event type
            et = data.get("versioned_event_type") or data.get("event_type") or "UNSPECIFIED"
            data["versioned_event_type"] = et
            data["event_type"] = et

            # Module
            mod = data.get("module") or data.get("source_module") or "CORE"
            data["module"] = mod
            data["source_module"] = mod

            # Versions
            vers = data.get("associated_versions") or {}
            mv = data.get("model_version") or "v1.0"
            if mv and "model_version" not in vers:
                vers["model_version"] = mv
            data["associated_versions"] = vers
            data["model_version"] = mv
        return data

    @property
    def structured_data(self) -> Dict[str, Any]:
        return self.payload

    @property
    def model_policy_versions(self) -> Dict[str, str]:
        return self.associated_versions


# =============================================================================
# 5. Event Publishing Infrastructure
# =============================================================================

class EventPublisher(abc.ABC):
    """Abstract event publisher interface."""

    @abc.abstractmethod
    def publish(self, event: LearningEvent) -> None:
        """Publish a learning or telemetry event."""
        pass


class NoOpEventPublisher(EventPublisher):
    """Silent event publisher for testing or headless execution."""

    def publish(self, event: LearningEvent) -> None:
        pass


class InMemoryEventPublisher(EventPublisher):
    """Stores events in-memory for testing, auditing, and inspection."""

    def __init__(self) -> None:
        self.events: List[LearningEvent] = []
        self._subscribers: List[Callable[[LearningEvent], None]] = []

    def subscribe(self, callback: Callable[[LearningEvent], None]) -> None:
        """Register a callback for published events."""
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
        return [e for e in self.events if e.event_type == event_type or e.versioned_event_type == event_type]
