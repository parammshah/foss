"""
Concrete test fixtures and mock generators conforming to Section 3 Protocols.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Sequence

from reserve_juggler.decision.protocols import (
    AssetPosition,
    ExecutionOutcome,
    ForecastBundle,
    ForecastPoint,
    Obligation,
    RiskSignal,
    TreasuryState,
)


@dataclass
class ConcreteObligation:
    """Concrete obligation implementation."""

    due_time: datetime
    amount: float
    description: str = "Mandatory customer payout"
    is_mandatory: bool = True


@dataclass
class ConcreteAssetPosition:
    """Concrete asset position implementation."""

    asset_id: str
    tier: int  # 1: Cash/Repo, 2: Sovereign Bonds, 3: Equities
    market_value: float
    liquidatable_value: float
    availability: datetime
    maturity: Optional[datetime] = None
    settlement_delay_hours: float = 0.0
    annual_yield: float = 0.048  # Decimal rate (4.8%)
    exit_fees: float = 0.0  # 0.5% early exit fee = 0.005
    haircut: float = 0.0  # 2.0% haircut = 0.02
    duration: float = 0.0  # Duration in years
    volatility: float = 0.0  # Annualized volatility
    cost_basis: float = 0.0
    units: float = 0.0
    entry_price: float = 0.0
    purchase_timestamp: Optional[datetime] = None
    instrument_type: str = "repo"  # 'cash', 'repo', 'bond', 'equity'


@dataclass
class ConcreteForecastPoint:
    """Concrete forecast point implementation."""

    forecast_horizon: float
    deposit_p10: float
    deposit_p50: float
    deposit_p90: float
    withdrawal_p10: float
    withdrawal_p50: float
    withdrawal_p90: float
    conservative_outflow: float = 0.0
    reliability_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.conservative_outflow == 0.0:
            self.conservative_outflow = max(0.0, self.withdrawal_p90 - self.deposit_p10)


@dataclass
class ConcreteForecastBundle:
    """Concrete forecast bundle implementation."""

    point_forecasts: Sequence[ForecastPoint]
    scenarios: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    quality_status: str = "HEALTHY"
    model_versions: dict[str, str] = field(
        default_factory=lambda: {"flow_forecast_model": "1.2.0"}
    )


@dataclass
class ConcreteRiskSignal:
    """Concrete risk signal implementation."""

    source: str
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    confidence: float  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    supporting_evidence: dict[str, Any] = field(default_factory=dict)
    scenario_tags: list[str] = field(default_factory=list)


@dataclass
class ConcreteExecutionOutcome:
    """Concrete execution outcome implementation."""

    realized_cash_movements: float
    costs: float
    penalties: float
    losses: float
    liquidity_shortfalls: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConcreteTreasuryState:
    """Concrete treasury state implementation."""

    timestamp: datetime
    available_cash: float
    asset_positions: list[AssetPosition] = field(default_factory=list)
    future_obligations: list[Obligation] = field(default_factory=list)
    credit_availability: float = 5000000.0  # $5M authorized credit facility
    credit_limit: float = 5000000.0
    credit_apr: float = 0.12  # 12.0% APR
    data_quality_warnings: list[str] = field(default_factory=list)


# -------------------------------------------------------------
# Test State Factory Generators
# -------------------------------------------------------------


def make_healthy_forecast_bundle(
    now: datetime, hours: int = 72
) -> ConcreteForecastBundle:
    """Create healthy multi-horizon forecast bundle."""
    points = []
    for h in [1, 2, 4, 8, 12, 24, 48, 72]:
        points.append(
            ConcreteForecastPoint(
                forecast_horizon=float(h),
                deposit_p10=200000.0 * (h / 24.0),
                deposit_p50=500000.0 * (h / 24.0),
                deposit_p90=800000.0 * (h / 24.0),
                withdrawal_p10=150000.0 * (h / 24.0),
                withdrawal_p50=400000.0 * (h / 24.0),
                withdrawal_p90=700000.0 * (h / 24.0),
                reliability_metadata={"r_squared": 0.94},
            )
        )
    return ConcreteForecastBundle(point_forecasts=points, timestamp=now, quality_status="HEALTHY")


def make_benchmark_repo_bridge_state(
    now: datetime, credit_available: float = 15000000.0
) -> ConcreteTreasuryState:
    """Create state matching Section 7 Benchmark Example:

    - Immediate customer payout obligation of $10,000,000 due in 1 hour.
    - Available cash: $0.
    - $10,000,000 repo position unlocking in 2 hours (exit fee 0.5% = $50k).
    - Bridge credit facility: $15,000,000 authorized at 12% APR.
    """
    repo_pos = ConcreteAssetPosition(
        asset_id="REPO_10M_BENCHMARK",
        tier=1,
        market_value=10000000.0,
        liquidatable_value=9950000.0,
        availability=now + timedelta(hours=2),
        maturity=now + timedelta(hours=2),
        settlement_delay_hours=0.0,
        annual_yield=0.048,
        exit_fees=0.005,  # 0.5% early break fee = $50,000
        haircut=0.0,
        duration=0.0,
        volatility=0.0,
        cost_basis=10000000.0,
        instrument_type="repo",
    )

    obligation = ConcreteObligation(
        due_time=now + timedelta(hours=1),
        amount=10000000.0,
        description="Mandatory Customer Settlement Batch 1",
        is_mandatory=True,
    )

    return ConcreteTreasuryState(
        timestamp=now,
        available_cash=0.0,  # Zero cash - funding deficit
        asset_positions=[repo_pos],
        future_obligations=[obligation],
        credit_availability=credit_available,
        credit_limit=15000000.0,
        credit_apr=0.12,  # 12% APR
        data_quality_warnings=[],
    )
