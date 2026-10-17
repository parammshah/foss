"""
Domain models for Treasury Safety Score, Capital Firewall, and Risk Management.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class SafetyMode(str, Enum):
    """Operational mode mapped from Treasury Safety Score."""

    CRITICAL = "Critical"  # 0 - 39
    DEFENSIVE = "Defensive"  # 40 - 59
    BALANCED = "Balanced"  # 60 - 79
    OPTIMIZING = "Optimizing"  # 80 - 100


@dataclass(frozen=True)
class TSSWeights:
    """Configurable weights for the Treasury Safety Score composite."""

    liquidity_coverage: float = 0.30
    stress_survival: float = 0.20
    forecast_reliability: float = 0.15
    time_to_liquidity: float = 0.15
    forced_liquidation_exposure: float = 0.10
    market_risk_safety: float = 0.10

    def __post_init__(self) -> None:
        total = (
            self.liquidity_coverage
            + self.stress_survival
            + self.forecast_reliability
            + self.time_to_liquidity
            + self.forced_liquidation_exposure
            + self.market_risk_safety
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"TSS weights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class TSSComponents:
    """Component breakdown of the Treasury Safety Score (each 0 - 100)."""

    liquidity_coverage: float
    stress_survival: float
    forecast_reliability: float
    time_to_liquidity: float
    forced_liquidation_exposure: float
    market_risk_safety: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SafetyScoreResult:
    """Result of calculate_treasury_safety_score."""

    composite_score: float  # 0 to 100
    mode: SafetyMode
    components: TSSComponents
    warnings: list[str] = field(default_factory=list)
    shortfall_detected: bool = False
    is_calibrated_probability: bool = False  # Explicit disclaimer

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite_score": round(self.composite_score, 2),
            "mode": self.mode.value,
            "components": self.components.to_dict(),
            "warnings": list(self.warnings),
            "shortfall_detected": self.shortfall_detected,
            "is_calibrated_probability": self.is_calibrated_probability,
        }


@dataclass(frozen=True)
class ReserveAssessment:
    """Result of assess_liquidity_reserve."""

    protected_capital: float
    future_obligation_reserve: float
    eligible_investable_surplus: float
    immediate_obligations: float
    operational_safety_buffer: float
    conservative_net_deficit: float
    maturing_repo_liquidity: float
    uncovered_stress_scenarios: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protected_capital": round(self.protected_capital, 2),
            "future_obligation_reserve": round(self.future_obligation_reserve, 2),
            "eligible_investable_surplus": round(self.eligible_investable_surplus, 2),
            "immediate_obligations": round(self.immediate_obligations, 2),
            "operational_safety_buffer": round(self.operational_safety_buffer, 2),
            "conservative_net_deficit": round(self.conservative_net_deficit, 2),
            "maturing_repo_liquidity": round(self.maturing_repo_liquidity, 2),
            "uncovered_stress_scenarios": list(self.uncovered_stress_scenarios),
        }


@dataclass(frozen=True)
class HardGateEvaluation:
    """Result of evaluate_hard_safety_gates."""

    passed: bool
    gates: dict[str, bool]
    violated_gates: list[str]
    reason_codes: list[str]
    uncovered_scenarios: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "gates": dict(self.gates),
            "violated_gates": list(self.violated_gates),
            "reason_codes": list(self.reason_codes),
            "uncovered_scenarios": list(self.uncovered_scenarios),
        }


@dataclass(frozen=True)
class CapitalClassification:
    """Result of classify_hot_core_capital."""

    hot_money: float
    core_capital: float
    investable_core_capital: float
    total_treasury_assets: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "hot_money": round(self.hot_money, 2),
            "core_capital": round(self.core_capital, 2),
            "investable_core_capital": round(self.investable_core_capital, 2),
            "total_treasury_assets": round(self.total_treasury_assets, 2),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class HoldingTimeGateResult:
    """Result of calculate_holding_time_gate."""

    passed: bool
    asset_id: str
    instrument_type: str
    minimum_holding_days: float
    expected_holding_days: float
    haircut: float
    annual_yield: float
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "asset_id": self.asset_id,
            "instrument_type": self.instrument_type,
            "minimum_holding_days": round(self.minimum_holding_days, 2),
            "expected_holding_days": round(self.expected_holding_days, 2),
            "haircut": self.haircut,
            "annual_yield": self.annual_yield,
            "notes": self.notes,
        }
