"""
Data models and state representations for System-Wide Self-Learning.
Separates forecast matching from decision evaluation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class CandidateStatus(str, Enum):
    PROPOSED = "PROPOSED"
    EVALUATING = "EVALUATING"
    SHADOW_MODE = "SHADOW_MODE"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"


class SubsystemName(str, Enum):
    FORECASTING = "FORECASTING"
    DECISION_OPTIMIZER = "DECISION_OPTIMIZER"
    SENTINEL_RISK = "SENTINEL_RISK"
    SAFETY_TSS = "SAFETY_TSS"
    EXPLAINABILITY = "EXPLAINABILITY"


@dataclass
class MatchedForecastOutcome:
    """Matches a forecast point to actual observed customer cash flows."""
    correlation_id: str
    forecast_id: str
    target_timestamp: str
    horizon: str
    predicted_net_flow: float
    predicted_inflow: float
    predicted_outflow: float
    uncertainty_p05: float
    uncertainty_p50: float
    uncertainty_p90: float
    uncertainty_p95: float
    uncertainty_p99: float
    model_version: str
    actual_net_flow: Optional[float] = None
    actual_inflow: Optional[float] = None
    actual_outflow: Optional[float] = None
    is_eligible: bool = True
    ineligibility_reason: Optional[str] = None
    abs_error: Optional[float] = None
    within_90_interval: Optional[bool] = None


@dataclass
class MatchedDecisionOutcome:
    """Matches an approved and executed decision to realized financial outcomes."""
    correlation_id: str
    decision_id: str
    timestamp: str
    selected_action: str
    target_asset: str
    proposed_amount: float
    approval_status: str
    is_executed: bool
    execution_status: str
    expected_cost: float
    realized_cost: float
    cost_error: float = 0.0
    cost_error_bps: float = 0.0
    liquidity_shortfall: float = 0.0
    is_shortfall: bool = False
    hard_gates: List[Dict[str, Any]] = field(default_factory=list)


# Backward compatibility alias
MatchedOutcomePair = MatchedDecisionOutcome


@dataclass
class DriftAnalysisResult:
    drift_detected: bool
    statistic: float
    p_value: float
    method: str  # "two_sample_ks_test"
    threshold: float
    sample_size_reference: int
    sample_size_test: int
    status: str  # "STABLE", "DRIFT_CONFIRMED", "INCONCLUSIVE"
    details: str


@dataclass
class ForecastPerformanceMetrics:
    total_predictions: int
    valid_evaluations: int
    excluded_count: int
    exclusion_reasons: List[str]
    mape: Optional[float]  # None if all actuals are 0
    wape: float  # Weighted Absolute Percentage Error
    mae: float
    rmse: float
    interval_coverage_pct: float  # Percent of actuals inside [p05, p95]
    nominal_confidence: float  # 0.90 for 90%
    drift_analysis: DriftAnalysisResult
    status: str  # "STABLE", "DEGRADED", "INCONCLUSIVE"

    @property
    def drift_detected(self) -> bool:
        return self.drift_analysis.drift_detected if self.drift_analysis else False


@dataclass
class DecisionPerformanceMetrics:
    total_decisions: int
    executed_decisions: int
    cancelled_or_rejected: int
    avg_cost_slippage_bps: float
    total_shortfalls: int
    total_shortfall_amount: float
    hard_gate_pass_rate: float
    hard_gate_status: str  # "PASS", "FAIL", "UNKNOWN", "INCONCLUSIVE"
    human_override_rate: float
    human_approval_rate: float


@dataclass
class SentinelPerformanceMetrics:
    total_signals: int
    corroboration_rate: float
    false_alarm_rate: float
    precision_status: str  # "UNAVAILABLE_WITHOUT_GROUND_TRUTH" or value
    precision: Optional[float]
    alert_counts_by_severity: Dict[str, int]


@dataclass
class SystemPerformanceReport:
    start_time: str
    end_time: str
    total_learning_events: int
    forecast_metrics: ForecastPerformanceMetrics
    decision_metrics: DecisionPerformanceMetrics
    sentinel_metrics: SentinelPerformanceMetrics
    overall_system_health: str  # "EXCELLENT", "HEALTHY", "WARNING", "ACTION_REQUIRED", "INCONCLUSIVE"
    active_versions: Dict[str, str]
    inconclusive_reason: Optional[str] = None


@dataclass
class CandidateValidationReport:
    candidate_id: str
    component: str
    validation_dataset: str
    validation_sample_size: int
    champion_metric: float
    candidate_metric: float
    metric_name: str
    metric_improved: bool
    stress_shortfall_count: int
    safety_invariants_passed: bool
    violations: List[str]
    is_promotable: bool
    runtime_integrated: bool = True


@dataclass
class CandidateUpdate:
    candidate_id: str
    component: str
    parameter_key: str
    current_value: Any
    proposed_value: Any
    description: str
    author: str
    created_at: str
    status: CandidateStatus = CandidateStatus.PROPOSED
    validation_report: Optional[CandidateValidationReport] = None
    holdout_evaluation_score: float = 0.0
    holdout_sample_size: int = 0
    stress_shortfall_count: int = 0
    safety_check_passed: bool = False
    safety_violations: List[str] = field(default_factory=list)
    shadow_mode_runs: int = 0
    shadow_win_rate: float = 0.0
    promoted_at: Optional[str] = None
    promoted_by: Optional[str] = None
    promotion_role: Optional[str] = None
    rejection_reason: Optional[str] = None
    is_metadata_only: bool = False
