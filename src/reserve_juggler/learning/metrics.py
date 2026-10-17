"""
System-Wide Performance Evaluation and Drift Detection.
Calculates statistically grounded error metrics, true empirical prediction-interval coverage,
KS-test drift diagnostics, hard-gate pass rates, and honest sentinel reporting.
"""

from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional
from scipy.stats import ks_2samp

from reserve_juggler.learning.models import (
    DecisionPerformanceMetrics,
    DriftAnalysisResult,
    ForecastPerformanceMetrics,
    MatchedDecisionOutcome,
    MatchedForecastOutcome,
    SentinelPerformanceMetrics,
    SystemPerformanceReport,
)
from reserve_juggler.learning.store import EventStore


class SystemPerformanceEvaluator:
    """
    Evaluates system performance across the entire treasury lifecycle.
    Adheres strictly to the requirement that insufficient data reports as 'INCONCLUSIVE'
    rather than fabricating synthetic improvement.
    """

    MIN_SAMPLES_FOR_EVALUATION = 3
    MIN_SAMPLES_FOR_DRIFT = 10

    def __init__(self, event_store: EventStore):
        self.store = event_store

    def evaluate(
        self,
        matched_decisions: List[MatchedDecisionOutcome],
        matched_forecasts: Optional[List[MatchedForecastOutcome]] = None,
        start_iso: Optional[str] = None,
        end_iso: Optional[str] = None,
        active_versions: Optional[Dict[str, str]] = None,
    ) -> SystemPerformanceReport:
        now_iso = datetime.now(timezone.utc).isoformat()
        start = start_iso or (matched_decisions[0].timestamp if matched_decisions else now_iso)
        end = end_iso or now_iso
        versions = active_versions or {
            "forecasting_model": "v1.4.0",
            "decision_optimizer": "v1.2.0",
            "sentinel_risk": "v1.0.0",
            "reserve_policy": "v2.0.0",
        }

        # 1. Filter by time range
        if start_iso or end_iso:
            if start_iso:
                matched_decisions = [d for d in matched_decisions if d.timestamp >= start_iso]
            if end_iso:
                matched_decisions = [d for d in matched_decisions if d.timestamp <= end_iso]

        if matched_forecasts and (start_iso or end_iso):
            if start_iso:
                matched_forecasts = [f for f in matched_forecasts if f.target_timestamp >= start_iso]
            if end_iso:
                matched_forecasts = [f for f in matched_forecasts if f.target_timestamp <= end_iso]

        # 2. Forecast Evaluation
        eligible_forecasts = [f for f in (matched_forecasts or []) if f.is_eligible and f.actual_net_flow is not None]
        total_predictions = len(matched_forecasts or [])
        valid_evaluations = len(eligible_forecasts)
        excluded_count = total_predictions - valid_evaluations
        exclusion_reasons = []
        if excluded_count > 0:
            exclusion_reasons.append(f"{excluded_count} forecast horizons pending actual settlement observations.")

        if valid_evaluations < self.MIN_SAMPLES_FOR_EVALUATION:
            f_metrics = ForecastPerformanceMetrics(
                total_predictions=total_predictions,
                valid_evaluations=valid_evaluations,
                excluded_count=excluded_count,
                exclusion_reasons=exclusion_reasons or [f"Insufficient evaluation sample size ({valid_evaluations} < {self.MIN_SAMPLES_FOR_EVALUATION})."],
                mape=None,
                wape=None,
                mae=None,
                rmse=None,
                interval_coverage_pct=None,
                nominal_confidence=0.90,
                drift_analysis=DriftAnalysisResult(
                    drift_detected=False,
                    statistic=0.0,
                    p_value=1.0,
                    method="two_sample_ks_test",
                    threshold=0.05,
                    sample_size_reference=0,
                    sample_size_test=0,
                    status="INCONCLUSIVE",
                    details=f"Insufficient evaluation sample size ({valid_evaluations} < {self.MIN_SAMPLES_FOR_DRIFT}) for statistical drift analysis.",
                ),
                status="INCONCLUSIVE",
            )
        else:
            # Errors for actual cash flow net
            errors = [abs(f.actual_net_flow - f.predicted_net_flow) for f in eligible_forecasts]  # type: ignore
            actuals = [abs(f.actual_net_flow) for f in eligible_forecasts]  # type: ignore
            mae = sum(errors) / len(errors)
            rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
            sum_actuals = sum(actuals)
            if sum_actuals > 0:
                wape = sum(errors) / sum_actuals * 100.0
            else:
                wape = None
                exclusion_reasons.append("Actual net flows sum to zero; WAPE undefined (zero denominator).")

            # MAPE with zero-actual guard
            non_zero_pairs = [(e, a) for e, a in zip(errors, actuals) if a >= 1000.0]
            if non_zero_pairs:
                mape = sum(e / a * 100.0 for e, a in non_zero_pairs) / len(non_zero_pairs)
            else:
                mape = None
                exclusion_reasons.append("Actual net flows near zero; MAPE undefined, WAPE and MAE reported.")

            # Empirical Interval Coverage
            forecasts_with_intervals = [f for f in eligible_forecasts if f.within_90_interval is not None]
            if forecasts_with_intervals:
                covered = sum(1 for f in forecasts_with_intervals if f.within_90_interval is True)
                coverage_pct = (covered / len(forecasts_with_intervals)) * 100.0
            else:
                coverage_pct = None

            # Kolmogorov-Smirnov Drift Test
            if valid_evaluations >= self.MIN_SAMPLES_FOR_DRIFT:
                mid = valid_evaluations // 2
                ref_errors = errors[:mid]
                test_errors = errors[mid:]
                ks_stat, ks_pval = ks_2samp(ref_errors, test_errors)
                drift_flag = bool(ks_pval < 0.05)
                drift_analysis = DriftAnalysisResult(
                    drift_detected=drift_flag,
                    statistic=round(float(ks_stat), 4),
                    p_value=round(float(ks_pval), 4),
                    method="two_sample_ks_test",
                    threshold=0.05,
                    sample_size_reference=len(ref_errors),
                    sample_size_test=len(test_errors),
                    status="DRIFT_CONFIRMED" if drift_flag else "STABLE",
                    details=f"Two-sample KS test (stat={ks_stat:.3f}, p={ks_pval:.3f}) across reference and test error distributions.",
                )
            else:
                drift_analysis = DriftAnalysisResult(
                    drift_detected=False,
                    statistic=0.0,
                    p_value=1.0,
                    method="two_sample_ks_test",
                    threshold=0.05,
                    sample_size_reference=0,
                    sample_size_test=0,
                    status="INCONCLUSIVE",
                    details=f"Sample size ({valid_evaluations} < {self.MIN_SAMPLES_FOR_DRIFT}) insufficient for conclusive KS test.",
                )

            f_metrics = ForecastPerformanceMetrics(
                total_predictions=total_predictions,
                valid_evaluations=valid_evaluations,
                excluded_count=excluded_count,
                exclusion_reasons=exclusion_reasons,
                mape=round(mape, 2) if mape is not None else None,
                wape=round(wape, 2) if wape is not None else None,
                mae=round(mae, 2),
                rmse=round(rmse, 2),
                interval_coverage_pct=round(coverage_pct, 1) if coverage_pct is not None else None,
                nominal_confidence=0.90,
                drift_analysis=drift_analysis,
                status="DEGRADED" if drift_analysis.drift_detected else "STABLE",
            )

        # 3. Decision Evaluation
        total_decisions = len(matched_decisions)
        executed = [d for d in matched_decisions if d.is_executed]
        cancelled_or_rejected = [d for d in matched_decisions if d.execution_status in ("CANCELLED", "REJECTED")]
        pending_or_unresolved = [d for d in matched_decisions if not d.is_executed and d.execution_status not in ("CANCELLED", "REJECTED")]

        shortfall_decisions = [d for d in executed if d.is_shortfall]
        total_shortfalls = len(shortfall_decisions)
        total_shortfall_amt = sum(d.liquidity_shortfall for d in shortfall_decisions)

        slippages = [d.cost_error_bps for d in executed]
        avg_slippage = (sum(slippages) / len(slippages)) if slippages else 0.0

        # Hard-Gate Pass Rate from actual records
        total_gates = 0
        passed_gates = 0
        for d in matched_decisions:
            for g in d.hard_gates:
                total_gates += 1
                if g.get("status") in ("PASS", True):
                    passed_gates += 1

        if total_gates > 0:
            gate_pass_rate = passed_gates / total_gates
            gate_status = "PASS" if gate_pass_rate == 1.0 else "FAIL"
        else:
            gate_pass_rate = 0.0
            gate_status = "UNKNOWN"

        # Human approvals and overrides
        approval_events = self.store.get_by_module("HUMAN_OVERSIGHT")
        if start_iso:
            approval_events = [e for e in approval_events if e.timestamp >= start_iso]
        if end_iso:
            approval_events = [e for e in approval_events if e.timestamp <= end_iso]

        tot_human = len(approval_events)
        if tot_human > 0:
            overrides = sum(
                1 for e in approval_events if e.payload.get("status") in ("OVERRIDDEN", "MODIFIED", "REJECTED", "CANCELLED")
            )
            approvals = sum(1 for e in approval_events if e.payload.get("status") == "APPROVED")
            override_rate = round(overrides / tot_human, 2)
            approval_rate = round(approvals / tot_human, 2)
        else:
            override_rate = None
            approval_rate = None

        d_metrics = DecisionPerformanceMetrics(
            total_decisions=total_decisions,
            executed_decisions=len(executed),
            cancelled_or_rejected=len(cancelled_or_rejected),
            pending_or_unresolved=len(pending_or_unresolved),
            avg_cost_slippage_bps=round(avg_slippage, 1),
            total_shortfalls=total_shortfalls,
            total_shortfall_amount=round(total_shortfall_amt, 2),
            hard_gate_pass_rate=round(gate_pass_rate, 3),
            hard_gate_status=gate_status,
            human_override_rate=override_rate,
            human_approval_rate=approval_rate,
        )

        # 4. Sentinel Evaluation
        sentinel_events = self.store.get_by_module("SENTINEL")
        if start_iso:
            sentinel_events = [e for e in sentinel_events if e.timestamp >= start_iso]
        if end_iso:
            sentinel_events = [e for e in sentinel_events if e.timestamp <= end_iso]

        tot_sig = len(sentinel_events)
        corroborated_sig = sum(
            1 for e in sentinel_events if e.payload.get("corroboration_status") in ("CORROBORATED", "OFFICIAL")
        )
        corrob_rate = round(corroborated_sig / tot_sig, 2) if tot_sig > 0 else None

        sev_counts: Dict[str, int] = {}
        for e in sentinel_events:
            sev = e.payload.get("severity", "UNKNOWN")
            sev_counts[sev] = sev_counts.get(sev, 0) + 1

        # Check for ground-truth labelled signals (Part D2)
        labelled_sentinel = [e for e in sentinel_events if "ground_truth_label" in e.payload]
        if labelled_sentinel:
            fp_count = sum(1 for e in labelled_sentinel if e.payload.get("ground_truth_label") == "FALSE_ALARM")
            tp_count = sum(1 for e in labelled_sentinel if e.payload.get("ground_truth_label") == "REAL_EVENT")
            tot_lab = len(labelled_sentinel)
            fa_rate = round(fp_count / tot_lab, 2) if tot_lab > 0 else None
            prec = round(tp_count / (tp_count + fp_count), 2) if (tp_count + fp_count) > 0 else None
            prec_status = "CALCULATED_FROM_LABELS"
            fa_status = "CALCULATED_FROM_LABELS"
        else:
            fa_rate = None
            prec = None
            prec_status = "UNAVAILABLE_WITHOUT_GROUND_TRUTH_LABELS"
            fa_status = "UNAVAILABLE_WITHOUT_GROUND_TRUTH_LABELS"

        s_metrics = SentinelPerformanceMetrics(
            total_signals=tot_sig,
            corroboration_rate=corrob_rate,
            false_alarm_rate=fa_rate,
            precision_status=prec_status,
            precision=prec,
            alert_counts_by_severity=sev_counts,
            false_alarm_status=fa_status,
        )

        # 5. Overall Health Synthesis (Part D3)
        if total_shortfalls > 0:
            health = "ACTION_REQUIRED"
        elif (f_metrics.drift_analysis and f_metrics.drift_analysis.drift_detected) or avg_slippage > 50.0:
            health = "WARNING"
        elif f_metrics.status == "INCONCLUSIVE" or total_decisions < self.MIN_SAMPLES_FOR_EVALUATION:
            health = "INCONCLUSIVE"
        else:
            health = "HEALTHY"

        inconclusive_reason = None
        if f_metrics.status == "INCONCLUSIVE" or total_decisions < self.MIN_SAMPLES_FOR_EVALUATION:
            inconclusive_reason = (
                f"Evaluation sample size ({total_decisions} decisions, {valid_evaluations} forecasts) is below minimal statistical threshold ({self.MIN_SAMPLES_FOR_EVALUATION})."
            )

        return SystemPerformanceReport(
            start_time=start,
            end_time=end,
            total_learning_events=len(self.store.get_all()),
            forecast_metrics=f_metrics,
            decision_metrics=d_metrics,
            sentinel_metrics=s_metrics,
            overall_system_health=health,
            active_versions=versions,
            inconclusive_reason=inconclusive_reason,
        )


def detect_distribution_drift(
    ref_errors: List[float],
    test_errors: List[float],
    threshold: float = 0.05,
) -> DriftAnalysisResult:
    """
    Executes a two-sample Kolmogorov-Smirnov test to detect distribution drift
    in forecast errors. Requires valid sample sets.
    """
    if len(ref_errors) < 3 or len(test_errors) < 3:
        return DriftAnalysisResult(
            drift_detected=False,
            statistic=0.0,
            p_value=1.0,
            method="two_sample_ks_test",
            threshold=threshold,
            sample_size_reference=len(ref_errors),
            sample_size_test=len(test_errors),
            status="INCONCLUSIVE",
            details="Insufficient samples for two-sample KS test.",
        )

    ks_stat, ks_pval = ks_2samp(ref_errors, test_errors)
    drift_flag = bool(ks_pval < threshold)
    return DriftAnalysisResult(
        drift_detected=drift_flag,
        statistic=round(float(ks_stat), 4),
        p_value=round(float(ks_pval), 4),
        method="two_sample_ks_test",
        threshold=threshold,
        sample_size_reference=len(ref_errors),
        sample_size_test=len(test_errors),
        status="DRIFT_CONFIRMED" if drift_flag else "STABLE",
        details=f"Two-sample KS test (stat={ks_stat:.3f}, p={ks_pval:.3f}) across reference and test error distributions.",
    )


def calculate_forecast_metrics(
    matched_forecasts: List[MatchedForecastOutcome],
    nominal_confidence: float = 0.90,
) -> ForecastPerformanceMetrics:
    """
    Calculates forecast metrics from valid matched outcomes.
    Explicitly guards against zero/near-zero actuals and reports WAPE & MAE.
    """
    total_predictions = len(matched_forecasts)
    eligible = [f for f in matched_forecasts if f.is_eligible and f.actual_outflow is not None]
    valid_evaluations = len(eligible)
    excluded_count = total_predictions - valid_evaluations
    exclusion_reasons: List[str] = []

    if valid_evaluations == 0:
        return ForecastPerformanceMetrics(
            total_predictions=total_predictions,
            valid_evaluations=0,
            excluded_count=excluded_count,
            exclusion_reasons=exclusion_reasons or ["No eligible forecast outcomes observed."],
            mape=None,
            wape=None,
            mae=None,
            rmse=None,
            interval_coverage_pct=None,
            nominal_confidence=nominal_confidence,
            drift_analysis=DriftAnalysisResult(
                drift_detected=False,
                statistic=0.0,
                p_value=1.0,
                method="two_sample_ks_test",
                threshold=0.05,
                sample_size_reference=0,
                sample_size_test=0,
                status="INCONCLUSIVE",
                details="No forecast outcomes available for drift analysis.",
            ),
            status="INCONCLUSIVE",
        )

    errors = [abs(f.actual_outflow - f.predicted_outflow) for f in eligible]  # type: ignore
    actuals = [abs(f.actual_outflow) for f in eligible]  # type: ignore
    mae = sum(errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    sum_actuals = sum(actuals)
    if sum_actuals > 0:
        wape = sum(errors) / sum_actuals * 100.0
    else:
        wape = 100.0 if sum(errors) > 0 else 0.0
        exclusion_reasons.append("Zero actual outflow across all observations; WAPE bounded at 100.0%.")

    # Guard zero actuals
    non_zero = [(e, a) for e, a in zip(errors, actuals) if a > 1e-6]
    if non_zero:
        mape = sum(e / a * 100.0 for e, a in non_zero) / len(non_zero)
        if len(non_zero) < len(errors):
            exclusion_reasons.append(f"Zero actual outflow in {len(errors) - len(non_zero)} observations excluded from MAPE.")
            excluded_count += (len(errors) - len(non_zero))
    else:
        mape = None
        exclusion_reasons.append("Zero actual outflow across all observations; MAPE undefined, WAPE and MAE reported.")
        excluded_count = valid_evaluations

    # Interval coverage only where genuine prediction intervals are available
    forecasts_with_intervals = [f for f in eligible if f.within_90_interval is not None]
    if forecasts_with_intervals:
        covered = sum(1 for f in forecasts_with_intervals if f.within_90_interval is True)
        coverage_pct = (covered / len(forecasts_with_intervals)) * 100.0
    else:
        coverage_pct = None

    # Drift
    if valid_evaluations >= 10:
        mid = valid_evaluations // 2
        drift_analysis = detect_distribution_drift(errors[:mid], errors[mid:])
    else:
        drift_analysis = DriftAnalysisResult(
            drift_detected=False,
            statistic=0.0,
            p_value=1.0,
            method="two_sample_ks_test",
            threshold=0.05,
            sample_size_reference=0,
            sample_size_test=0,
            status="INCONCLUSIVE",
            details=f"Insufficient sample size ({valid_evaluations} < 10) for statistical drift analysis.",
        )

    status = "STABLE" if not drift_analysis.drift_detected else "DEGRADED"

    return ForecastPerformanceMetrics(
        total_predictions=total_predictions,
        valid_evaluations=valid_evaluations,
        excluded_count=excluded_count,
        exclusion_reasons=exclusion_reasons,
        mape=mape,
        wape=round(wape, 2) if wape is not None else None,
        mae=round(mae, 2),
        rmse=round(rmse, 2),
        interval_coverage_pct=round(coverage_pct, 2) if coverage_pct is not None else None,
        nominal_confidence=nominal_confidence,
        drift_analysis=drift_analysis,
        status=status,
    )
