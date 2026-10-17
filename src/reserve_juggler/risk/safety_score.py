"""
Treasury Safety Score (TSS) calculation engine.

Computes a configurable composite score from 0 to 100 reflecting the safety,
resilience, and liquidity adequacy of the treasury.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from reserve_juggler.decision.protocols import (
        AssetPosition,
        ForecastBundle,
        Obligation,
        RiskSignal,
        TreasuryState,
    )
from reserve_juggler.risk.models import (
    SafetyMode,
    SafetyScoreResult,
    TSSComponents,
    TSSWeights,
)


def _get_immediate_obligations(
    obligations: Sequence[Obligation], current_time: datetime, window_hours: float = 24.0
) -> float:
    """Sum obligations due within the immediate window."""
    total = 0.0
    for ob in obligations:
        hours_diff = (ob.due_time - current_time).total_seconds() / 3600.0
        if 0.0 <= hours_diff <= window_hours:
            total += float(ob.amount)
        elif hours_diff < 0.0:
            # Overdue obligation - must count immediately
            total += float(ob.amount)
    return total


def calculate_treasury_safety_score(
    state: TreasuryState,
    forecast_bundle: ForecastBundle,
    risk_signals: Optional[Sequence[RiskSignal]] = None,
    weights: Optional[TSSWeights] = None,
) -> SafetyScoreResult:
    """Calculate the composite Treasury Safety Score (TSS) from 0 to 100.

    Components:
    1. Liquidity Coverage (30%):
       Measures immediately available cash relative to immediate obligations and
       near-term conservative net outflows.
    2. Stress Survival (20%):
       Evaluates whether liquidatable asset value after severe haircuts and stress
       covers worst-case cumulative outflows.
    3. Forecast Reliability (15%):
       Reflects forecast bundle data quality, model freshness, and quantile spread.
    4. Time-to-Liquidity Alignment (15%):
       Evaluates asset maturity and settlement timing against obligation schedules.
    5. Forced-Liquidation Exposure (10%):
       Penalizes dependencies on liquidating high-haircut, high-penalty assets.
    6. Market-Risk Safety (10%):
       Assesses bond duration and equity volatility risk across asset positions.

    Circuit-Breaker Principle:
    A high composite score must never mask an acute immediate cash shortfall.
    If available cash + authorized credit cannot cover immediate obligations, the
    score is capped strictly into the Critical mode (<= 35.0).
    """
    weights = weights or TSSWeights()
    risk_signals = risk_signals or []
    warnings: list[str] = list(state.data_quality_warnings or [])

    now = state.timestamp
    available_cash = max(0.0, float(state.available_cash))
    credit_avail = max(0.0, float(state.credit_availability))
    total_immediate_funding = available_cash + credit_avail

    # 1. Immediate Obligations and Outflows
    immediate_obligations = _get_immediate_obligations(state.future_obligations, now, window_hours=24.0)

    # Calculate near-term conservative net outflow from forecast points (0-24h)
    near_term_conservative_outflow = 0.0
    if forecast_bundle.point_forecasts:
        for fp in forecast_bundle.point_forecasts:
            if float(fp.forecast_horizon) <= 24.0:
                # Conservative net outflow = withdrawal P90 - deposit P10
                pt_outflow = max(0.0, float(fp.withdrawal_p90) - float(fp.deposit_p10))
                near_term_conservative_outflow += pt_outflow

    total_immediate_demand = immediate_obligations + near_term_conservative_outflow
    shortfall_detected = total_immediate_funding < immediate_obligations

    # Component 1: Liquidity Coverage (0 to 100)
    # Ratio of available liquid cash to immediate demand
    if total_immediate_demand <= 0.0:
        c_liquidity = 100.0
    else:
        # A ratio of 1.5x or higher receives 100; 1.0x receives ~70; 0.0x receives 0.
        coverage_ratio = available_cash / total_immediate_demand
        if coverage_ratio >= 1.5:
            c_liquidity = 100.0
        elif coverage_ratio >= 1.0:
            c_liquidity = 70.0 + (coverage_ratio - 1.0) * 60.0
        else:
            c_liquidity = max(0.0, coverage_ratio * 70.0)

    # Component 2: Stress Survival (0 to 100)
    # Total stressed liquidatable value vs severe stress outflow
    total_stress_liquidatable = available_cash + credit_avail
    for pos in state.asset_positions:
        # Stressed liquidation value with additional 10% market crash buffer
        stress_haircut = min(1.0, float(pos.haircut) + (0.15 if pos.tier == 3 else 0.05))
        stress_val = float(pos.market_value) * (1.0 - stress_haircut) * (1.0 - float(pos.exit_fees))
        total_stress_liquidatable += max(0.0, stress_val)

    # Severe stress demand (P90 outflows + 25% stress spike + obligations)
    severe_stress_demand = (immediate_obligations * 1.1) + (near_term_conservative_outflow * 1.25)
    if severe_stress_demand <= 0.0:
        c_stress = 100.0
    else:
        stress_ratio = total_stress_liquidatable / severe_stress_demand
        if stress_ratio >= 2.0:
            c_stress = 100.0
        elif stress_ratio >= 1.0:
            c_stress = 60.0 + (stress_ratio - 1.0) * 40.0
        else:
            c_stress = max(0.0, stress_ratio * 60.0)

    # Component 3: Forecast Reliability (0 to 100)
    # Evaluates quality status and risk signals
    quality = forecast_bundle.quality_status.upper()
    if quality == "HEALTHY":
        base_reliability = 95.0
    elif quality == "DEGRADED":
        base_reliability = 65.0
        warnings.append("Forecast bundle is DEGRADED.")
    elif quality == "STALE":
        base_reliability = 30.0
        warnings.append("Forecast bundle is STALE; using conservative adjustments.")
    else:  # CRITICAL or UNKNOWN
        base_reliability = 10.0
        warnings.append("Forecast bundle is CRITICAL/UNRELIABLE.")

    # Deduct for high/critical risk signals
    signal_penalty = 0.0
    for sig in risk_signals:
        sev = sig.severity.upper()
        if sev == "CRITICAL":
            signal_penalty += 30.0 * float(sig.confidence)
        elif sev == "HIGH":
            signal_penalty += 15.0 * float(sig.confidence)
        elif sev == "MEDIUM":
            signal_penalty += 5.0 * float(sig.confidence)

    c_forecast = max(0.0, min(100.0, base_reliability - signal_penalty))

    # Component 4: Time-to-Liquidity Alignment (0 to 100)
    # Checks whether non-cash assets mature before or after obligations
    mismatched_amount = 0.0
    total_non_cash = 0.0
    for pos in state.asset_positions:
        val = float(pos.market_value)
        total_non_cash += val
        delay = float(pos.settlement_delay_hours)
        if pos.maturity:
            hours_to_mat = max(0.0, (pos.maturity - now).total_seconds() / 3600.0)
            delay = max(delay, hours_to_mat)
        # If asset takes longer than 24 hours to be accessible while obligations exist
        if delay > 24.0 and immediate_obligations > available_cash:
            mismatched_amount += val

    if total_non_cash <= 0.0:
        c_timing = 100.0
    else:
        mismatch_ratio = min(1.0, mismatched_amount / total_non_cash)
        c_timing = max(0.0, 100.0 * (1.0 - mismatch_ratio))

    # Component 5: Forced-Liquidation Exposure (0 to 100)
    # Measures the penalty / loss that would be incurred if all non-cash assets were liquidated
    total_liquidation_loss = 0.0
    total_asset_mv = sum(float(p.market_value) for p in state.asset_positions)
    for pos in state.asset_positions:
        loss_rate = float(pos.haircut) + float(pos.exit_fees)
        total_liquidation_loss += float(pos.market_value) * loss_rate

    if total_asset_mv <= 0.0:
        c_forced = 100.0
    else:
        avg_loss_rate = total_liquidation_loss / total_asset_mv
        # 0% loss = 100, 10% loss = 50, >=20% loss = 0
        c_forced = max(0.0, min(100.0, 100.0 - (avg_loss_rate * 500.0)))

    # Component 6: Market-Risk Safety (0 to 100)
    # Measures duration exposure and equity volatility
    duration_risk = 0.0
    equity_risk = 0.0
    total_portfolio_val = available_cash + total_asset_mv

    if total_portfolio_val > 0.0:
        for pos in state.asset_positions:
            pos_weight = float(pos.market_value) / total_portfolio_val
            if pos.tier == 2:  # Bonds: duration risk
                duration_risk += pos_weight * float(pos.duration) * 10.0
            elif pos.tier == 3:  # Equities: volatility risk
                equity_risk += pos_weight * float(pos.volatility) * 100.0

    c_market = max(0.0, min(100.0, 100.0 - (duration_risk + equity_risk)))

    # Assemble components
    components = TSSComponents(
        liquidity_coverage=round(c_liquidity, 2),
        stress_survival=round(c_stress, 2),
        forecast_reliability=round(c_forecast, 2),
        time_to_liquidity=round(c_timing, 2),
        forced_liquidation_exposure=round(c_forced, 2),
        market_risk_safety=round(c_market, 2),
    )

    # Weighted composite score
    raw_composite = (
        weights.liquidity_coverage * components.liquidity_coverage
        + weights.stress_survival * components.stress_survival
        + weights.forecast_reliability * components.forecast_reliability
        + weights.time_to_liquidity * components.time_to_liquidity
        + weights.forced_liquidation_exposure * components.forced_liquidation_exposure
        + weights.market_risk_safety * components.market_risk_safety
    )

    # Critical circuit-breaker:
    # If cash is insufficient to meet immediate mandatory obligations, score cannot exceed 35.0
    if shortfall_detected:
        warnings.append("CRITICAL: Available cash is below immediate payout obligations! Score is capped.")
        composite_score = min(35.0, raw_composite)
    elif quality in ("STALE", "CRITICAL"):
        # Conservative ceiling when forecasts are unreliable
        composite_score = min(55.0, raw_composite)
    else:
        composite_score = raw_composite

    composite_score = max(0.0, min(100.0, composite_score))

    # Map to operational mode
    if composite_score < 40.0:
        mode = SafetyMode.CRITICAL
    elif composite_score < 60.0:
        mode = SafetyMode.DEFENSIVE
    elif composite_score < 80.0:
        mode = SafetyMode.BALANCED
    else:
        mode = SafetyMode.OPTIMIZING

    return SafetyScoreResult(
        composite_score=round(composite_score, 2),
        mode=mode,
        components=components,
        warnings=warnings,
        shortfall_detected=shortfall_detected,
        is_calibrated_probability=False,
    )
