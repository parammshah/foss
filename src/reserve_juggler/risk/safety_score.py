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
       Reflects forecast bundle data quality, model freshness (age), quantile spread,
       and forecast-specific risk signals. Market signals are isolated from forecast failure.
    4. Time-to-Liquidity Alignment (15%):
       Evaluates asset maturity and settlement timing against obligation schedules.
    5. Forced-Liquidation Exposure (10%):
       Penalizes dependencies on liquidating high-haircut, high-penalty assets.
    6. Market-Risk Safety (10%):
       Assesses bond duration, equity volatility risk, and market risk signals.

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
    total_quantile_spread = 0.0
    point_count_24h = 0

    if forecast_bundle.point_forecasts:
        for fp in forecast_bundle.point_forecasts:
            if float(fp.forecast_horizon) <= 24.0:
                pt_outflow = max(0.0, float(fp.withdrawal_p90) - float(fp.deposit_p10))
                near_term_conservative_outflow += pt_outflow

                # Quantile spread: P90 - P10 normalized by P50
                w_p50 = max(1.0, float(fp.withdrawal_p50))
                w_spread = max(0.0, float(fp.withdrawal_p90) - float(fp.withdrawal_p10)) / w_p50
                total_quantile_spread += w_spread
                point_count_24h += 1

    total_immediate_demand = immediate_obligations + near_term_conservative_outflow
    shortfall_detected = total_immediate_funding < immediate_obligations

    # Component 1: Liquidity Coverage (0 to 100)
    if total_immediate_demand <= 0.0:
        c_liquidity = 100.0
    else:
        coverage_ratio = available_cash / total_immediate_demand
        if coverage_ratio >= 1.5:
            c_liquidity = 100.0
        elif coverage_ratio >= 1.0:
            c_liquidity = 70.0 + (coverage_ratio - 1.0) * 60.0
        else:
            c_liquidity = max(0.0, coverage_ratio * 70.0)

    # Component 2: Stress Survival (0 to 100)
    total_stress_liquidatable = available_cash + credit_avail
    for pos in state.asset_positions:
        stress_haircut = min(1.0, float(pos.haircut) + (0.15 if pos.tier == 3 else 0.05))
        stress_val = float(pos.market_value) * (1.0 - stress_haircut) * (1.0 - float(pos.exit_fees))
        total_stress_liquidatable += max(0.0, stress_val)

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
    # Uses quality_status, forecast age, quantile spread, and forecast-specific signals
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

    # Forecast age penalty if bundle timestamp is aged
    try:
        age_hours = max(0.0, (now - forecast_bundle.timestamp).total_seconds() / 3600.0)
        if age_hours > 24.0:
            age_penalty = min(30.0, (age_hours - 24.0) * 1.2)
            base_reliability -= age_penalty
            warnings.append(f"Forecast age is {age_hours:.1f}h; applied age penalty of {age_penalty:.1f} pts.")
    except Exception:
        pass

    # Quantile spread penalty: wide spreads indicate prediction uncertainty
    if point_count_24h > 0:
        avg_spread = total_quantile_spread / point_count_24h
        # A normal spread ratio is ~0.5 to 1.0; spread > 1.5 triggers penalty
        if avg_spread > 1.5:
            spread_penalty = min(20.0, (avg_spread - 1.5) * 10.0)
            base_reliability -= spread_penalty

    # Deduct specifically for forecasting/pipeline signals (market signals penalize market-risk component)
    forecast_signal_penalty = 0.0
    market_signal_penalty = 0.0

    for sig in risk_signals:
        src = sig.source.upper()
        sev = sig.severity.upper()
        conf = float(sig.confidence)

        # Distinguish market risk signals from forecasting model failures
        is_market_signal = any(
            tag in src or tag in " ".join(sig.scenario_tags or []).upper()
            for tag in ["MARKET", "RATE", "YIELD", "VOLATILITY", "EQUITY", "CRASH"]
        )

        penalty = 30.0 * conf if sev == "CRITICAL" else (15.0 * conf if sev == "HIGH" else 5.0 * conf)

        if is_market_signal:
            market_signal_penalty += penalty
        else:
            forecast_signal_penalty += penalty

    c_forecast = max(0.0, min(100.0, base_reliability - forecast_signal_penalty))

    # Component 4: Time-to-Liquidity Alignment (0 to 100)
    mismatched_amount = 0.0
    total_non_cash = 0.0
    for pos in state.asset_positions:
        val = float(pos.market_value)
        total_non_cash += val
        delay = float(pos.settlement_delay_hours)
        if pos.maturity:
            hours_to_mat = max(0.0, (pos.maturity - now).total_seconds() / 3600.0)
            delay = max(delay, hours_to_mat)
        if delay > 24.0 and immediate_obligations > available_cash:
            mismatched_amount += val

    if total_non_cash <= 0.0:
        c_timing = 100.0
    else:
        mismatch_ratio = min(1.0, mismatched_amount / total_non_cash)
        c_timing = max(0.0, 100.0 * (1.0 - mismatch_ratio))

    # Component 5: Forced-Liquidation Exposure (0 to 100)
    total_liquidation_loss = 0.0
    total_asset_mv = sum(float(p.market_value) for p in state.asset_positions)
    for pos in state.asset_positions:
        loss_rate = float(pos.haircut) + float(pos.exit_fees)
        total_liquidation_loss += float(pos.market_value) * loss_rate

    if total_asset_mv <= 0.0:
        c_forced = 100.0
    else:
        avg_loss_rate = total_liquidation_loss / total_asset_mv
        c_forced = max(0.0, min(100.0, 100.0 - (avg_loss_rate * 500.0)))

    # Component 6: Market-Risk Safety (0 to 100)
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

    c_market = max(0.0, min(100.0, 100.0 - (duration_risk + equity_risk) - market_signal_penalty))

    components = TSSComponents(
        liquidity_coverage=round(c_liquidity, 2),
        stress_survival=round(c_stress, 2),
        forecast_reliability=round(c_forecast, 2),
        time_to_liquidity=round(c_timing, 2),
        forced_liquidation_exposure=round(c_forced, 2),
        market_risk_safety=round(c_market, 2),
    )

    raw_composite = (
        weights.liquidity_coverage * components.liquidity_coverage
        + weights.stress_survival * components.stress_survival
        + weights.forecast_reliability * components.forecast_reliability
        + weights.time_to_liquidity * components.time_to_liquidity
        + weights.forced_liquidation_exposure * components.forced_liquidation_exposure
        + weights.market_risk_safety * components.market_risk_safety
    )

    # Circuit breaker: if cash is insufficient to meet immediate mandatory obligations,
    # score cannot exceed 35.0 (Critical mode)
    if shortfall_detected:
        warnings.append("CRITICAL: Available cash is below immediate payout obligations! Score is capped.")
        composite_score = min(35.0, raw_composite)
    elif quality in ("STALE", "CRITICAL"):
        composite_score = min(55.0, raw_composite)
    else:
        composite_score = raw_composite

    composite_score = max(0.0, min(100.0, composite_score))

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
