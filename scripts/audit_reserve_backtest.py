#!/usr/bin/env python3
"""Auditable evaluation script reproducing row-level reserve backtest results,
peak-outflow coverage, actual decision paths, and moving-block-bootstrap statistical intervals.

Outputs:
1. artifacts/reserve_backtest_row_level.csv (600 rows: 200 rolling origins x 3 horizons)
2. artifacts/reserve_backtest_daily_expanded.csv (963 rows: 321 daily origins x 3 horizons)
"""

from __future__ import annotations

import os
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from reserve_juggler.data import resolve_dataset_path
from reserve_juggler.data.schema import DEPOSITS_COL, TIMESTAMP_COL, WITHDRAWALS_COL
from reserve_juggler.forecasting.service import ForecastService
from reserve_juggler.risk.liquidity_shield import assess_liquidity_reserve
from reserve_juggler.contracts import TreasuryState
from reserve_juggler.decision.service import TreasuryDecisionService
from reserve_juggler.tax.tax_profile import create_statutory_us_corp_profile


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute exact Wilson score confidence interval strictly bounded in [0, 1].
    Identified strictly as a descriptive reference interval under an i.i.d. Bernoulli assumption.
    """
    if n == 0:
        return 0.0, 0.0
    z = 1.95996 if confidence == 0.95 else 2.57583
    p = k / n
    denom = 1.0 + (z**2 / n)
    center = (p + (z**2 / (2.0 * n))) / denom
    half_width = (z * np.sqrt((p * (1.0 - p) + (z**2 / (4.0 * n))) / n)) / denom
    return max(0.0, float(center - half_width)), min(1.0, float(center + half_width))


def moving_block_bootstrap_ci(
    indicators: np.ndarray,
    block_length: int,
    num_bootstraps: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, any]:
    """Compute Moving Block Bootstrap (MBB) empirical percentile confidence interval
    accounting for serial correlation across consecutive time steps.

    If the sample has zero observed failures (k_failures = 0), standard empirical bootstrap
    is degenerate (resampling all-successes yields 100% in every replicate, sample variance=0).
    In this case, this function explicitly flags degeneracy.
    """
    n = len(indicators)
    k_success = int(np.sum(indicators))
    k_failures = n - k_success

    alpha = 1.0 - confidence

    # Degenerate Case: 0 observed failures
    if k_failures == 0:
        p_failure_upper = 1.0 - (alpha ** (1.0 / n))
        coverage_lower_bound = 1.0 - p_failure_upper
        return {
            "lower": float(coverage_lower_bound),
            "upper": 1.0,
            "is_degenerate": True,
            "display": "Degenerate (0 failures)",
            "rule_of_three_bound": float(coverage_lower_bound),
            "failure_rate_upper": float(p_failure_upper),
        }

    b = max(1, min(block_length, n // 2))
    num_blocks = int(np.ceil(n / b))
    blocks = [indicators[i : i + b] for i in range(n - b + 1)]
    if not blocks:
        mean_val = float(np.mean(indicators))
        return {
            "lower": mean_val,
            "upper": mean_val,
            "is_degenerate": True,
            "display": f"[{mean_val*100:.1f}%, {mean_val*100:.1f}%]",
            "rule_of_three_bound": mean_val,
            "failure_rate_upper": 1.0 - mean_val,
        }

    rng = np.random.default_rng(seed)
    boot_means = np.empty(num_bootstraps)
    for i in range(num_bootstraps):
        chosen_indices = rng.integers(0, len(blocks), size=num_blocks)
        resampled = np.concatenate([blocks[idx] for idx in chosen_indices])[:n]
        boot_means[i] = np.mean(resampled)

    lower = max(0.0, float(np.percentile(boot_means, (alpha / 2.0) * 100.0)))
    upper = min(1.0, float(np.percentile(boot_means, (1.0 - (alpha / 2.0)) * 100.0)))
    return {
        "lower": lower,
        "upper": upper,
        "is_degenerate": False,
        "display": f"[{lower * 100:.1f}%, {upper * 100:.1f}%]",
        "rule_of_three_bound": lower,
        "failure_rate_upper": 1.0 - lower,
    }


def evaluate_block_sensitivity(
    indicators: np.ndarray,
    candidate_blocks: list[int],
    seed: int = 42,
    num_bootstraps: int = 2000,
) -> dict[int, dict[str, any]]:
    """Evaluate bootstrap interval sensitivity across varying block lengths."""
    results = {}
    for b in candidate_blocks:
        results[b] = moving_block_bootstrap_ci(
            indicators, block_length=b, num_bootstraps=num_bootstraps, seed=seed
        )
    return results


def main() -> int:
    dataset_path = resolve_dataset_path()
    with open(dataset_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    print("=" * 155)
    print("RESERVE JUGGLER AUDITABLE BACKTEST — DATASET INTEGRITY & PROVENANCE")
    print(f"Dataset Path : {dataset_path}")
    print(f"SHA-256 Hash : {file_hash}")
    print("=" * 155)

    df = pd.read_csv(dataset_path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    total_dataset_rows = len(df)

    # Strict training/calibration partition on first 1000 rows
    N_FIT = 1000
    VAL_RATIO = 0.25
    train_df = df.iloc[:N_FIT].copy()
    svc = ForecastService()
    svc.fit(train_df, horizons=[1, 6, 24, 48, 72], val_split_ratio=VAL_RATIO)

    print(f"Fitted ForecastService: {N_FIT} total rows ({N_FIT * (1 - VAL_RATIO):.0f} train, {N_FIT * VAL_RATIO:.0f} val)")
    print(f"Total rows in dataset: {total_dataset_rows} (rows 0..{total_dataset_rows - 1})")
    print(f"Available holdout pool: rows {N_FIT}..{total_dataset_rows - 1} ({total_dataset_rows - N_FIT} rows)")
    print(f"Evaluation origins strictly begin at row {N_FIT} ({df.iloc[N_FIT][TIMESTAMP_COL]})\n")

    # Baseline operational cash for canonical execution simulation
    INITIAL_OP_CASH = 65_000_000.0
    dec_svc = TreasuryDecisionService()
    tax_prof = create_statutory_us_corp_profile()

    # -------------------------------------------------------------------------
    # PART A: ROLLING-ORIGIN BACKTEST (N=200, origins 1000..1199, step=1h)
    # -------------------------------------------------------------------------
    EVAL_ORIGINS = 200
    horizons = [6, 24, 72]
    rolling_records = []

    start_origin_idx = N_FIT
    end_origin_idx = N_FIT + EVAL_ORIGINS - 1
    start_ts = df.iloc[start_origin_idx][TIMESTAMP_COL]
    end_ts = df.iloc[end_origin_idx][TIMESTAMP_COL]

    print(f"Rolling-Origin Window: Rows {start_origin_idx}..{end_origin_idx} (N={EVAL_ORIGINS} origins)")
    print(f"Start Origin Timestamp: {start_ts.isoformat()}")
    print(f"End Origin Timestamp  : {end_ts.isoformat()}")
    print(f"Max Target Timestamp  : {(end_ts + pd.Timedelta(hours=72)).isoformat()} (Row {end_origin_idx + 72})\n")

    for h in horizons:
        for idx in range(N_FIT, N_FIT + EVAL_ORIGINS):
            as_of_ts = df.iloc[idx][TIMESTAMP_COL]
            bundle = svc.forecast(as_of_ts)
            fp = bundle.get_point(h)

            act_pt = float(df.iloc[idx + h][WITHDRAWALS_COL] - df.iloc[idx + h][DEPOSITS_COL])
            cum_series = (df.iloc[idx + 1 : idx + h + 1][WITHDRAWALS_COL] - df.iloc[idx + 1 : idx + h + 1][DEPOSITS_COL]).cumsum()
            realized_terminal_flow = float(cum_series.iloc[-1])
            realized_peak_flow = float(cum_series.max())

            raw_pt_p95 = float(getattr(fp, "net_outflow_calibrated_p95", 0.0) or 0.0)
            robust_pt_p95 = float(fp.net_outflow_p95)
            raw_cum_p95 = float(getattr(fp, "cumulative_net_outflow_p95", 0.0) or 0.0)
            applied_buffer = robust_pt_p95 - raw_pt_p95

            # 1. Horizon-specific component assessment (isolated component test)
            state_comp = TreasuryState(
                timestamp=as_of_ts,
                operational_cash_usd=100_000_000.0,
                obligations=[],
            )
            assessment_comp = assess_liquidity_reserve(
                state=state_comp,
                forecast_bundle=bundle,
                use_calibrated_net_quantile=True,
                payout_window_hours=float(h),
                forecast_horizon_hours=float(h),
            )
            comp_cons_deficit = float(assessment_comp.conservative_net_deficit)
            comp_tot_reserve = float(assessment_comp.total_required_reserve)

            # 2. Canonical complete multi-horizon production assessment (payout=1h, horizon=72h)
            state_canon = TreasuryState(
                timestamp=as_of_ts,
                operational_cash_usd=INITIAL_OP_CASH,
                obligations=[],
            )
            assessment_canon = assess_liquidity_reserve(
                state=state_canon,
                forecast_bundle=bundle,
                use_calibrated_net_quantile=True,
                payout_window_hours=1.0,
                forecast_horizon_hours=72.0,
            )
            canon_tot_reserve = float(assessment_canon.total_required_reserve)
            canon_surplus = float(assessment_canon.eligible_investable_surplus)

            # 3. Actual Decision Logic Invocation (in simulation mode)
            actual_rec = dec_svc.decide(state_canon, bundle, tax_profile=tax_prof)
            actual_action = actual_rec.action_type.value if hasattr(actual_rec.action_type, "value") else str(actual_rec.action_type)
            actual_amount = float(getattr(actual_rec, "amount", 0.0) or 0.0)
            actual_post_cash = INITIAL_OP_CASH - (actual_amount if actual_action == "PLACE_REPO" else 0.0)

            # Path-wise cash simulation under Actual Decision
            actual_shortfall = False
            for tau in range(1, h + 1):
                c_tau = float(cum_series.iloc[tau - 1])
                avail_act = (actual_post_cash if tau <= 15 else INITIAL_OP_CASH) if actual_action == "PLACE_REPO" else (INITIAL_OP_CASH - c_tau)
                if (avail_act if actual_action != "PLACE_REPO" else (avail_act - c_tau)) < 0:
                    actual_shortfall = True
                    break

            # 4. Counterfactual Scenario (assuming 100% of eligible surplus is placed into 15h repo)
            counterfactual_sweep = canon_surplus
            post_sweep_cash = INITIAL_OP_CASH - counterfactual_sweep

            first_shortfall_ts = ""
            lockup_shortfall = False
            counterfactual_shortfall = False

            for tau in range(1, h + 1):
                c_tau = float(cum_series.iloc[tau - 1])
                avail_c = (post_sweep_cash if tau <= 15 else INITIAL_OP_CASH) - c_tau
                if avail_c < 0:
                    if not first_shortfall_ts:
                        first_shortfall_ts = df.iloc[idx + tau][TIMESTAMP_COL].isoformat()
                    if tau <= 15:
                        lockup_shortfall = True
                    counterfactual_shortfall = True

            comp_term_exc = realized_terminal_flow > comp_cons_deficit
            comp_peak_exc = realized_peak_flow > comp_cons_deficit
            canon_term_exc = realized_terminal_flow > canon_tot_reserve
            canon_peak_exc = realized_peak_flow > canon_tot_reserve

            rolling_records.append({
                "forecast_origin_idx": idx,
                "as_of_timestamp": as_of_ts.isoformat(),
                "horizon_hours": h,
                "target_timestamp": (as_of_ts + pd.Timedelta(hours=h)).isoformat(),
                "actual_point_net_usd": act_pt,
                "realized_terminal_cumulative_net_outflow": realized_terminal_flow,
                "realized_peak_cumulative_net_outflow": realized_peak_flow,
                "raw_point_p95_usd": raw_pt_p95,
                "robust_point_p95_usd": robust_pt_p95,
                "raw_cumulative_p95_usd": raw_cum_p95,
                "applied_point_buffer_usd": applied_buffer,
                "component_conservative_deficit_usd": comp_cons_deficit,
                "component_total_reserve_usd": comp_tot_reserve,
                "canonical_required_reserve_usd": canon_tot_reserve,
                "canonical_eligible_surplus_usd": canon_surplus,
                "post_sweep_cleared_cash_usd": post_sweep_cash,
                "actual_decision_action": actual_action,
                "actual_decision_amount_usd": actual_amount,
                "actual_post_decision_cash_usd": actual_post_cash,
                "actual_decision_cash_shortfall": actual_shortfall,
                "counterfactual_cash_shortfall": counterfactual_shortfall,
                "counterfactual_lockup_shortfall_15h": lockup_shortfall,
                "cash_shortfall_timestamps": first_shortfall_ts,
                "reserve_vs_terminal_exceeded": canon_term_exc,
                "reserve_vs_peak_exceeded": canon_peak_exc,
                "canonical_reserve_vs_terminal_exceeded": canon_term_exc,
                "canonical_reserve_vs_peak_exceeded": canon_peak_exc,
                "component_reserve_vs_terminal_exceeded": comp_term_exc,
                "component_reserve_vs_peak_exceeded": comp_peak_exc,
                "exceeded_raw_point": act_pt > raw_pt_p95,
                "exceeded_robust_point": act_pt > robust_pt_p95,
                "exceeded_raw_cumulative": realized_terminal_flow > raw_cum_p95,
                "exceeded_component_reserve": comp_term_exc,
                "exceeded_canonical_reserve": canon_term_exc,
                "point_reliability_status": fp.point_reliability_status,
                "cumulative_reliability_status": fp.cumulative_reliability_status,
                "policy_reliability_status": assessment_comp.policy_reliability_status,
            })

    # Persist rolling artifact
    rolling_df = pd.DataFrame(rolling_records)
    artifact_path_rolling = Path("artifacts/reserve_backtest_row_level.csv")
    artifact_path_rolling.parent.mkdir(parents=True, exist_ok=True)
    rolling_df.to_csv(artifact_path_rolling, index=False)
    print(f"Saved {len(rolling_df)} row-level backtest evaluations to: {artifact_path_rolling}")

    # -------------------------------------------------------------------------
    # PART B: NON-OVERLAPPING ORIGINS (step=h)
    # -------------------------------------------------------------------------
    non_overlap_results = {}
    for h in [1, 6, 24, 72]:
        pt_cov = []
        cum_cov = []
        comp_term_cov = []
        comp_peak_cov = []
        canon_term_cov = []
        canon_peak_cov = []
        origins_used = []
        idx = N_FIT

        while idx + h < len(df):
            origins_used.append(idx)
            as_of_ts = df.iloc[idx][TIMESTAMP_COL]
            bundle = svc.forecast(as_of_ts)
            fp = bundle.get_point(h)

            act_pt = float(df.iloc[idx + h][WITHDRAWALS_COL] - df.iloc[idx + h][DEPOSITS_COL])
            cum_series = (df.iloc[idx + 1 : idx + h + 1][WITHDRAWALS_COL] - df.iloc[idx + 1 : idx + h + 1][DEPOSITS_COL]).cumsum()
            term_flow = float(cum_series.iloc[-1])
            peak_flow = float(cum_series.max())

            pt_cov.append(act_pt <= fp.net_outflow_p95)
            cum_cov.append(term_flow <= getattr(fp, "cumulative_net_outflow_p95", 0.0))

            # Isolated component reserve
            state_comp = TreasuryState(timestamp=as_of_ts, operational_cash_usd=100_000_000.0, obligations=[])
            assess_comp = assess_liquidity_reserve(
                state=state_comp,
                forecast_bundle=bundle,
                use_calibrated_net_quantile=True,
                payout_window_hours=float(h),
                forecast_horizon_hours=float(h),
            )
            comp_term_cov.append(term_flow <= assess_comp.conservative_net_deficit)
            comp_peak_cov.append(peak_flow <= assess_comp.conservative_net_deficit)

            # Canonical production reserve
            state_canon = TreasuryState(timestamp=as_of_ts, operational_cash_usd=INITIAL_OP_CASH, obligations=[])
            assess_canon = assess_liquidity_reserve(
                state=state_canon,
                forecast_bundle=bundle,
                use_calibrated_net_quantile=True,
                payout_window_hours=1.0,
                forecast_horizon_hours=72.0,
            )
            canon_term_cov.append(term_flow <= assess_canon.total_required_reserve)
            canon_peak_cov.append(peak_flow <= assess_canon.total_required_reserve)

            idx += h
            if h <= 24 and len(origins_used) >= 100:
                break

        n_w = len(origins_used)
        arr_pt_no = np.array(pt_cov, dtype=float)
        arr_cum_no = np.array(cum_cov, dtype=float)
        arr_comp_term = np.array(comp_term_cov, dtype=float)
        arr_comp_peak = np.array(comp_peak_cov, dtype=float)
        arr_canon_term = np.array(canon_term_cov, dtype=float)
        arr_canon_peak = np.array(canon_peak_cov, dtype=float)

        non_overlap_results[h] = {
            "n_windows": n_w,
            "first_origin": origins_used[0],
            "last_origin": origins_used[-1],
            "first_ts": df.iloc[origins_used[0]][TIMESTAMP_COL].isoformat(),
            "last_ts": df.iloc[origins_used[-1]][TIMESTAMP_COL].isoformat(),
            "last_target_ts": df.iloc[origins_used[-1] + h][TIMESTAMP_COL].isoformat(),
            "pt_cov": float(np.mean(pt_cov)),
            "pt_exc": n_w - sum(pt_cov),
            "pt_wilson": wilson_score_interval(sum(pt_cov), n_w),
            "pt_mbb": moving_block_bootstrap_ci(arr_pt_no, block_length=1),
            "cum_cov": float(np.mean(cum_cov)),
            "cum_exc": n_w - sum(cum_cov),
            "cum_wilson": wilson_score_interval(sum(cum_cov), n_w),
            "cum_mbb": moving_block_bootstrap_ci(arr_cum_no, block_length=1),
            "comp_term_cov": float(np.mean(comp_term_cov)),
            "comp_term_exc": n_w - sum(comp_term_cov),
            "comp_term_wilson": wilson_score_interval(sum(comp_term_cov), n_w),
            "comp_term_mbb": moving_block_bootstrap_ci(arr_comp_term, block_length=1),
            "comp_peak_cov": float(np.mean(comp_peak_cov)),
            "comp_peak_exc": n_w - sum(comp_peak_cov),
            "comp_peak_wilson": wilson_score_interval(sum(comp_peak_cov), n_w),
            "comp_peak_mbb": moving_block_bootstrap_ci(arr_comp_peak, block_length=1),
            "canon_term_cov": float(np.mean(canon_term_cov)),
            "canon_term_exc": n_w - sum(canon_term_cov),
            "canon_term_wilson": wilson_score_interval(sum(canon_term_cov), n_w),
            "canon_term_mbb": moving_block_bootstrap_ci(arr_canon_term, block_length=1),
            "canon_peak_cov": float(np.mean(canon_peak_cov)),
            "canon_peak_exc": n_w - sum(canon_peak_cov),
            "canon_peak_wilson": wilson_score_interval(sum(canon_peak_cov), n_w),
            "canon_peak_mbb": moving_block_bootstrap_ci(arr_canon_peak, block_length=1),
            "policy_status": "CONDITIONALLY_CALIBRATED_FALLBACK",
        }

    # -------------------------------------------------------------------------
    # PART C: EXPANDED FULL-HOLDOUT EVALUATION (Daily step=24h, N=321 origins)
    # -------------------------------------------------------------------------
    daily_origins = [idx for idx in range(N_FIT, len(df) - 72, 24)]
    expanded_records = []

    for h in [6, 24, 72]:
        for idx in daily_origins:
            as_of_ts = df.iloc[idx][TIMESTAMP_COL]
            bundle = svc.forecast(as_of_ts)
            fp = bundle.get_point(h)

            act_pt = float(df.iloc[idx + h][WITHDRAWALS_COL] - df.iloc[idx + h][DEPOSITS_COL])
            cum_series = (df.iloc[idx + 1 : idx + h + 1][WITHDRAWALS_COL] - df.iloc[idx + 1 : idx + h + 1][DEPOSITS_COL]).cumsum()
            realized_terminal_flow = float(cum_series.iloc[-1])
            realized_peak_flow = float(cum_series.max())

            raw_pt_p95 = float(getattr(fp, "net_outflow_calibrated_p95", 0.0) or 0.0)
            robust_pt_p95 = float(fp.net_outflow_p95)
            raw_cum_p95 = float(getattr(fp, "cumulative_net_outflow_p95", 0.0) or 0.0)
            applied_buffer = robust_pt_p95 - raw_pt_p95

            # 1. Component reserve
            state_comp = TreasuryState(timestamp=as_of_ts, operational_cash_usd=100_000_000.0, obligations=[])
            assess_comp = assess_liquidity_reserve(
                state=state_comp,
                forecast_bundle=bundle,
                use_calibrated_net_quantile=True,
                payout_window_hours=float(h),
                forecast_horizon_hours=float(h),
            )
            comp_cons_deficit = float(assess_comp.conservative_net_deficit)
            comp_tot_reserve = float(assess_comp.total_required_reserve)

            # 2. Canonical production reserve
            state_canon = TreasuryState(timestamp=as_of_ts, operational_cash_usd=INITIAL_OP_CASH, obligations=[])
            assess_canon = assess_liquidity_reserve(
                state=state_canon,
                forecast_bundle=bundle,
                use_calibrated_net_quantile=True,
                payout_window_hours=1.0,
                forecast_horizon_hours=72.0,
            )
            canon_tot_reserve = float(assess_canon.total_required_reserve)
            canon_surplus = float(assess_canon.eligible_investable_surplus)

            # 3. Actual Decision Logic Invocation (in simulation mode)
            actual_rec = dec_svc.decide(state_canon, bundle, tax_profile=tax_prof)
            actual_action = actual_rec.action_type.value if hasattr(actual_rec.action_type, "value") else str(actual_rec.action_type)
            actual_amount = float(getattr(actual_rec, "amount", 0.0) or 0.0)
            actual_post_cash = INITIAL_OP_CASH - (actual_amount if actual_action == "PLACE_REPO" else 0.0)

            actual_shortfall = False
            for tau in range(1, h + 1):
                c_tau = float(cum_series.iloc[tau - 1])
                avail_act = (actual_post_cash if tau <= 15 else INITIAL_OP_CASH) if actual_action == "PLACE_REPO" else (INITIAL_OP_CASH - c_tau)
                if (avail_act if actual_action != "PLACE_REPO" else (avail_act - c_tau)) < 0:
                    actual_shortfall = True
                    break

            # 4. Counterfactual Scenario (assuming 100% surplus is swept into 15h repo)
            counterfactual_sweep = canon_surplus
            post_sweep_cash = INITIAL_OP_CASH - counterfactual_sweep

            first_shortfall_ts = ""
            lockup_shortfall = False
            counterfactual_shortfall = False

            for tau in range(1, h + 1):
                c_tau = float(cum_series.iloc[tau - 1])
                avail_c = (post_sweep_cash if tau <= 15 else INITIAL_OP_CASH) - c_tau
                if avail_c < 0:
                    if not first_shortfall_ts:
                        first_shortfall_ts = df.iloc[idx + tau][TIMESTAMP_COL].isoformat()
                    if tau <= 15:
                        lockup_shortfall = True
                    counterfactual_shortfall = True

            comp_term_exc = realized_terminal_flow > comp_cons_deficit
            comp_peak_exc = realized_peak_flow > comp_cons_deficit
            canon_term_exc = realized_terminal_flow > canon_tot_reserve
            canon_peak_exc = realized_peak_flow > canon_tot_reserve

            expanded_records.append({
                "forecast_origin_idx": idx,
                "as_of_timestamp": as_of_ts.isoformat(),
                "horizon_hours": h,
                "target_timestamp": (as_of_ts + pd.Timedelta(hours=h)).isoformat(),
                "actual_point_net_usd": act_pt,
                "realized_terminal_cumulative_net_outflow": realized_terminal_flow,
                "realized_peak_cumulative_net_outflow": realized_peak_flow,
                "raw_point_p95_usd": raw_pt_p95,
                "robust_point_p95_usd": robust_pt_p95,
                "raw_cumulative_p95_usd": raw_cum_p95,
                "applied_point_buffer_usd": applied_buffer,
                "component_conservative_deficit_usd": comp_cons_deficit,
                "component_total_reserve_usd": comp_tot_reserve,
                "canonical_required_reserve_usd": canon_tot_reserve,
                "canonical_eligible_surplus_usd": canon_surplus,
                "post_sweep_cleared_cash_usd": post_sweep_cash,
                "actual_decision_action": actual_action,
                "actual_decision_amount_usd": actual_amount,
                "actual_post_decision_cash_usd": actual_post_cash,
                "actual_decision_cash_shortfall": actual_shortfall,
                "counterfactual_cash_shortfall": counterfactual_shortfall,
                "counterfactual_lockup_shortfall_15h": lockup_shortfall,
                "cash_shortfall_timestamps": first_shortfall_ts,
                "reserve_vs_terminal_exceeded": canon_term_exc,
                "reserve_vs_peak_exceeded": canon_peak_exc,
                "canonical_reserve_vs_terminal_exceeded": canon_term_exc,
                "canonical_reserve_vs_peak_exceeded": canon_peak_exc,
                "component_reserve_vs_terminal_exceeded": comp_term_exc,
                "component_reserve_vs_peak_exceeded": comp_peak_exc,
                "exceeded_raw_point": act_pt > raw_pt_p95,
                "exceeded_robust_point": act_pt > robust_pt_p95,
                "exceeded_raw_cumulative": realized_terminal_flow > raw_cum_p95,
                "exceeded_component_reserve": comp_term_exc,
                "exceeded_canonical_reserve": canon_term_exc,
                "point_reliability_status": fp.point_reliability_status,
                "cumulative_reliability_status": fp.cumulative_reliability_status,
                "policy_reliability_status": assess_comp.policy_reliability_status,
            })

    # Persist expanded artifact
    expanded_df = pd.DataFrame(expanded_records)
    artifact_path_expanded = Path("artifacts/reserve_backtest_daily_expanded.csv")
    expanded_df.to_csv(artifact_path_expanded, index=False)
    print(f"Saved {len(expanded_df)} expanded row-level evaluations to: {artifact_path_expanded}")

    # Copy to brain artifact directory if present
    brain_artifact_dir = Path("/Users/paramshah/.gemini/antigravity-ide/brain/4f700b38-a8e5-4609-a671-eefc75b6fe00")
    if brain_artifact_dir.exists():
        shutil.copy(artifact_path_rolling, brain_artifact_dir / "reserve_backtest_row_level.csv")
        shutil.copy(artifact_path_expanded, brain_artifact_dir / "reserve_backtest_daily_expanded.csv")
        print(f"Copied both artifacts to brain directory: {brain_artifact_dir}")

    # -------------------------------------------------------------------------
    # AUTOMATED INTEGRITY & RECONCILIATION ASSERTIONS
    # -------------------------------------------------------------------------
    assert len(rolling_df) == 600, f"Expected 600 records, got {len(rolling_df)}"
    assert len(rolling_df["forecast_origin_idx"].unique()) == 200, "Expected 200 unique origins"
    assert rolling_df["forecast_origin_idx"].min() == 1000
    assert rolling_df["forecast_origin_idx"].max() == 1199

    assert len(expanded_df) == 963, f"Expected 963 records (321 x 3), got {len(expanded_df)}"
    assert len(expanded_df["forecast_origin_idx"].unique()) == 321, "Expected 321 unique daily origins"
    assert expanded_df["forecast_origin_idx"].min() == 1000
    assert expanded_df["forecast_origin_idx"].max() == 8680
    assert (expanded_df["forecast_origin_idx"] + expanded_df["horizon_hours"]).max() <= 8759, "Horizon breached dataset length"

    # Verify peak is always >= terminal cumulative outflow
    assert (rolling_df["realized_peak_cumulative_net_outflow"] >= rolling_df["realized_terminal_cumulative_net_outflow"] - 1e-6).all(), "Peak outflow < terminal outflow in rolling"
    assert (expanded_df["realized_peak_cumulative_net_outflow"] >= expanded_df["realized_terminal_cumulative_net_outflow"] - 1e-6).all(), "Peak outflow < terminal outflow in expanded"

    # -------------------------------------------------------------------------
    # PRINT RESULTS TABLES DIRECTLY FROM PERSISTED ARTIFACT DATA
    # -------------------------------------------------------------------------
    print("\n" + "=" * 165)
    print("TABLE 1: ROLLING-ORIGIN BACKTEST (N=200, Origins 1000..1199, step=1h)")
    print("MBB configuration: B=2,000, seed=42. Block lengths in hourly observations: 6h=2 obs (2h), 24h=6 obs (6h), 72h=18 obs (18h)")
    print("=" * 165)
    print(f"{'Horizon':<8} | {'Metric / Scope':<42} | {'Cov%':<7} | {'Exc':<6} | {'Wilson 95% Ref CI':<19} | {'MBB 95% Bound':<26} | {'Zero-Failure Bound':<20} | {'Status'}")
    print("-" * 165)
    rolling_blocks = {6: 2, 24: 6, 72: 18}
    for h in horizons:
        sub = rolling_df[rolling_df["horizon_hours"] == h]
        n_sub = len(sub)
        b_roll = rolling_blocks[h]

        # 1. Robust point flow
        pt_exc = int(sub["exceeded_robust_point"].sum())
        pt_cov = (n_sub - pt_exc) / n_sub
        pt_w = wilson_score_interval(n_sub - pt_exc, n_sub)
        pt_mbb = moving_block_bootstrap_ci(np.array(~sub["exceeded_robust_point"], dtype=float), block_length=b_roll)
        pt_zf = f">={pt_mbb['rule_of_three_bound']*100:.1f}%" if pt_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | {'Point Flow Forecast (t+h)':<42} | {pt_cov*100:5.1f}% | {pt_exc:2d}/200 | [{pt_w[0]*100:.1f}%, {pt_w[1]*100:.1f}%]   | {pt_mbb['display']:<26} | {pt_zf:<20} | {sub['point_reliability_status'].iloc[0]}")

        # 2. Raw cumulative flow (Terminal)
        cum_exc = int(sub["exceeded_raw_cumulative"].sum())
        cum_cov = (n_sub - cum_exc) / n_sub
        cum_w = wilson_score_interval(n_sub - cum_exc, n_sub)
        cum_mbb = moving_block_bootstrap_ci(np.array(~sub["exceeded_raw_cumulative"], dtype=float), block_length=b_roll)
        cum_zf = f">={cum_mbb['rule_of_three_bound']*100:.1f}%" if cum_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | {'Raw Cumulative Forecast (t,t+h] [Terminal]':<42} | {cum_cov*100:5.1f}% | {cum_exc:2d}/200 | [{cum_w[0]*100:.1f}%, {cum_w[1]*100:.1f}%]   | {cum_mbb['display']:<26} | {cum_zf:<20} | {sub['cumulative_reliability_status'].iloc[0]}")

        # 3. Component Reserve vs Terminal
        comp_t_exc = int(sub["component_reserve_vs_terminal_exceeded"].sum())
        comp_t_cov = (n_sub - comp_t_exc) / n_sub
        comp_t_w = wilson_score_interval(n_sub - comp_t_exc, n_sub)
        comp_t_mbb = moving_block_bootstrap_ci(np.array(~sub["component_reserve_vs_terminal_exceeded"], dtype=float), block_length=b_roll)
        comp_t_zf = f">={comp_t_mbb['rule_of_three_bound']*100:.1f}%" if comp_t_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | {'Component Reserve [Terminal Outflow]':<42} | {comp_t_cov*100:5.1f}% | {comp_t_exc:2d}/200 | [{comp_t_w[0]*100:.1f}%, {comp_t_w[1]*100:.1f}%]   | {comp_t_mbb['display']:<26} | {comp_t_zf:<20} | {sub['policy_reliability_status'].iloc[0]}")

        # 4. Component Reserve vs Peak
        comp_p_exc = int(sub["component_reserve_vs_peak_exceeded"].sum())
        comp_p_cov = (n_sub - comp_p_exc) / n_sub
        comp_p_w = wilson_score_interval(n_sub - comp_p_exc, n_sub)
        comp_p_mbb = moving_block_bootstrap_ci(np.array(~sub["component_reserve_vs_peak_exceeded"], dtype=float), block_length=b_roll)
        comp_p_zf = f">={comp_p_mbb['rule_of_three_bound']*100:.1f}%" if comp_p_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | {'Component Reserve [Intrahorizon Peak]':<42} | {comp_p_cov*100:5.1f}% | {comp_p_exc:2d}/200 | [{comp_p_w[0]*100:.1f}%, {comp_p_w[1]*100:.1f}%]   | {comp_p_mbb['display']:<26} | {comp_p_zf:<20} | {sub['policy_reliability_status'].iloc[0]}")

        # 5. Canonical Multi-Horizon Reserve vs Terminal
        canon_t_exc = int(sub["canonical_reserve_vs_terminal_exceeded"].sum())
        canon_t_cov = (n_sub - canon_t_exc) / n_sub
        canon_t_w = wilson_score_interval(n_sub - canon_t_exc, n_sub)
        canon_t_mbb = moving_block_bootstrap_ci(np.array(~sub["canonical_reserve_vs_terminal_exceeded"], dtype=float), block_length=b_roll)
        canon_t_zf = f">={canon_t_mbb['rule_of_three_bound']*100:.1f}%" if canon_t_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | {'Canonical Reserve (72h) [Terminal Outflow]':<42} | {canon_t_cov*100:5.1f}% | {canon_t_exc:2d}/200 | [{canon_t_w[0]*100:.1f}%, {canon_t_w[1]*100:.1f}%]   | {canon_t_mbb['display']:<26} | {canon_t_zf:<20} | {sub['policy_reliability_status'].iloc[0]}")

        # 6. Canonical Multi-Horizon Reserve vs Peak
        canon_p_exc = int(sub["canonical_reserve_vs_peak_exceeded"].sum())
        canon_p_cov = (n_sub - canon_p_exc) / n_sub
        canon_p_w = wilson_score_interval(n_sub - canon_p_exc, n_sub)
        canon_p_mbb = moving_block_bootstrap_ci(np.array(~sub["canonical_reserve_vs_peak_exceeded"], dtype=float), block_length=b_roll)
        canon_p_zf = f">={canon_p_mbb['rule_of_three_bound']*100:.1f}%" if canon_p_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | {'Canonical Reserve (72h) [Intrahorizon Peak]':<42} | {canon_p_cov*100:5.1f}% | {canon_p_exc:2d}/200 | [{canon_p_w[0]*100:.1f}%, {canon_p_w[1]*100:.1f}%]   | {canon_p_mbb['display']:<26} | {canon_p_zf:<20} | {sub['policy_reliability_status'].iloc[0]}")

        # 7. Actual Decision Simulation Cash Path
        act_sf_cnt = int(sub["actual_decision_cash_shortfall"].sum())
        act_sf_cov = (n_sub - act_sf_cnt) / n_sub
        act_w = wilson_score_interval(n_sub - act_sf_cnt, n_sub)
        act_mbb = moving_block_bootstrap_ci(np.array(~sub["actual_decision_cash_shortfall"], dtype=float), block_length=b_roll)
        act_zf = f">={act_mbb['rule_of_three_bound']*100:.1f}%" if act_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | {'Actual Decision Policy Path Sufficiency':<42} | {act_sf_cov*100:5.1f}% | {act_sf_cnt:2d}/200 | [{act_w[0]*100:.1f}%, {act_w[1]*100:.1f}%]   | {act_mbb['display']:<26} | {act_zf:<20} | ACTUAL_ORCHESTRATOR")

        # 8. Counterfactual Stress Scenario (100% Surplus Placed into 15h Repo)
        cf_sf_cnt = int(sub["counterfactual_cash_shortfall"].sum())
        cf_sf_cov = (n_sub - cf_sf_cnt) / n_sub
        cf_w = wilson_score_interval(n_sub - cf_sf_cnt, n_sub)
        cf_mbb = moving_block_bootstrap_ci(np.array(~sub["counterfactual_cash_shortfall"], dtype=float), block_length=b_roll)
        cf_zf = f">={cf_mbb['rule_of_three_bound']*100:.1f}%" if cf_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | {'Counterfactual 15h Repo Sweep Stress Path':<42} | {cf_sf_cov*100:5.1f}% | {cf_sf_cnt:2d}/200 | [{cf_w[0]*100:.1f}%, {cf_w[1]*100:.1f}%]   | {cf_mbb['display']:<26} | {cf_zf:<20} | COUNTERFACTUAL_REPO")
        print("-" * 165)

    print("\n" + "=" * 165)
    print("TABLE 2: NON-OVERLAPPING ORIGINS (step=h)")
    print("MBB configuration: B=2,000, seed=42, block_length=1 observation (disjoint non-overlapping windows)")
    print("=" * 165)
    print(f"{'Horizon':<8} | {'Windows (N)':<12} | {'Metric / Scope':<42} | {'Cov%':<7} | {'Exc':<6} | {'Wilson 95% Ref CI':<19} | {'MBB 95% Bound':<26} | {'Zero-Failure Bound':<20} | {'Status'}")
    print("-" * 165)
    for h in [1, 6, 24, 72]:
        r = non_overlap_results[h]
        nw = r["n_windows"]
        pt_zf = f">={r['pt_mbb']['rule_of_three_bound']*100:.1f}%" if r["pt_mbb"]["is_degenerate"] else "N/A (>0 exc)"
        cum_zf = f">={r['cum_mbb']['rule_of_three_bound']*100:.1f}%" if r["cum_mbb"]["is_degenerate"] else "N/A (>0 exc)"
        comp_t_zf = f">={r['comp_term_mbb']['rule_of_three_bound']*100:.1f}%" if r["comp_term_mbb"]["is_degenerate"] else "N/A (>0 exc)"
        comp_p_zf = f">={r['comp_peak_mbb']['rule_of_three_bound']*100:.1f}%" if r["comp_peak_mbb"]["is_degenerate"] else "N/A (>0 exc)"
        canon_t_zf = f">={r['canon_term_mbb']['rule_of_three_bound']*100:.1f}%" if r["canon_term_mbb"]["is_degenerate"] else "N/A (>0 exc)"
        canon_p_zf = f">={r['canon_peak_mbb']['rule_of_three_bound']*100:.1f}%" if r["canon_peak_mbb"]["is_degenerate"] else "N/A (>0 exc)"

        print(f"{h:2d} Hours  | N={nw:<10} | {'Point Flow Forecast (t+h)':<42} | {r['pt_cov']*100:5.1f}% | {r['pt_exc']:2d}/{nw:<3} | [{r['pt_wilson'][0]*100:.1f}%, {r['pt_wilson'][1]*100:.1f}%]   | {r['pt_mbb']['display']:<26} | {pt_zf:<20} | {'EMPIRICALLY_VALIDATED' if h == 1 else 'CONDITIONALLY_CALIBRATED_FALLBACK'}")
        print(f"{h:2d} Hours  | N={nw:<10} | {'Raw Cumulative Forecast (t,t+h]':<42} | {r['cum_cov']*100:5.1f}% | {r['cum_exc']:2d}/{nw:<3} | [{r['cum_wilson'][0]*100:.1f}%, {r['cum_wilson'][1]*100:.1f}%]   | {r['cum_mbb']['display']:<26} | {cum_zf:<20} | {'EMPIRICALLY_VALIDATED' if h == 1 else 'CONDITIONALLY_CALIBRATED_FALLBACK'}")
        print(f"{h:2d} Hours  | N={nw:<10} | {'Component Reserve [Terminal Outflow]':<42} | {r['comp_term_cov']*100:5.1f}% | {r['comp_term_exc']:2d}/{nw:<3} | [{r['comp_term_wilson'][0]*100:.1f}%, {r['comp_term_wilson'][1]*100:.1f}%]   | {r['comp_term_mbb']['display']:<26} | {comp_t_zf:<20} | {r['policy_status']}")
        print(f"{h:2d} Hours  | N={nw:<10} | {'Component Reserve [Intrahorizon Peak]':<42} | {r['comp_peak_cov']*100:5.1f}% | {r['comp_peak_exc']:2d}/{nw:<3} | [{r['comp_peak_wilson'][0]*100:.1f}%, {r['comp_peak_wilson'][1]*100:.1f}%]   | {r['comp_peak_mbb']['display']:<26} | {comp_p_zf:<20} | {r['policy_status']}")
        print(f"{h:2d} Hours  | N={nw:<10} | {'Canonical Reserve (72h) [Terminal Outflow]':<42} | {r['canon_term_cov']*100:5.1f}% | {r['canon_term_exc']:2d}/{nw:<3} | [{r['canon_term_wilson'][0]*100:.1f}%, {r['canon_term_wilson'][1]*100:.1f}%]   | {r['canon_term_mbb']['display']:<26} | {canon_t_zf:<20} | {r['policy_status']}")
        print(f"{h:2d} Hours  | N={nw:<10} | {'Canonical Reserve (72h) [Intrahorizon Peak]':<42} | {r['canon_peak_cov']*100:5.1f}% | {r['canon_peak_exc']:2d}/{nw:<3} | [{r['canon_peak_wilson'][0]*100:.1f}%, {r['canon_peak_wilson'][1]*100:.1f}%]   | {r['canon_peak_mbb']['display']:<26} | {canon_p_zf:<20} | {r['policy_status']}")
        print("-" * 165)

    print("\n" + "=" * 165)
    print("TABLE 3: EXPANDED FULL-HOLDOUT EVALUATION (Daily step=24h, N=321 origins: 1000..8680)")
    print("MBB configuration: B=2,000, seed=42. Block lengths in daily observations: 6h=1 obs (24h ref), 24h=1 obs (24h ref), 72h=3 obs (72h structural overlap)")
    print("=" * 165)
    print(f"{'Horizon':<8} | {'Origins (N)':<12} | {'Metric / Scope':<42} | {'Cov%':<7} | {'Exc':<6} | {'Wilson 95% Ref CI':<19} | {'MBB 95% Bound':<26} | {'Zero-Failure Bound':<20} | {'Status'}")
    print("-" * 165)
    daily_blocks = {6: 1, 24: 1, 72: 3}
    for h in [6, 24, 72]:
        sub = expanded_df[expanded_df["horizon_hours"] == h]
        n_sub = len(sub)
        b_daily = daily_blocks[h]

        # 1. Point flow
        pt_exc = int(sub["exceeded_robust_point"].sum())
        pt_cov = (n_sub - pt_exc) / n_sub
        pt_w = wilson_score_interval(n_sub - pt_exc, n_sub)
        pt_mbb = moving_block_bootstrap_ci(np.array(~sub["exceeded_robust_point"], dtype=float), block_length=b_daily)
        pt_zf = f">={pt_mbb['rule_of_three_bound']*100:.1f}%" if pt_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | N={n_sub:<10} | {'Point Flow Forecast (t+h)':<42} | {pt_cov*100:5.1f}% | {pt_exc:2d}/{n_sub:<3} | [{pt_w[0]*100:.1f}%, {pt_w[1]*100:.1f}%]   | {pt_mbb['display']:<26} | {pt_zf:<20} | {sub['point_reliability_status'].iloc[0]}")

        # 2. Raw cumulative flow (Terminal)
        cum_exc = int(sub["exceeded_raw_cumulative"].sum())
        cum_cov = (n_sub - cum_exc) / n_sub
        cum_w = wilson_score_interval(n_sub - cum_exc, n_sub)
        cum_mbb = moving_block_bootstrap_ci(np.array(~sub["exceeded_raw_cumulative"], dtype=float), block_length=b_daily)
        cum_zf = f">={cum_mbb['rule_of_three_bound']*100:.1f}%" if cum_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | N={n_sub:<10} | {'Raw Cumulative Forecast (t,t+h] [Terminal]':<42} | {cum_cov*100:5.1f}% | {cum_exc:2d}/{n_sub:<3} | [{cum_w[0]*100:.1f}%, {cum_w[1]*100:.1f}%]   | {cum_mbb['display']:<26} | {cum_zf:<20} | {sub['cumulative_reliability_status'].iloc[0]}")

        # 3. Component Reserve vs Terminal
        comp_t_exc = int(sub["component_reserve_vs_terminal_exceeded"].sum())
        comp_t_cov = (n_sub - comp_t_exc) / n_sub
        comp_t_w = wilson_score_interval(n_sub - comp_t_exc, n_sub)
        comp_t_mbb = moving_block_bootstrap_ci(np.array(~sub["component_reserve_vs_terminal_exceeded"], dtype=float), block_length=b_daily)
        comp_t_zf = f">={comp_t_mbb['rule_of_three_bound']*100:.1f}%" if comp_t_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | N={n_sub:<10} | {'Component Reserve [Terminal Outflow]':<42} | {comp_t_cov*100:5.1f}% | {comp_t_exc:2d}/{n_sub:<3} | [{comp_t_w[0]*100:.1f}%, {comp_t_w[1]*100:.1f}%]   | {comp_t_mbb['display']:<26} | {comp_t_zf:<20} | {sub['policy_reliability_status'].iloc[0]}")

        # 4. Component Reserve vs Peak
        comp_p_exc = int(sub["component_reserve_vs_peak_exceeded"].sum())
        comp_p_cov = (n_sub - comp_p_exc) / n_sub
        comp_p_w = wilson_score_interval(n_sub - comp_p_exc, n_sub)
        comp_p_mbb = moving_block_bootstrap_ci(np.array(~sub["component_reserve_vs_peak_exceeded"], dtype=float), block_length=b_daily)
        comp_p_zf = f">={comp_p_mbb['rule_of_three_bound']*100:.1f}%" if comp_p_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | N={n_sub:<10} | {'Component Reserve [Intrahorizon Peak]':<42} | {comp_p_cov*100:5.1f}% | {comp_p_exc:2d}/{n_sub:<3} | [{comp_p_w[0]*100:.1f}%, {comp_p_w[1]*100:.1f}%]   | {comp_p_mbb['display']:<26} | {comp_p_zf:<20} | {sub['policy_reliability_status'].iloc[0]}")

        # 5. Canonical Multi-Horizon Reserve vs Terminal
        canon_t_exc = int(sub["canonical_reserve_vs_terminal_exceeded"].sum())
        canon_t_cov = (n_sub - canon_t_exc) / n_sub
        canon_t_w = wilson_score_interval(n_sub - canon_t_exc, n_sub)
        canon_t_mbb = moving_block_bootstrap_ci(np.array(~sub["canonical_reserve_vs_terminal_exceeded"], dtype=float), block_length=b_daily)
        canon_t_zf = f">={canon_t_mbb['rule_of_three_bound']*100:.1f}%" if canon_t_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | N={n_sub:<10} | {'Canonical Reserve (72h) [Terminal Outflow]':<42} | {canon_t_cov*100:5.1f}% | {canon_t_exc:2d}/{n_sub:<3} | [{canon_t_w[0]*100:.1f}%, {canon_t_w[1]*100:.1f}%]   | {canon_t_mbb['display']:<26} | {canon_t_zf:<20} | {sub['policy_reliability_status'].iloc[0]}")

        # 6. Canonical Multi-Horizon Reserve vs Peak
        canon_p_exc = int(sub["canonical_reserve_vs_peak_exceeded"].sum())
        canon_p_cov = (n_sub - canon_p_exc) / n_sub
        canon_p_w = wilson_score_interval(n_sub - canon_p_exc, n_sub)
        canon_p_mbb = moving_block_bootstrap_ci(np.array(~sub["canonical_reserve_vs_peak_exceeded"], dtype=float), block_length=b_daily)
        canon_p_zf = f">={canon_p_mbb['rule_of_three_bound']*100:.1f}%" if canon_p_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | N={n_sub:<10} | {'Canonical Reserve (72h) [Intrahorizon Peak]':<42} | {canon_p_cov*100:5.1f}% | {canon_p_exc:2d}/{n_sub:<3} | [{canon_p_w[0]*100:.1f}%, {canon_p_w[1]*100:.1f}%]   | {canon_p_mbb['display']:<26} | {canon_p_zf:<20} | {sub['policy_reliability_status'].iloc[0]}")

        # 7. Actual Decision Simulation Cash Path
        act_sf_cnt = int(sub["actual_decision_cash_shortfall"].sum())
        act_sf_cov = (n_sub - act_sf_cnt) / n_sub
        act_w = wilson_score_interval(n_sub - act_sf_cnt, n_sub)
        act_mbb = moving_block_bootstrap_ci(np.array(~sub["actual_decision_cash_shortfall"], dtype=float), block_length=b_daily)
        act_zf = f">={act_mbb['rule_of_three_bound']*100:.1f}%" if act_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | N={n_sub:<10} | {'Actual Decision Policy Path Sufficiency':<42} | {act_sf_cov*100:5.1f}% | {act_sf_cnt:2d}/{n_sub:<3} | [{act_w[0]*100:.1f}%, {act_w[1]*100:.1f}%]   | {act_mbb['display']:<26} | {act_zf:<20} | ACTUAL_ORCHESTRATOR")

        # 8. Counterfactual Stress Scenario (100% Surplus Placed into 15h Repo)
        cf_sf_cnt = int(sub["counterfactual_cash_shortfall"].sum())
        cf_sf_cov = (n_sub - cf_sf_cnt) / n_sub
        cf_w = wilson_score_interval(n_sub - cf_sf_cnt, n_sub)
        cf_mbb = moving_block_bootstrap_ci(np.array(~sub["counterfactual_cash_shortfall"], dtype=float), block_length=b_daily)
        cf_zf = f">={cf_mbb['rule_of_three_bound']*100:.1f}%" if cf_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | N={n_sub:<10} | {'Counterfactual 15h Repo Sweep Stress Path':<42} | {cf_sf_cov*100:5.1f}% | {cf_sf_cnt:2d}/{n_sub:<3} | [{cf_w[0]*100:.1f}%, {cf_w[1]*100:.1f}%]   | {cf_mbb['display']:<26} | {cf_zf:<20} | COUNTERFACTUAL_REPO")
        print("-" * 165)

    # -------------------------------------------------------------------------
    # PRINT DOCUMENTED SENSITIVITY ANALYSIS TABLES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 165)
    print("SENSITIVITY ANALYSIS 1: ROLLING-ORIGIN BACKTEST (N=200, hourly stride)")
    print("Evaluates Moving Block Bootstrap (B=2,000, seed=42) across block lengths b in [1, 2, 6, 12, 18, 24, 36, 72] hourly observations")
    print("=" * 165)
    roll_candidate_blocks = [1, 2, 6, 12, 18, 24, 36, 72]
    header_str = f"{'Horizon':<8} | {'Metric':<36} | " + " | ".join([f"b={b:2d} ({b:2d}h)" for b in roll_candidate_blocks])
    print(header_str)
    print("-" * len(header_str))
    for h in horizons:
        sub = rolling_df[rolling_df["horizon_hours"] == h]
        ind_comp_term = np.array(~sub["component_reserve_vs_terminal_exceeded"], dtype=float)
        ind_comp_peak = np.array(~sub["component_reserve_vs_peak_exceeded"], dtype=float)
        ind_canon_term = np.array(~sub["canonical_reserve_vs_terminal_exceeded"], dtype=float)
        ind_canon_peak = np.array(~sub["canonical_reserve_vs_peak_exceeded"], dtype=float)

        sens_comp_term = evaluate_block_sensitivity(ind_comp_term, roll_candidate_blocks)
        sens_comp_peak = evaluate_block_sensitivity(ind_comp_peak, roll_candidate_blocks)
        sens_canon_term = evaluate_block_sensitivity(ind_canon_term, roll_candidate_blocks)
        sens_canon_peak = evaluate_block_sensitivity(ind_canon_peak, roll_candidate_blocks)

        print(f"{h:2d} Hours  | {'Component Reserve [Terminal]':<36} | " + " | ".join([sens_comp_term[b]["display"] for b in roll_candidate_blocks]))
        print(f"{h:2d} Hours  | {'Component Reserve [Peak]':<36} | " + " | ".join([sens_comp_peak[b]["display"] for b in roll_candidate_blocks]))
        print(f"{h:2d} Hours  | {'Canonical Reserve (72h) [Terminal]':<36} | " + " | ".join([sens_canon_term[b]["display"] for b in roll_candidate_blocks]))
        print(f"{h:2d} Hours  | {'Canonical Reserve (72h) [Peak]':<36} | " + " | ".join([sens_canon_peak[b]["display"] for b in roll_candidate_blocks]))
        print("-" * len(header_str))

    print("\n" + "=" * 165)
    print("SENSITIVITY ANALYSIS 2: EXPANDED DAILY BACKTEST (N=321, daily stride=24h)")
    print("Evaluates Moving Block Bootstrap (B=2,000, seed=42) across block lengths b in [1, 2, 3, 5, 7, 14, 18, 30] daily observations")
    print("b=1 is an independent-resampling reference baseline; b=3 is the structural 72h window overlap; b=7/14/30 reflect multi-week persistence.")
    print("=" * 165)
    daily_candidate_blocks = [1, 2, 3, 5, 7, 14, 18, 30]
    header_str_daily = f"{'Horizon':<8} | {'Metric':<36} | " + " | ".join([f"b={b:2d} ({b*24:3d}h)" for b in daily_candidate_blocks])
    print(header_str_daily)
    print("-" * len(header_str_daily))
    for h in [6, 24, 72]:
        sub = expanded_df[expanded_df["horizon_hours"] == h]
        ind_comp_term = np.array(~sub["component_reserve_vs_terminal_exceeded"], dtype=float)
        ind_comp_peak = np.array(~sub["component_reserve_vs_peak_exceeded"], dtype=float)
        ind_canon_term = np.array(~sub["canonical_reserve_vs_terminal_exceeded"], dtype=float)
        ind_canon_peak = np.array(~sub["canonical_reserve_vs_peak_exceeded"], dtype=float)

        sens_comp_term = evaluate_block_sensitivity(ind_comp_term, daily_candidate_blocks)
        sens_comp_peak = evaluate_block_sensitivity(ind_comp_peak, daily_candidate_blocks)
        sens_canon_term = evaluate_block_sensitivity(ind_canon_term, daily_candidate_blocks)
        sens_canon_peak = evaluate_block_sensitivity(ind_canon_peak, daily_candidate_blocks)

        print(f"{h:2d} Hours  | {'Component Reserve [Terminal]':<36} | " + " | ".join([sens_comp_term[b]["display"] for b in daily_candidate_blocks]))
        print(f"{h:2d} Hours  | {'Component Reserve [Peak]':<36} | " + " | ".join([sens_comp_peak[b]["display"] for b in daily_candidate_blocks]))
        print(f"{h:2d} Hours  | {'Canonical Reserve (72h) [Terminal]':<36} | " + " | ".join([sens_canon_term[b]["display"] for b in daily_candidate_blocks]))
        print(f"{h:2d} Hours  | {'Canonical Reserve (72h) [Peak]':<36} | " + " | ".join([sens_canon_peak[b]["display"] for b in daily_candidate_blocks]))
        print("-" * len(header_str_daily))

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
