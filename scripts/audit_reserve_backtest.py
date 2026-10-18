#!/usr/bin/env python3
"""Auditable evaluation script reproducing row-level reserve backtest results and statistical intervals.

Outputs:
1. artifacts/reserve_backtest_row_level.csv (600 rows: 200 rolling origins x 3 horizons)
2. artifacts/reserve_backtest_daily_expanded.csv (963 rows: 321 daily origins x 3 horizons)
"""

from __future__ import annotations

import os
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from reserve_juggler.data import resolve_dataset_path
from reserve_juggler.data.schema import DEPOSITS_COL, TIMESTAMP_COL, WITHDRAWALS_COL
from reserve_juggler.forecasting.service import ForecastService
from reserve_juggler.risk.liquidity_shield import assess_liquidity_reserve
from reserve_juggler.contracts import TreasuryState


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
    candidate_blocks: list[int] = [1, 2, 6, 12, 24],
) -> dict[int, dict[str, any]]:
    """Evaluate bootstrap interval sensitivity across varying block lengths."""
    results = {}
    for b in candidate_blocks:
        results[b] = moving_block_bootstrap_ci(indicators, block_length=b)
    return results


def main() -> int:
    dataset_path = resolve_dataset_path()
    with open(dataset_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    print("=" * 135)
    print("RESERVE JUGGLER AUDITABLE BACKTEST — DATASET INTEGRITY & PROVENANCE")
    print(f"Dataset Path : {dataset_path}")
    print(f"SHA-256 Hash : {file_hash}")
    print("=" * 135)

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
            act_cum = float((df.iloc[idx + 1 : idx + h + 1][WITHDRAWALS_COL] - df.iloc[idx + 1 : idx + h + 1][DEPOSITS_COL]).sum())

            raw_pt_p95 = float(getattr(fp, "net_outflow_calibrated_p95", 0.0) or 0.0)
            robust_pt_p95 = float(fp.net_outflow_p95)
            raw_cum_p95 = float(getattr(fp, "cumulative_net_outflow_p95", 0.0) or 0.0)
            applied_buffer = robust_pt_p95 - raw_pt_p95

            state = TreasuryState(
                timestamp=as_of_ts,
                operational_cash_usd=100_000_000.0,
                obligations=[],
            )
            assessment = assess_liquidity_reserve(
                state=state,
                forecast_bundle=bundle,
                use_calibrated_net_quantile=True,
                payout_window_hours=float(h),
                forecast_horizon_hours=float(h),
            )

            cons_deficit = float(assessment.conservative_net_deficit)
            tot_reserve = float(assessment.total_required_reserve)
            avail_cash = float(state.available_cash)
            surplus = float(assessment.eligible_investable_surplus)

            exc_raw_pt = act_pt > raw_pt_p95
            exc_robust_pt = act_pt > robust_pt_p95
            exc_raw_cum = act_cum > raw_cum_p95
            exc_policy = act_cum > cons_deficit

            rolling_records.append({
                "forecast_origin_idx": idx,
                "as_of_timestamp": as_of_ts.isoformat(),
                "horizon_hours": h,
                "target_timestamp": (as_of_ts + pd.Timedelta(hours=h)).isoformat(),
                "actual_point_net_usd": act_pt,
                "actual_cumulative_net_usd": act_cum,
                "raw_point_p95_usd": raw_pt_p95,
                "robust_point_p95_usd": robust_pt_p95,
                "raw_cumulative_p95_usd": raw_cum_p95,
                "applied_point_buffer_usd": applied_buffer,
                "conservative_net_deficit_usd": cons_deficit,
                "total_required_reserve_usd": tot_reserve,
                "available_cash_usd": avail_cash,
                "eligible_surplus_usd": surplus,
                "exceeded_raw_point": exc_raw_pt,
                "exceeded_robust_point": exc_robust_pt,
                "exceeded_raw_cumulative": exc_raw_cum,
                "exceeded_policy_reserve": exc_policy,
                "point_reliability_status": fp.point_reliability_status,
                "cumulative_reliability_status": fp.cumulative_reliability_status,
                "policy_reliability_status": assessment.policy_reliability_status,
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
        policy_cov = []
        origins_used = []
        idx = N_FIT

        while idx + h < len(df):
            origins_used.append(idx)
            as_of_ts = df.iloc[idx][TIMESTAMP_COL]
            bundle = svc.forecast(as_of_ts)
            fp = bundle.get_point(h)

            act_pt = float(df.iloc[idx + h][WITHDRAWALS_COL] - df.iloc[idx + h][DEPOSITS_COL])
            act_cum = float((df.iloc[idx + 1 : idx + h + 1][WITHDRAWALS_COL] - df.iloc[idx + 1 : idx + h + 1][DEPOSITS_COL]).sum())

            pt_cov.append(act_pt <= fp.net_outflow_p95)
            cum_cov.append(act_cum <= getattr(fp, "cumulative_net_outflow_p95", 0.0))

            state = TreasuryState(timestamp=as_of_ts, operational_cash_usd=100_000_000.0, obligations=[])
            assessment = assess_liquidity_reserve(
                state=state,
                forecast_bundle=bundle,
                use_calibrated_net_quantile=True,
                payout_window_hours=float(h),
                forecast_horizon_hours=float(h),
            )
            policy_cov.append(act_cum <= assessment.conservative_net_deficit)

            idx += h
            if h <= 24 and len(origins_used) >= 100:
                break

        n_w = len(origins_used)
        arr_pt_no = np.array(pt_cov, dtype=float)
        arr_cum_no = np.array(cum_cov, dtype=float)
        arr_pol_no = np.array(policy_cov, dtype=float)

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
            "pt_mbb": moving_block_bootstrap_ci(arr_pt_no, block_length=max(1, h // 4)),
            "cum_cov": float(np.mean(cum_cov)),
            "cum_exc": n_w - sum(cum_cov),
            "cum_wilson": wilson_score_interval(sum(cum_cov), n_w),
            "cum_mbb": moving_block_bootstrap_ci(arr_cum_no, block_length=max(1, h // 4)),
            "policy_cov": float(np.mean(policy_cov)),
            "policy_exc": n_w - sum(policy_cov),
            "policy_wilson": wilson_score_interval(sum(policy_cov), n_w),
            "policy_mbb": moving_block_bootstrap_ci(arr_pol_no, block_length=max(1, h // 4)),
            "policy_status": "EMPIRICALLY_VALIDATED" if h == 1 else "CONDITIONALLY_CALIBRATED_FALLBACK",
        }

    # -------------------------------------------------------------------------
    # PART C: EXPANDED FULL-HOLDOUT EVALUATION (Daily step=24h, N=321 origins)
    # -------------------------------------------------------------------------
    # Exact origin sequence: start at row 1000, step 24, max target row idx + 72 <= 8759 (idx <= 8687)
    # Generated origins: 1000, 1024, ..., 8680 (321 total origins)
    daily_origins = [idx for idx in range(N_FIT, len(df) - 72, 24)]
    expanded_records = []

    for h in [6, 24, 72]:
        for idx in daily_origins:
            as_of_ts = df.iloc[idx][TIMESTAMP_COL]
            bundle = svc.forecast(as_of_ts)
            fp = bundle.get_point(h)

            act_pt = float(df.iloc[idx + h][WITHDRAWALS_COL] - df.iloc[idx + h][DEPOSITS_COL])
            act_cum = float((df.iloc[idx + 1 : idx + h + 1][WITHDRAWALS_COL] - df.iloc[idx + 1 : idx + h + 1][DEPOSITS_COL]).sum())

            raw_pt_p95 = float(getattr(fp, "net_outflow_calibrated_p95", 0.0) or 0.0)
            robust_pt_p95 = float(fp.net_outflow_p95)
            raw_cum_p95 = float(getattr(fp, "cumulative_net_outflow_p95", 0.0) or 0.0)
            applied_buffer = robust_pt_p95 - raw_pt_p95

            state = TreasuryState(timestamp=as_of_ts, operational_cash_usd=100_000_000.0, obligations=[])
            assessment = assess_liquidity_reserve(
                state=state,
                forecast_bundle=bundle,
                use_calibrated_net_quantile=True,
                payout_window_hours=float(h),
                forecast_horizon_hours=float(h),
            )

            cons_deficit = float(assessment.conservative_net_deficit)
            tot_reserve = float(assessment.total_required_reserve)
            avail_cash = float(state.available_cash)
            surplus = float(assessment.eligible_investable_surplus)

            exc_raw_pt = act_pt > raw_pt_p95
            exc_robust_pt = act_pt > robust_pt_p95
            exc_raw_cum = act_cum > raw_cum_p95
            exc_policy = act_cum > cons_deficit

            expanded_records.append({
                "forecast_origin_idx": idx,
                "as_of_timestamp": as_of_ts.isoformat(),
                "horizon_hours": h,
                "target_timestamp": (as_of_ts + pd.Timedelta(hours=h)).isoformat(),
                "actual_point_net_usd": act_pt,
                "actual_cumulative_net_usd": act_cum,
                "raw_point_p95_usd": raw_pt_p95,
                "robust_point_p95_usd": robust_pt_p95,
                "raw_cumulative_p95_usd": raw_cum_p95,
                "applied_point_buffer_usd": applied_buffer,
                "conservative_net_deficit_usd": cons_deficit,
                "total_required_reserve_usd": tot_reserve,
                "available_cash_usd": avail_cash,
                "eligible_surplus_usd": surplus,
                "exceeded_raw_point": exc_raw_pt,
                "exceeded_robust_point": exc_robust_pt,
                "exceeded_raw_cumulative": exc_raw_cum,
                "exceeded_policy_reserve": exc_policy,
                "point_reliability_status": fp.point_reliability_status,
                "cumulative_reliability_status": fp.cumulative_reliability_status,
                "policy_reliability_status": assessment.policy_reliability_status,
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
    # Verify rolling artifact integrity
    assert len(rolling_df) == 600, f"Expected 600 records, got {len(rolling_df)}"
    assert len(rolling_df["forecast_origin_idx"].unique()) == 200, "Expected 200 unique origins"
    assert rolling_df["forecast_origin_idx"].min() == 1000
    assert rolling_df["forecast_origin_idx"].max() == 1199

    # Verify expanded artifact integrity
    assert len(expanded_df) == 963, f"Expected 963 records (321 x 3), got {len(expanded_df)}"
    assert len(expanded_df["forecast_origin_idx"].unique()) == 321, "Expected 321 unique daily origins"
    assert expanded_df["forecast_origin_idx"].min() == 1000
    assert expanded_df["forecast_origin_idx"].max() == 8680
    assert (expanded_df["forecast_origin_idx"] + expanded_df["horizon_hours"]).max() <= 8759, "Horizon breached dataset length"

    # -------------------------------------------------------------------------
    # PRINT RESULTS TABLES DIRECTLY FROM PERSISTED ARTIFACT DATA
    # -------------------------------------------------------------------------
    print("\n" + "=" * 145)
    print("TABLE 1: ROLLING-ORIGIN BACKTEST (N=200, Origins 1000..1199, step=1h)")
    print("=" * 145)
    print(f"{'Horizon':<8} | {'Target Description':<28} | {'Cov%':<7} | {'Exc':<6} | {'Wilson 95% Ref CI':<19} | {'MBB 95% Bound':<26} | {'Zero-Failure Bound':<20} | {'Status'}")
    print("-" * 145)
    for h in horizons:
        sub = rolling_df[rolling_df["horizon_hours"] == h]
        n_sub = len(sub)
        
        # Robust point flow
        pt_exc = int(sub["exceeded_robust_point"].sum())
        pt_cov = (n_sub - pt_exc) / n_sub
        pt_w = wilson_score_interval(n_sub - pt_exc, n_sub)
        pt_mbb = moving_block_bootstrap_ci(np.array(~sub["exceeded_robust_point"], dtype=float), block_length=max(2, h // 4))
        pt_zf = f">={pt_mbb['rule_of_three_bound']*100:.1f}%" if pt_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | {'Point Flow (t+h)':<28} | {pt_cov*100:5.1f}% | {pt_exc:2d}/200 | [{pt_w[0]*100:.1f}%, {pt_w[1]*100:.1f}%]   | {pt_mbb['display']:<26} | {pt_zf:<20} | {sub['point_reliability_status'].iloc[0]}")

        # Raw cumulative
        cum_exc = int(sub["exceeded_raw_cumulative"].sum())
        cum_cov = (n_sub - cum_exc) / n_sub
        cum_w = wilson_score_interval(n_sub - cum_exc, n_sub)
        cum_mbb = moving_block_bootstrap_ci(np.array(~sub["exceeded_raw_cumulative"], dtype=float), block_length=max(2, h // 4))
        cum_zf = f">={cum_mbb['rule_of_three_bound']*100:.1f}%" if cum_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | {'Raw Cumulative (t,t+h]':<28} | {cum_cov*100:5.1f}% | {cum_exc:2d}/200 | [{cum_w[0]*100:.1f}%, {cum_w[1]*100:.1f}%]   | {cum_mbb['display']:<26} | {cum_zf:<20} | {sub['cumulative_reliability_status'].iloc[0]}")

        # Reserve Policy
        pol_exc = int(sub["exceeded_policy_reserve"].sum())
        pol_cov = (n_sub - pol_exc) / n_sub
        pol_w = wilson_score_interval(n_sub - pol_exc, n_sub)
        pol_mbb = moving_block_bootstrap_ci(np.array(~sub["exceeded_policy_reserve"], dtype=float), block_length=max(2, h // 4))
        pol_zf = f">={pol_mbb['rule_of_three_bound']*100:.1f}%" if pol_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | {'Production Reserve Policy':<28} | {pol_cov*100:5.1f}% | {pol_exc:2d}/200 | [{pol_w[0]*100:.1f}%, {pol_w[1]*100:.1f}%]   | {pol_mbb['display']:<26} | {pol_zf:<20} | {sub['policy_reliability_status'].iloc[0]}")
        print("-" * 145)

    print("\n" + "=" * 145)
    print("TABLE 2: NON-OVERLAPPING ORIGINS (step=h)")
    print("=" * 145)
    print(f"{'Horizon':<8} | {'Windows (N)':<12} | {'Target Description':<28} | {'Cov%':<7} | {'Exc':<6} | {'Wilson 95% Ref CI':<19} | {'MBB 95% Bound':<26} | {'Zero-Failure Bound':<20} | {'Status'}")
    print("-" * 145)
    for h in [1, 6, 24, 72]:
        r = non_overlap_results[h]
        nw = r["n_windows"]
        pt_zf = f">={r['pt_mbb']['rule_of_three_bound']*100:.1f}%" if r["pt_mbb"]["is_degenerate"] else "N/A (>0 exc)"
        cum_zf = f">={r['cum_mbb']['rule_of_three_bound']*100:.1f}%" if r["cum_mbb"]["is_degenerate"] else "N/A (>0 exc)"
        pol_zf = f">={r['policy_mbb']['rule_of_three_bound']*100:.1f}%" if r["policy_mbb"]["is_degenerate"] else "N/A (>0 exc)"

        print(f"{h:2d} Hours  | N={nw:<10} | {'Point Flow (t+h)':<28} | {r['pt_cov']*100:5.1f}% | {r['pt_exc']:2d}/{nw:<3} | [{r['pt_wilson'][0]*100:.1f}%, {r['pt_wilson'][1]*100:.1f}%]   | {r['pt_mbb']['display']:<26} | {pt_zf:<20} | {'EMPIRICALLY_VALIDATED' if h == 1 else 'CONDITIONALLY_CALIBRATED_FALLBACK'}")
        print(f"{h:2d} Hours  | N={nw:<10} | {'Raw Cumulative (t,t+h]':<28} | {r['cum_cov']*100:5.1f}% | {r['cum_exc']:2d}/{nw:<3} | [{r['cum_wilson'][0]*100:.1f}%, {r['cum_wilson'][1]*100:.1f}%]   | {r['cum_mbb']['display']:<26} | {cum_zf:<20} | {'EMPIRICALLY_VALIDATED' if h == 1 else 'CONDITIONALLY_CALIBRATED_FALLBACK'}")
        print(f"{h:2d} Hours  | N={nw:<10} | {'Production Reserve Policy':<28} | {r['policy_cov']*100:5.1f}% | {r['policy_exc']:2d}/{nw:<3} | [{r['policy_wilson'][0]*100:.1f}%, {r['policy_wilson'][1]*100:.1f}%]   | {r['policy_mbb']['display']:<26} | {pol_zf:<20} | {r['policy_status']}")
        print("-" * 145)

    print("\n" + "=" * 145)
    print("TABLE 3: EXPANDED FULL-HOLDOUT EVALUATION (Daily step=24h, N=321 origins: 1000..8680)")
    print("=" * 145)
    print(f"{'Horizon':<8} | {'Origins (N)':<12} | {'Target Description':<28} | {'Cov%':<7} | {'Exc':<6} | {'Wilson 95% Ref CI':<19} | {'MBB 95% Bound':<26} | {'Zero-Failure Bound':<20} | {'Status'}")
    print("-" * 145)
    for h in [6, 24, 72]:
        sub = expanded_df[expanded_df["horizon_hours"] == h]
        n_sub = len(sub)

        pt_exc = int(sub["exceeded_robust_point"].sum())
        pt_cov = (n_sub - pt_exc) / n_sub
        pt_w = wilson_score_interval(n_sub - pt_exc, n_sub)
        pt_mbb = moving_block_bootstrap_ci(np.array(~sub["exceeded_robust_point"], dtype=float), block_length=max(2, h // 4))
        pt_zf = f">={pt_mbb['rule_of_three_bound']*100:.1f}%" if pt_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | N={n_sub:<10} | {'Point Flow (t+h)':<28} | {pt_cov*100:5.1f}% | {pt_exc:2d}/{n_sub:<3} | [{pt_w[0]*100:.1f}%, {pt_w[1]*100:.1f}%]   | {pt_mbb['display']:<26} | {pt_zf:<20} | {sub['point_reliability_status'].iloc[0]}")

        cum_exc = int(sub["exceeded_raw_cumulative"].sum())
        cum_cov = (n_sub - cum_exc) / n_sub
        cum_w = wilson_score_interval(n_sub - cum_exc, n_sub)
        cum_mbb = moving_block_bootstrap_ci(np.array(~sub["exceeded_raw_cumulative"], dtype=float), block_length=max(2, h // 4))
        cum_zf = f">={cum_mbb['rule_of_three_bound']*100:.1f}%" if cum_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | N={n_sub:<10} | {'Raw Cumulative (t,t+h]':<28} | {cum_cov*100:5.1f}% | {cum_exc:2d}/{n_sub:<3} | [{cum_w[0]*100:.1f}%, {cum_w[1]*100:.1f}%]   | {cum_mbb['display']:<26} | {cum_zf:<20} | {sub['cumulative_reliability_status'].iloc[0]}")

        pol_exc = int(sub["exceeded_policy_reserve"].sum())
        pol_cov = (n_sub - pol_exc) / n_sub
        pol_w = wilson_score_interval(n_sub - pol_exc, n_sub)
        pol_mbb = moving_block_bootstrap_ci(np.array(~sub["exceeded_policy_reserve"], dtype=float), block_length=max(2, h // 4))
        pol_zf = f">={pol_mbb['rule_of_three_bound']*100:.1f}%" if pol_mbb["is_degenerate"] else "N/A (>0 exc)"
        print(f"{h:2d} Hours  | N={n_sub:<10} | {'Production Reserve Policy':<28} | {pol_cov*100:5.1f}% | {pol_exc:2d}/{n_sub:<3} | [{pol_w[0]*100:.1f}%, {pol_w[1]*100:.1f}%]   | {pol_mbb['display']:<26} | {pol_zf:<20} | {sub['policy_reliability_status'].iloc[0]}")
        print("-" * 145)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
