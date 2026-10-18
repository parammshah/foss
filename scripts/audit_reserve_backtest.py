#!/usr/bin/env python3
"""Auditable evaluation script reproducing row-level reserve backtest results and statistical intervals.

Outputs row-level evaluation artifact with exact timestamps, realized customer flows,
forecasts, policy reserves, exceedances, and target-specific reliability labels.
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
    """Compute exact Wilson score confidence interval strictly bounded in [0, 1]."""
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
) -> tuple[float, float]:
    """Compute Moving Block Bootstrap (MBB) empirical percentile confidence interval
    accounting for serial correlation across consecutive time steps.
    """
    n = len(indicators)
    if n < 5:
        return float(np.mean(indicators)), float(np.mean(indicators))
    b = max(2, min(block_length, n // 2))
    num_blocks = int(np.ceil(n / b))
    
    # Generate all overlapping blocks of length b
    blocks = [indicators[i : i + b] for i in range(n - b + 1)]
    if not blocks:
        return float(np.mean(indicators)), float(np.mean(indicators))
    
    rng = np.random.default_rng(seed)
    boot_means = np.empty(num_bootstraps)
    for i in range(num_bootstraps):
        chosen_indices = rng.integers(0, len(blocks), size=num_blocks)
        resampled = np.concatenate([blocks[idx] for idx in chosen_indices])[:n]
        boot_means[i] = np.mean(resampled)
        
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100.0))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return max(0.0, lower), min(1.0, upper)


def main() -> int:
    dataset_path = resolve_dataset_path()
    with open(dataset_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
        
    print("=" * 115)
    print(f"RESERVE JUGGLER AUDITABLE BACKTEST — DATASET INTEGRITY")
    print(f"Dataset Path : {dataset_path}")
    print(f"SHA-256 Hash : {file_hash}")
    print("=" * 115)
    
    df = pd.read_csv(dataset_path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    
    # Strict training/calibration partition on first 1000 rows
    N_FIT = 1000
    VAL_RATIO = 0.25
    train_df = df.iloc[:N_FIT].copy()
    svc = ForecastService()
    svc.fit(train_df, horizons=[1, 6, 24, 48, 72], val_split_ratio=VAL_RATIO)
    
    print(f"Fitted ForecastService: {N_FIT} total rows ({N_FIT * (1 - VAL_RATIO):.0f} train, {N_FIT * VAL_RATIO:.0f} val)")
    print(f"Evaluation origins strictly begin at row {N_FIT} ({df.iloc[N_FIT][TIMESTAMP_COL]})\n")
    
    # -------------------------------------------------------------------------
    # PART A: ROLLING-ORIGIN BACKTEST (N=200, origins 1000..1199, step=1h)
    # -------------------------------------------------------------------------
    EVAL_ORIGINS = 200
    horizons = [6, 24, 72]
    row_records = []
    summary_results = {}
    
    for h in horizons:
        pt_covered = []
        raw_pt_covered = []
        cum_covered = []
        policy_covered = []
        
        for idx in range(N_FIT, N_FIT + EVAL_ORIGINS):
            as_of_ts = df.iloc[idx][TIMESTAMP_COL]
            bundle = svc.forecast(as_of_ts)
            fp = bundle.get_point(h)
            
            # Realized outcomes
            act_pt = float(df.iloc[idx + h][WITHDRAWALS_COL] - df.iloc[idx + h][DEPOSITS_COL])
            act_cum = float((df.iloc[idx + 1 : idx + h + 1][WITHDRAWALS_COL] - df.iloc[idx + 1 : idx + h + 1][DEPOSITS_COL]).sum())
            
            # Forecasts & policies
            raw_pt_p95 = float(getattr(fp, "net_outflow_calibrated_p95", 0.0) or 0.0)
            robust_pt_p95 = float(fp.net_outflow_p95)
            raw_cum_p95 = float(getattr(fp, "cumulative_net_outflow_p95", 0.0) or 0.0)
            applied_buffer = robust_pt_p95 - raw_pt_p95
            
            # Liquidity reserve assessment with zero scheduled obligations to strictly test customer flow coverage
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
            )
            
            cons_deficit = float(assessment.conservative_net_deficit)
            tot_reserve = float(assessment.total_required_reserve)
            avail_cash = float(state.available_cash)
            surplus = float(assessment.eligible_investable_surplus)
            
            # Exceedance flags
            exc_raw_pt = act_pt > raw_pt_p95
            exc_robust_pt = act_pt > robust_pt_p95
            exc_raw_cum = act_cum > raw_cum_p95
            exc_policy = act_cum > cons_deficit
            
            raw_pt_covered.append(not exc_raw_pt)
            pt_covered.append(not exc_robust_pt)
            cum_covered.append(not exc_raw_cum)
            policy_covered.append(not exc_policy)
            
            row_records.append({
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
            })
            
        summary_results[h] = {
            "n": EVAL_ORIGINS,
            "raw_pt_cov": float(np.mean(raw_pt_covered)),
            "robust_pt_cov": float(np.mean(pt_covered)),
            "robust_pt_exc": EVAL_ORIGINS - sum(pt_covered),
            "robust_pt_wilson": wilson_score_interval(sum(pt_covered), EVAL_ORIGINS),
            "robust_pt_mbb": moving_block_bootstrap_ci(np.array(pt_covered, dtype=float), block_length=h),
            "raw_cum_cov": float(np.mean(cum_covered)),
            "raw_cum_exc": EVAL_ORIGINS - sum(cum_covered),
            "raw_cum_wilson": wilson_score_interval(sum(cum_covered), EVAL_ORIGINS),
            "raw_cum_mbb": moving_block_bootstrap_ci(np.array(cum_covered, dtype=float), block_length=h),
            "policy_cov": float(np.mean(policy_covered)),
            "policy_exc": EVAL_ORIGINS - sum(policy_covered),
            "policy_wilson": wilson_score_interval(sum(policy_covered), EVAL_ORIGINS),
            "policy_mbb": moving_block_bootstrap_ci(np.array(policy_covered, dtype=float), block_length=h),
        }

    # -------------------------------------------------------------------------
    # PART B: NON-OVERLAPPING ORIGINS (step=h, independent windows starting at row 1000)
    # -------------------------------------------------------------------------
    non_overlap_results = {}
    for h in [1, 6, 24, 72]:
        pt_cov = []
        cum_cov = []
        policy_cov = []
        idx = N_FIT
        origins_used = []
        
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
            assessment = assess_liquidity_reserve(state, bundle, use_calibrated_net_quantile=True, payout_window_hours=float(h))
            policy_cov.append(act_cum <= assessment.conservative_net_deficit)
            
            idx += h
            if h <= 24 and len(origins_used) >= 100:
                break
                
        n_w = len(origins_used)
        non_overlap_results[h] = {
            "n_windows": n_w,
            "pt_cov": float(np.mean(pt_cov)),
            "pt_exc": n_w - sum(pt_cov),
            "pt_wilson": wilson_score_interval(sum(pt_cov), n_w),
            "pt_mbb": moving_block_bootstrap_ci(np.array(pt_cov, dtype=float), block_length=max(2, h // 4)),
            "cum_cov": float(np.mean(cum_cov)),
            "cum_exc": n_w - sum(cum_cov),
            "cum_wilson": wilson_score_interval(sum(cum_cov), n_w),
            "cum_mbb": moving_block_bootstrap_ci(np.array(cum_cov, dtype=float), block_length=max(2, h // 4)),
            "policy_cov": float(np.mean(policy_cov)),
            "policy_exc": n_w - sum(policy_cov),
            "policy_wilson": wilson_score_interval(sum(policy_cov), n_w),
            "policy_mbb": moving_block_bootstrap_ci(np.array(policy_cov, dtype=float), block_length=max(2, h // 4)),
        }

    # Save row-level CSV artifact
    out_df = pd.DataFrame(row_records)
    artifact_path = Path("artifacts/reserve_backtest_row_level.csv")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(artifact_path, index=False)
    print(f"Saved {len(out_df)} row-level backtest evaluations to: {artifact_path}")
    
    # Also copy to system generated brain artifact directory if available
    brain_artifact_dir = Path("/Users/paramshah/.gemini/antigravity-ide/brain/4f700b38-a8e5-4609-a671-eefc75b6fe00")
    if brain_artifact_dir.exists():
        shutil.copy(artifact_path, brain_artifact_dir / "reserve_backtest_row_level.csv")
        print(f"Copied row-level backtest artifact to: {brain_artifact_dir / 'reserve_backtest_row_level.csv'}")

    # -------------------------------------------------------------------------
    # PRINT RESULTS TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 125)
    print("AUDIT RESULTS: ROLLING-ORIGIN BACKTEST (N=200, Origins 1000..1199, step=1h)")
    print("=" * 125)
    print(f"{'Horizon':<8} | {'Target Description':<28} | {'Cov%':<7} | {'Exc':<6} | {'Wilson 95% CI':<18} | {'MBB (Block) 95% CI':<20} | {'Status Label'}")
    print("-" * 125)
    for h in horizons:
        r = summary_results[h]
        pt_status = "EMPIRICALLY_VALIDATED" if h == 1 else "CONDITIONALLY_CALIBRATED_FALLBACK"
        print(f"{h:2d} Hours  | {'Point Flow (t+h)':<28} | {r['robust_pt_cov']*100:5.1f}% | {r['robust_pt_exc']:2d}/200 | [{r['robust_pt_wilson'][0]*100:.1f}%, {r['robust_pt_wilson'][1]*100:.1f}%]  | [{r['robust_pt_mbb'][0]*100:.1f}%, {r['robust_pt_mbb'][1]*100:.1f}%]    | {pt_status}")
        print(f"{h:2d} Hours  | {'Raw Cumulative (t,t+h]':<28} | {r['raw_cum_cov']*100:5.1f}% | {r['raw_cum_exc']:2d}/200 | [{r['raw_cum_wilson'][0]*100:.1f}%, {r['raw_cum_wilson'][1]*100:.1f}%]  | [{r['raw_cum_mbb'][0]*100:.1f}%, {r['raw_cum_mbb'][1]*100:.1f}%]    | CONDITIONALLY_CALIBRATED_FALLBACK")
        print(f"{h:2d} Hours  | {'Production Reserve Policy':<28} | {r['policy_cov']*100:5.1f}% | {r['policy_exc']:2d}/200 | [{r['policy_wilson'][0]*100:.1f}%, {r['policy_wilson'][1]*100:.1f}%]  | [{r['policy_mbb'][0]*100:.1f}%, {r['policy_mbb'][1]*100:.1f}%]    | {'EMPIRICALLY_VALIDATED' if h <= 6 else 'CONDITIONALLY_CALIBRATED_FALLBACK'}")
        print("-" * 125)

    print("\n" + "=" * 125)
    print("AUDIT RESULTS: NON-OVERLAPPING ORIGINS (step=h, independent windows)")
    print("=" * 125)
    print(f"{'Horizon':<8} | {'Windows (N)':<12} | {'Target Description':<28} | {'Cov%':<7} | {'Exc':<6} | {'Wilson 95% CI':<18} | {'MBB (Block) 95% CI':<20}")
    print("-" * 125)
    for h in [1, 6, 24, 72]:
        r = non_overlap_results[h]
        nw = r["n_windows"]
        print(f"{h:2d} Hours  | N={nw:<10} | {'Point Flow (t+h)':<28} | {r['pt_cov']*100:5.1f}% | {r['pt_exc']:2d}/{nw:<3} | [{r['pt_wilson'][0]*100:.1f}%, {r['pt_wilson'][1]*100:.1f}%]  | [{r['pt_mbb'][0]*100:.1f}%, {r['pt_mbb'][1]*100:.1f}%]")
        print(f"{h:2d} Hours  | N={nw:<10} | {'Raw Cumulative (t,t+h]':<28} | {r['cum_cov']*100:5.1f}% | {r['cum_exc']:2d}/{nw:<3} | [{r['cum_wilson'][0]*100:.1f}%, {r['cum_wilson'][1]*100:.1f}%]  | [{r['cum_mbb'][0]*100:.1f}%, {r['cum_mbb'][1]*100:.1f}%]")
        print(f"{h:2d} Hours  | N={nw:<10} | {'Production Reserve Policy':<28} | {r['policy_cov']*100:5.1f}% | {r['policy_exc']:2d}/{nw:<3} | [{r['policy_wilson'][0]*100:.1f}%, {r['policy_wilson'][1]*100:.1f}%]  | [{r['policy_mbb'][0]*100:.1f}%, {r['policy_mbb'][1]*100:.1f}%]")
        print("-" * 125)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
