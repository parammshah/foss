#!/usr/bin/env python3
"""Evaluate the final untouched 720-hour holdout of the treasury dataset.

Protocol:
- Development data: Rows 0 to 8039 (2026-01-01 00:00:00 to 2026-12-01 23:00:00 UTC).
- Holdout data: Rows 8040 to 8759 (2026-12-02 00:00:00 to 2026-12-31 23:00:00 UTC, 720 hours).
- Evaluation origins: Daily origins inside the holdout window.
- Models: Recent-Average Baseline, Seasonal Baseline, Ridge ML Model, OR Constrained Ensemble.
- Horizons: 1, 6, 24, 48, 72 hours for Deposits and Withdrawals.
- Zero future leakage: Model hyperparameters and ensemble weights are strictly fitted on data prior to each origin.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

from reserve_juggler.data import (
    DEPOSITS_COL,
    DatasetPreprocessor,
    WITHDRAWALS_COL,
    resolve_dataset_path,
)
from reserve_juggler.evaluation.backtest import RollingOriginBacktest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run holdout evaluation against untouched final 720 hours of treasury dataset."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to input dataset CSV (defaults to RESERVE_JUGGLER_DATA_PATH or default challenge path)",
    )
    parser.add_argument(
        "--horizons",
        "-H",
        nargs="+",
        type=int,
        default=[1, 6, 24, 48, 72],
        help="Forecast horizons in hours to evaluate (default: 1 6 24 48 72)",
    )
    parser.add_argument(
        "--holdout-hours",
        type=int,
        default=720,
        help="Final holdout window in hours (default: 720 hours, December)",
    )
    parser.add_argument(
        "--roll-step-hours",
        type=int,
        default=24,
        help="Step size between evaluation origins inside holdout (default: 24 hours, daily)",
    )
    args = parser.parse_args()

    try:
        dataset_path = resolve_dataset_path(args.input)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Loading and preprocessing dataset from: {dataset_path}")
    raw_df = pd.read_csv(dataset_path)
    preprocessor = DatasetPreprocessor()
    clean_df = preprocessor.preprocess(raw_df)

    n_total = len(clean_df)
    if n_total < args.holdout_hours + 100:
        print(f"ERROR: Dataset has {n_total} rows, insufficient for {args.holdout_hours} holdout.", file=sys.stderr)
        return 1

    holdout_start_idx = n_total - args.holdout_hours
    holdout_start_ts = clean_df.loc[holdout_start_idx, "timestamp"]
    holdout_end_ts = clean_df.loc[n_total - 1, "timestamp"]

    print("=" * 108)
    print(f"HOLDOUT SPECIFICATION:")
    print(f"  Total Dataset Observations : {n_total} hours")
    print(f"  Pre-Holdout History        : Rows 0 to {holdout_start_idx - 1} (Start: {clean_df.loc[0, 'timestamp']})")
    print(f"  Holdout Window             : Rows {holdout_start_idx} to {n_total - 1} ({args.holdout_hours} hours)")
    print(f"  Holdout Start Timestamp    : {holdout_start_ts}")
    print(f"  Holdout End Timestamp      : {holdout_end_ts}")
    print("=" * 108)

    # In holdout evaluation: initial_train_size is set to holdout_start_idx
    # holdout_hours=0 passed to backtest run because we are explicitly evaluating origins inside the holdout window!
    max_h = max(args.horizons)
    max_evals = (args.holdout_hours - max_h) // args.roll_step_hours

    backtester = RollingOriginBacktest()
    report = backtester.run(
        df=clean_df,
        target_cols=[DEPOSITS_COL, WITHDRAWALS_COL],
        horizons=args.horizons,
        initial_train_size=holdout_start_idx,
        roll_step_hours=args.roll_step_hours,
        max_evaluations=max_evals,
        holdout_hours=0,  # evaluating INSIDE the holdout
    )

    print("\n" + report.summary())

    print("\n" + "=" * 115)
    print("DETAILED HOLDOUT RESULTS BY HORIZON & TARGET:")
    print("=" * 115)
    for h_res in report.horizon_results:
        print(
            f"\nTarget: {h_res.target_variable:<14} | Horizon: {h_res.horizon_hours:2d} Hours | "
            f"Holdout Origins: {h_res.sample_count:2d} | Unique Target Timestamps: {h_res.unique_target_timestamps_count:2d}"
        )
        print(f"  Recent Average Baseline : MAE=${h_res.recent_baseline_mae:<12,.2f} | WAPE={h_res.recent_baseline_wape:.2%}")
        print(f"  Time-of-Week Seasonal   : MAE=${h_res.seasonal_baseline_mae:<12,.2f} | WAPE={h_res.seasonal_baseline_wape:.2%}")
        print(f"  Ridge ML Model          : MAE=${h_res.ridge_ml_mae:<12,.2f} | WAPE={h_res.ridge_ml_wape:.2%}")
        print(f"  OR Constrained Ensemble : MAE=${h_res.or_ensemble_mae:<12,.2f} | WAPE={h_res.or_ensemble_wape:.2%}")
        print(f"  OR Model Weights        : {h_res.or_ensemble_weights}")

        p50_str = f"PICP={h_res.picp_50:.1%} | Width=${h_res.mpiw_50:,.0f}" if h_res.picp_50 is not None else "N/A"
        p80_str = f"PICP={h_res.picp_80:.1%} | Width=${h_res.mpiw_80:,.0f}" if h_res.picp_80 is not None else "N/A"
        p95_str = f"PICP={h_res.picp_95:.1%} | Width=${h_res.mpiw_95:,.0f}" if h_res.picp_95 is not None else "N/A"
        print(f"  50% Interval (Target 50%): {p50_str}")
        print(f"  80% Interval (Target 80%): {p80_str} | Winkler Score=${h_res.winkler_80:,.0f}")
        print(f"  95% Interval (Target 95%): {p95_str}")
        print(f"  Calibration Residuals    : N={h_res.calibration_sample_size}")

        if h_res.regime_metrics:
            print("  Holdout Regime Breakdown:")
            for reg_name, reg_data in h_res.regime_metrics.items():
                if reg_data.get("status") == "VALID":
                    print(
                        f"    - {reg_name:<26}: N={reg_data['sample_count']:2d} | "
                        f"MAE=${reg_data['mae']:<11,.0f} | WAPE={reg_data['wape']:<6.2%} | "
                        f"PICP_80={reg_data.get('picp_80', 0.0):.1%}"
                    )
                else:
                    print(f"    - {reg_name:<26}: N={reg_data['sample_count']:2d} | Insufficient evidence (N < 10)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
