#!/usr/bin/env python3
"""CLI utility to run chronological rolling-origin backtesting on the actual treasury dataset.

Usage:
    python scripts/evaluate_treasury_forecasts.py --input /Users/paramshah/Downloads/treasury_4tier_dataset.csv
    python scripts/evaluate_treasury_forecasts.py --input /Users/paramshah/Downloads/treasury_4tier_dataset.csv --horizons 1 6 24 48 72
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
from reserve_juggler.evaluation import RollingOriginBacktest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run chronological rolling-origin backtesting against actual treasury dataset."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to input dataset CSV (defaults to RESERVE_JUGGLER_DATA_PATH or data/challenge_dataset_8760.csv)",
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
        "--initial-train-size",
        type=int,
        default=1500,
        help="Initial training window size in hours (default: 1500 hours, ~62.5 days)",
    )
    parser.add_argument(
        "--roll-step-hours",
        type=int,
        default=24,
        help="Roll step in hours between consecutive origins (default: 24 hours, daily)",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=60,
        help="Maximum number of chronological origins to evaluate (default: 60)",
    )
    parser.add_argument(
        "--holdout-hours",
        type=int,
        default=720,
        help="Final holdout window in hours reserved untouched (default: 720 hours, 30 days)",
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
    print(
        f"Preprocessed {len(clean_df)} observations. "
        f"Running backtest (train_init={args.initial_train_size}h, step={args.roll_step_hours}h, "
        f"max_evals={args.max_evaluations}, holdout={args.holdout_hours}h)..."
    )

    backtester = RollingOriginBacktest()
    report = backtester.run(
        df=clean_df,
        target_cols=[DEPOSITS_COL, WITHDRAWALS_COL],
        horizons=args.horizons,
        initial_train_size=args.initial_train_size,
        roll_step_hours=args.roll_step_hours,
        max_evaluations=args.max_evaluations,
        holdout_hours=args.holdout_hours,
    )

    print("\n" + report.summary())

    print("\n" + "=" * 115)
    print("DETAILED MULTI-INTERVAL CALIBRATION & REGIME EVALUATIONS:")
    print("=" * 115)
    for h_res in report.horizon_results:
        print(f"\nTarget: {h_res.target_variable:<14} | Horizon: {h_res.horizon_hours:2d} Hours | Evaluated Origins: {h_res.sample_count:2d} | Unique Targets: {h_res.unique_target_timestamps_count:2d}")
        print(f"  Recent Average Baseline : MAE=${h_res.recent_baseline_mae:<12,.2f} | WAPE={h_res.recent_baseline_wape:.2%}")
        print(f"  Time-of-Week Seasonal   : MAE=${h_res.seasonal_baseline_mae:<12,.2f} | WAPE={h_res.seasonal_baseline_wape:.2%}")
        print(f"  Ridge ML Model          : MAE=${h_res.ridge_ml_mae:<12,.2f} | WAPE={h_res.ridge_ml_wape:.2%}")
        print(f"  OR Constrained Ensemble : MAE=${h_res.or_ensemble_mae:<12,.2f} | WAPE={h_res.or_ensemble_wape:.2%}")
        print(f"  OR Model Weights        : {h_res.or_ensemble_weights}")

        # Multi-level interval uncertainty
        p50_str = f"PICP={h_res.picp_50:.1%} | Width=${h_res.mpiw_50:,.0f}" if h_res.picp_50 is not None else "N/A"
        p80_str = f"PICP={h_res.picp_80:.1%} | Width=${h_res.mpiw_80:,.0f}" if h_res.picp_80 is not None else "N/A"
        p95_str = f"PICP={h_res.picp_95:.1%} | Width=${h_res.mpiw_95:,.0f}" if h_res.picp_95 is not None else "N/A"
        print(f"  50% Interval (Target 50%): {p50_str}")
        print(f"  80% Interval (Target 80%): {p80_str} | Winkler Score=${h_res.winkler_80:,.0f}")
        print(f"  95% Interval (Target 95%): {p95_str}")
        print(f"  Calibration Sample Size  : N={h_res.calibration_sample_size} historical residuals")

        # Regimes
        if h_res.regime_metrics:
            print("  Regime Performance:")
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
