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
        default=1000,
        help="Initial training window size in hours (default: 1000 hours, ~41 days)",
    )
    parser.add_argument(
        "--roll-step-hours",
        type=int,
        default=168,
        help="Roll step in hours between consecutive origins (default: 168 hours, 1 week)",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=10,
        help="Maximum number of chronological origins to evaluate (default: 10)",
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
    print(f"Preprocessed {len(clean_df)} observations. Starting backtest...")

    backtester = RollingOriginBacktest()
    report = backtester.run(
        df=clean_df,
        target_cols=[DEPOSITS_COL, WITHDRAWALS_COL],
        horizons=args.horizons,
        initial_train_size=args.initial_train_size,
        roll_step_hours=args.roll_step_hours,
        max_evaluations=args.max_evaluations,
    )

    print("\n" + report.summary())

    print("\nDETAILED METRICS BY TARGET & HORIZON:")
    for h_res in report.horizon_results:
        print(f"\nTarget: {h_res.target_variable:<14} | Horizon: {h_res.horizon_hours} Hours | Samples: {h_res.sample_count}")
        print(f"  Recent Average Baseline : MAE=${h_res.recent_baseline_mae:<12,.2f} | WAPE={h_res.recent_baseline_wape:.2%}")
        print(f"  Time-of-Week Seasonal   : MAE=${h_res.seasonal_baseline_mae:<12,.2f} | WAPE={h_res.seasonal_baseline_wape:.2%}")
        print(f"  Ridge ML Model          : MAE=${h_res.ridge_ml_mae:<12,.2f} | WAPE={h_res.ridge_ml_wape:.2%}")
        print(f"  OR Constrained Ensemble : MAE=${h_res.or_ensemble_mae:<12,.2f} | WAPE={h_res.or_ensemble_wape:.2%}")
        if h_res.picp_80 is not None:
            print(f"  Prediction Interval 80% : PICP={h_res.picp_80:.1%} (Target=80.0%) | Calibration Error={h_res.calibration_error_80:.1%}")
        if h_res.pinball_p10 is not None and h_res.pinball_p90 is not None:
            print(f"  Quantile Pinball Losses : P10=${h_res.pinball_p10:,.2f} | P90=${h_res.pinball_p90:,.2f}")
        print(f"  OR Optimal Model Weights: {h_res.or_ensemble_weights}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
