#!/usr/bin/env python3
"""CLI utility to validate and profile treasury datasets for Reserve Juggler.

Usage:
    python scripts/validate_treasury_dataset.py --input /Users/paramshah/Downloads/treasury_4tier_dataset.csv
    python scripts/validate_treasury_dataset.py --input /Users/paramshah/Downloads/treasury_4tier_dataset.csv --output data/normalized_dataset.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

from reserve_juggler.data import (
    DatasetPreprocessor,
    DatasetValidator,
    resolve_dataset_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and profile a Reserve Juggler four-tier treasury dataset."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to input dataset CSV (defaults to RESERVE_JUGGLER_DATA_PATH or data/challenge_dataset_8760.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to save cleaned, normalized dataset without overwriting original",
    )
    parser.add_argument(
        "--net-flow-tolerance",
        type=float,
        default=1.0,
        help="Tolerance in USD for net_flow = deposits - withdrawals reconciliation (default: 1.0)",
    )
    args = parser.parse_args()

    try:
        dataset_path = resolve_dataset_path(args.input)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Loading raw dataset from: {dataset_path}")
    raw_df = pd.read_csv(dataset_path)

    # 1. Validation Report
    validator = DatasetValidator(net_flow_tolerance_usd=args.net_flow_tolerance)
    report = validator.validate(raw_df)
    print("\n" + report.summary())

    # 2. Detailed Data Quality Issues
    if report.issues:
        print("\nDETAILED ISSUES & WARNINGS:")
        for idx, issue in enumerate(report.issues, 1):
            aff = f" (Columns: {', '.join(issue.affected_columns)})" if issue.affected_columns else ""
            print(f"  [{issue.severity}] {issue.issue_type}: {issue.message}{aff}")

    # 3. Preprocessing & Ingestion Normalization
    preprocessor = DatasetPreprocessor()
    clean_df = preprocessor.preprocess(raw_df)
    print("\nUNIT NORMALIZATION SUMMARY:")
    for col, desc in preprocessor.normalization_log.items():
        print(f"  - {col}: {desc}")

    # 4. Net Flow Reconciliation Check
    calc_net = clean_df["deposits_usd"] - clean_df["withdrawals_usd"]
    diff = (clean_df["net_flow_usd"] - calc_net).abs()
    print(f"\nNET FLOW RECONCILIATION:")
    print(f"  Max absolute discrepancy: ${diff.max():.4f}")
    print(f"  Mean absolute discrepancy: ${diff.mean():.4f}")
    print(f"  Rows with discrepancy > $0.01: {(diff > 0.01).sum()}")
    print(f"  Rows with discrepancy > $1.00: {(diff > 1.00).sum()}")

    # 5. Optional Save
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(out_path, index=False)
        print(f"\nSaved normalized dataset to: {out_path.resolve()}")

    return 0 if report.is_valid else 2


if __name__ == "__main__":
    sys.exit(main())
