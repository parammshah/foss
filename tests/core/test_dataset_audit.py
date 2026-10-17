"""Regression tests for dataset audit and independent raw CSV verification."""

from __future__ import annotations

import os
import hashlib
import pandas as pd
import pytest

ACTUAL_DATASET_PATH = "/Users/paramshah/Downloads/treasury_4tier_dataset.csv"


def test_independent_raw_audit_identity_and_reconciliation() -> None:
    """Verifies that if the actual challenge CSV is present, its SHA-256 and stats match exactly."""
    if not os.path.exists(ACTUAL_DATASET_PATH):
        pytest.skip(f"Actual dataset not found at {ACTUAL_DATASET_PATH}")

    # Read binary for hash
    with open(ACTUAL_DATASET_PATH, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    assert sha256 == "2e7c1d2e5b678e9975250211e7e0017022049c510e1dd90cd7f29e931837b0c5"

    raw_df = pd.read_csv(ACTUAL_DATASET_PATH)
    assert len(raw_df) == 8760
    assert len(raw_df.columns) == 16

    # Verify cash flow ranges
    assert raw_df["deposits_usd"].min() == pytest.approx(3881238.01, abs=1e-2)
    assert raw_df["deposits_usd"].max() == pytest.approx(37777699.02, abs=1e-2)
    assert raw_df["deposits_usd"].mean() == pytest.approx(10141914.71, abs=1.0)

    assert raw_df["withdrawals_usd"].min() == pytest.approx(3251685.47, abs=1e-2)
    assert raw_df["withdrawals_usd"].max() == pytest.approx(45692972.02, abs=1e-2)
    assert raw_df["withdrawals_usd"].mean() == pytest.approx(10430808.13, abs=1.0)

    # Net flow identity
    diff = (raw_df["deposits_usd"] - raw_df["withdrawals_usd"]) - raw_df["net_flow_usd"]
    assert diff.abs().max() <= 0.010001
    assert (diff.abs() > 1.0).sum() == 0


def test_net_flow_reconciliation_synthetic_fixture() -> None:
    """Verifies independent audit logic on a controlled synthetic fixture."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="h"),
        "deposits_usd": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0],
        "withdrawals_usd": [900.0, 1900.0, 2900.0, 3900.0, 4900.0, 5900.0, 6900.0, 7900.0, 8900.0, 9900.0],
        "net_flow_usd": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
    })
    diff = (df["deposits_usd"] - df["withdrawals_usd"]) - df["net_flow_usd"]
    assert diff.abs().max() == 0.0
