"""Unit tests for feature engineering and mathematical verification of zero lookahead leakage."""

import numpy as np
import pandas as pd
import pytest
from reserve_juggler.data import FeatureEngineer, RealisticTreasuryDataGenerator, DEPOSITS_COL, WITHDRAWALS_COL


def test_feature_engineering_columns():
    gen = RealisticTreasuryDataGenerator(seed=42)
    df = gen.generate(hours=200)

    engineer = FeatureEngineer(lags=[1, 2, 6, 24], rolling_windows=[6, 24])
    featured = engineer.create_features(df)

    # Check temporal features exist
    for col in ["hour_of_day", "day_of_week", "sin_hour", "cos_hour", "is_weekend"]:
        assert col in featured.columns

    # Check lags exist
    for lag in [1, 2, 6, 24]:
        assert f"{DEPOSITS_COL}_lag_{lag}" in featured.columns
        assert f"{WITHDRAWALS_COL}_lag_{lag}" in featured.columns

    # Check rolling stats exist
    for win in [6, 24]:
        assert f"{DEPOSITS_COL}_roll_mean_{win}h" in featured.columns
        assert f"{DEPOSITS_COL}_roll_std_{win}h" in featured.columns


def test_zero_lookahead_leakage_assertion():
    """Verify that a massive perturbation in target at row T has ZERO impact on predictors for rows <= T."""
    gen = RealisticTreasuryDataGenerator(seed=42)
    df = gen.generate(hours=300)

    # Test perturbation at multiple index points (e.g., 50, 100, 150)
    for t in [50, 100, 150]:
        no_leakage = FeatureEngineer.assert_no_lookahead_leakage(
            original_df=df,
            featured_df=None,
            perturbed_index=t,
            target_col=DEPOSITS_COL,
        )
        assert no_leakage, f"Lookahead leakage detected when perturbing index {t}!"
