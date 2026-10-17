"""Unit tests for dataset validation, gap detection, and unit conversion."""

from datetime import datetime, timezone, timedelta
import pandas as pd
import pytest
from reserve_juggler.contracts import DataQualityStatus, InMemoryEventPublisher
from reserve_juggler.data import (
    DatasetPreprocessor,
    DatasetValidator,
    RealisticTreasuryDataGenerator,
    TIMESTAMP_COL,
    DEPOSITS_COL,
    WITHDRAWALS_COL,
    NET_FLOW_COL,
    TIER1_REPO_RATE_COL,
    TIER2_BOND_YIELD_COL,
)


def test_validator_on_clean_data():
    gen = RealisticTreasuryDataGenerator(seed=123)
    df = gen.generate(hours=100)

    publisher = InMemoryEventPublisher()
    validator = DatasetValidator(event_publisher=publisher)
    report = validator.validate(df)

    assert report.is_valid
    assert report.status == DataQualityStatus.OK
    assert report.total_rows == 100
    assert report.missing_hours_count == 0
    assert report.reconciliation_discrepancy_count == 0
    assert len(publisher.events) == 1
    assert publisher.events[0].payload["is_valid"] is True


def test_validator_detects_duplicate_and_gaps():
    gen = RealisticTreasuryDataGenerator(seed=123)
    df = gen.generate(hours=50)

    # Introduce a duplicate row
    df_dup = pd.concat([df, df.iloc[[10]]], ignore_index=True)
    validator = DatasetValidator()
    report_dup = validator.validate(df_dup)
    assert not report_dup.is_valid
    assert any(i.issue_type == "DUPLICATE_TIMESTAMPS" for i in report_dup.issues)

    # Introduce a 3-hour gap by dropping rows 10, 11, 12
    df_gap = df.drop(index=[10, 11, 12]).reset_index(drop=True)
    report_gap = validator.validate(df_gap)
    assert any(i.issue_type == "HOURLY_SEQUENCE_GAP" for i in report_gap.issues)
    assert report_gap.missing_hours_count == 3


def test_validator_detects_negative_and_discrepancies():
    gen = RealisticTreasuryDataGenerator(seed=123)
    df = gen.generate(hours=20)

    # Negative deposit
    df.loc[5, DEPOSITS_COL] = -100.0
    # Net flow discrepancy
    df.loc[10, NET_FLOW_COL] = df.loc[10, DEPOSITS_COL] - df.loc[10, WITHDRAWALS_COL] + 50_000.0

    validator = DatasetValidator(net_flow_tolerance_usd=1.0)
    report = validator.validate(df)

    assert not report.is_valid  # Has negative cash flow error
    assert any(i.issue_type == "NEGATIVE_CASH_FLOW" for i in report.issues)
    assert any(i.issue_type == "NET_FLOW_RECONCILIATION_MISMATCH" for i in report.issues)


def test_preprocessor_normalizes_units_and_fills_gaps():
    gen = RealisticTreasuryDataGenerator(seed=123)
    df = gen.generate(hours=48)

    # Un-normalize percentages to whole numbers (4.8% -> 4.8)
    df[TIER1_REPO_RATE_COL] = df[TIER1_REPO_RATE_COL] * 100.0
    df[TIER2_BOND_YIELD_COL] = df[TIER2_BOND_YIELD_COL] * 100.0
    assert df[TIER1_REPO_RATE_COL].max() > 1.0

    # Drop 2 hours
    df_missing = df.drop(index=[15, 16]).reset_index(drop=True)

    publisher = InMemoryEventPublisher()
    preprocessor = DatasetPreprocessor(event_publisher=publisher)
    clean_df = preprocessor.preprocess(df_missing, fill_hourly_gaps=True)

    assert len(clean_df) == 48
    # Rates normalized to decimal fraction <= 1.0
    assert clean_df[TIER1_REPO_RATE_COL].max() <= 0.15
    assert clean_df[TIER2_BOND_YIELD_COL].max() <= 0.15
    # Recorded imputation events
    assert len(preprocessor.imputed_timestamps) == 2
    assert len(publisher.get_events_by_type("data_gap_imputed")) == 1
