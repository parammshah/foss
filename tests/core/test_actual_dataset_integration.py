"""Unit and regression tests for actual four-tier treasury dataset integration."""

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from reserve_juggler.contracts import (
    AssetTier,
    DataQualityStatus,
    InMemoryEventPublisher,
)
from reserve_juggler.data import (
    ACTUAL_TO_CANONICAL_COLUMN_MAP,
    DatasetPreprocessor,
    DatasetValidator,
    DEPOSITS_COL,
    EMERGENCY_OVERDRAFT_PENALTY_APR_COL,
    NET_FLOW_COL,
    OBSERVED_ACTUAL_CONSTANTS,
    TIER1_EARLY_REDEMPTION_HAIRCUT_COL,
    TIER1_REPO_RATE_COL,
    TIER2_BOND_LIQUIDATION_HAIRCUT_COL,
    TIER2_BOND_MOD_DURATION_COL,
    TIER2_BOND_YIELD_COL,
    TIER3_STOCK_LIQUIDATION_HAIRCUT_COL,
    TIER3_STOCK_VOLATILITY_COL,
    TIMESTAMP_COL,
    WITHDRAWALS_COL,
    load_treasury_dataset,
    resolve_dataset_path,
)
from reserve_juggler.forecasting import ForecastService
from reserve_juggler.ledger import InitialTreasuryConfig, TreasuryLedger


@pytest.fixture
def actual_schema_sample_df() -> pd.DataFrame:
    """Fixture replicating the exact 16-column schema and value scales of the actual challenge CSV."""
    ts_range = pd.date_range("2026-01-01 00:00:00", periods=48, freq="1h")
    timestamps = [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in ts_range]
    rng = np.random.default_rng(42)


    deps = rng.uniform(4_000_000, 15_000_000, 48)
    wids = rng.uniform(3_000_000, 16_000_000, 48)
    net_flow = np.round(deps - wids, 2)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "deposits_usd": np.round(deps, 2),
        "withdrawals_usd": np.round(wids, 2),
        "net_flow_usd": net_flow,
        "repo_sweep_window_open": [h == 17 for h in range(48)],
        "repo_sweep_mature": [h == 8 for h in range(48)],
        "tier1_repo_rate_apy": rng.uniform(3.8, 4.3, 48),
        "tier2_bond_yield_pct": rng.uniform(4.2, 5.5, 48),
        "tier2_bond_price_index": rng.uniform(98.0, 102.0, 48),
        "tier2_bond_mod_duration": np.full(48, 1.92),
        "tier3_stock_index_price": rng.uniform(4000.0, 5200.0, 48),
        "tier3_stock_volatility_pct": np.full(48, 18.0),
        "tier1_early_redemption_haircut_pct": np.full(48, 0.5),
        "tier2_bond_liquidation_haircut_pct": np.full(48, 2.0),
        "tier3_stock_liquidation_haircut_pct": np.full(48, 15.0),
        "emergency_overdraft_penalty_apr_pct": np.full(48, 12.0),
    })
    return df


def test_actual_schema_mapping_and_normalization(actual_schema_sample_df):
    """Verify that actual CSV column names are normalized and mapped to canonical fields."""
    preprocessor = DatasetPreprocessor()
    clean_df = preprocessor.preprocess(actual_schema_sample_df)

    # 1. Check all canonical aliases exist alongside original columns
    assert "tier2_bond_duration_years" in clean_df.columns
    assert "penalty_rate_tier1_early_exit" in clean_df.columns
    assert "haircut_rate_tier2_emergency" in clean_df.columns
    assert "slippage_rate_tier3_emergency" in clean_df.columns

    # 2. Check single-pass percentage normalization
    # Rates must be normalized to decimals: e.g. 4.2% -> ~0.042, 0.5% -> 0.005, 12% -> 0.12
    assert clean_df["tier1_repo_rate_apy"].max() < 0.10
    assert clean_df["tier2_bond_yield_pct"].max() < 0.10
    assert clean_df["tier3_stock_volatility_pct"].iloc[0] == pytest.approx(0.18)
    assert clean_df["tier1_early_redemption_haircut_pct"].iloc[0] == pytest.approx(0.005)
    assert clean_df["tier2_bond_liquidation_haircut_pct"].iloc[0] == pytest.approx(0.02)
    assert clean_df["tier3_stock_liquidation_haircut_pct"].iloc[0] == pytest.approx(0.15)
    assert clean_df["emergency_overdraft_penalty_apr_pct"].iloc[0] == pytest.approx(0.12)

    # 3. Check timestamps are timezone-aware UTC
    assert clean_df["timestamp"].dt.tz == timezone.utc

    # 4. Check separate validated net flow
    assert "validated_net_flow_usd" in clean_df.columns
    diff = (clean_df["validated_net_flow_usd"] - (clean_df["deposits_usd"] - clean_df["withdrawals_usd"])).abs()
    assert diff.max() < 1e-4


def test_validator_identifies_constant_columns(actual_schema_sample_df):
    """Verify that validator detects constant simulation parameters and flags unnormalized percentages."""
    validator = DatasetValidator()
    report = validator.validate(actual_schema_sample_df)

    # Constant columns must be detected
    for col in [
        "tier2_bond_mod_duration",
        "tier3_stock_volatility_pct",
        "tier1_early_redemption_haircut_pct",
        "tier2_bond_liquidation_haircut_pct",
        "tier3_stock_liquidation_haircut_pct",
        "emergency_overdraft_penalty_apr_pct",
    ]:
        assert col in report.constant_columns

    # Percentage columns must be flagged as unnormalized on raw data
    unnormalized_issues = [i for i in report.issues if i.issue_type == "UNNORMALIZED_PERCENTAGE"]
    assert len(unnormalized_issues) > 0


def test_resolve_dataset_path_errors():
    """Verify that resolve_dataset_path raises FileNotFoundError with actionable instructions when missing."""
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_dataset_path("/path/that/does/not/exist_12345.csv")
    assert "Specified dataset path does not exist" in str(exc_info.value)


def test_forecast_and_ledger_with_actual_schema(actual_schema_sample_df):
    """Verify end-to-end forecasting and ledger simulation using preprocessed actual schema data."""
    preprocessor = DatasetPreprocessor()
    clean_df = preprocessor.preprocess(actual_schema_sample_df)

    # 1. ForecastService on clean data
    service = ForecastService()
    service.fit(clean_df.iloc[:36], horizons=[1, 6])
    now = clean_df["timestamp"].iloc[35]
    bundle = service.forecast(now, horizons_hours=[1, 6])
    assert len(bundle.points) == 2
    assert bundle.get_point(1).deposit_p50 > 0.0

    # 2. TreasuryLedger simulation
    config = InitialTreasuryConfig(opening_cash_usd=20_000_000.0)
    ledger = TreasuryLedger(config)
    state = ledger.get_state(now)
    assert state.operational_cash == 20_000_000.0

    # Step forward 1 hour
    row = clean_df.iloc[36]
    net_flow, shortfall = ledger.process_hourly_step(
        current_timestamp=row["timestamp"],
        deposits_usd=float(row["deposits_usd"]),
        withdrawals_usd=float(row["withdrawals_usd"]),
        repo_sweep_mature=bool(row["repo_sweep_mature"]),
    )
    assert shortfall == 0.0
    assert ledger.cash_usd == pytest.approx(20_000_000.0 + net_flow, abs=1.0)


@pytest.mark.skipif(
    not Path("/Users/paramshah/Downloads/treasury_4tier_dataset.csv").exists(),
    reason="Actual dataset not found in personal downloads directory",
)
def test_real_file_integration_if_present():
    """Integration test against the actual file when running on the developer workstation."""
    real_path = "/Users/paramshah/Downloads/treasury_4tier_dataset.csv"
    raw_df = pd.read_csv(real_path)

    # Assert 8760 rows and 16 columns
    assert raw_df.shape == (8760, 16)
    assert raw_df["timestamp"].iloc[0] == "2026-01-01 00:00:00"
    assert raw_df["timestamp"].iloc[-1] == "2026-12-31 23:00:00"

    # Preprocess
    preprocessor = DatasetPreprocessor()
    clean_df = preprocessor.preprocess(raw_df)

    # Net flow reconciliation
    calc_net = clean_df["deposits_usd"] - clean_df["withdrawals_usd"]
    diff = (clean_df["net_flow_usd"] - calc_net).abs()
    assert diff.max() <= 0.0101  # Currency rounding tolerance
