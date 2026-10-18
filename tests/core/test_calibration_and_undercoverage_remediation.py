"""Deterministic regression tests for forecast calibration, undercoverage remediation, and quantile diagnostics."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

from reserve_juggler.data import resolve_dataset_path
from reserve_juggler.data.schema import DEPOSITS_COL, TIMESTAMP_COL, WITHDRAWALS_COL
from reserve_juggler.forecasting.service import ForecastService
from reserve_juggler.forecasting.uncertainty import ORScenarioReconciler


def test_degraded_calibration_detection_when_sample_size_limited():
    """Verify that limited calibration sample size (< 150) triggers reliability warning and buffer."""
    reconciler = ORScenarioReconciler()
    rng = np.random.default_rng(42)

    # N = 50 (< 150 threshold)
    dep_true = rng.uniform(10_000_000, 20_000_000, 50)
    dep_pred = dep_true + rng.normal(0, 500_000, 50)
    wid_true = rng.uniform(12_000_000, 22_000_000, 50)
    wid_pred = wid_true + rng.normal(0, 500_000, 50)

    reconciler.fit_residuals(dep_true, dep_pred, wid_true, wid_pred, horizon_hours=6)

    assert reconciler.calibration_sample_size == 50
    assert reconciler.calibration_is_reliable is False
    assert reconciler.calibration_warning is not None
    assert "Limited calibration sample size" in reconciler.calibration_warning

    # Reconciled point contains diagnostics
    pt = reconciler.reconcile_point(15_000_000.0, 16_000_000.0, horizon_hours=6)
    assert pt["calibration_is_reliable"] is False
    assert pt["calibration_warning"] is not None
    assert pt["net_outflow_calibrated_p95"] <= pt["net_outflow_p95"]


def test_reliable_calibration_when_sample_size_sufficient():
    """Verify that sufficient calibration sample size (>= 150) is marked reliable."""
    reconciler = ORScenarioReconciler()
    rng = np.random.default_rng(42)

    # N = 200 (>= 150 threshold)
    dep_true = rng.uniform(10_000_000, 20_000_000, 200)
    dep_pred = dep_true + rng.normal(0, 500_000, 200)
    wid_true = rng.uniform(12_000_000, 22_000_000, 200)
    wid_pred = wid_true + rng.normal(0, 500_000, 200)

    reconciler.fit_residuals(dep_true, dep_pred, wid_true, wid_pred, horizon_hours=6)

    assert reconciler.calibration_sample_size == 200
    assert reconciler.calibration_is_reliable is True
    assert reconciler.calibration_warning is None


def test_distinction_between_calibrated_quantile_and_heuristic_proxy():
    """Verify distinction between direct calibrated net quantile and heuristic marginal proxy."""
    reconciler = ORScenarioReconciler()
    rng = np.random.default_rng(42)
    N = 100

    dep_true = rng.uniform(10_000_000, 20_000_000, N)
    dep_pred = dep_true + rng.normal(0, 500_000, N)
    wid_true = rng.uniform(12_000_000, 22_000_000, N)
    wid_pred = wid_true + rng.normal(0, 500_000, N)

    reconciler.fit_residuals(dep_true, dep_pred, wid_true, wid_pred, horizon_hours=1)
    pt = reconciler.reconcile_point(15_000_000.0, 16_000_000.0, horizon_hours=1)

    # All three quantities are distinctly accessible
    assert "net_outflow_p95" in pt
    assert "net_outflow_calibrated_p95" in pt
    assert "net_outflow_marginal_proxy" in pt

    # Marginal proxy is exactly Withdrawal_P90 - Deposit_P10
    expected_proxy = max(0.0, pt["withdrawal_p90"] - pt["deposit_p10"])
    assert pt["net_outflow_marginal_proxy"] == pytest.approx(expected_proxy)

    # Robust reserve requirement is at least as conservative as both
    assert pt["net_outflow_p95"] >= pt["net_outflow_calibrated_p95"]
    assert pt["net_outflow_p95"] >= pt["net_outflow_marginal_proxy"]


def test_net_outflow_empirical_holdout_coverage_ge_95():
    """Evaluate empirical holdout coverage across 200 origins and verify robust P95 >= 95% on all horizons."""
    df = pd.read_csv(resolve_dataset_path())
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)

    # Train strictly on first 500 rows (no future leakage)
    train_df = df.iloc[:500].copy()
    svc = ForecastService()
    svc.fit(train_df, horizons=[1, 6, 24])

    eval_origins = 200
    horizons = [1, 6, 24]
    coverage_results = {}

    for h in horizons:
        covered = []
        for idx in range(500, 500 + eval_origins):
            as_of = df.iloc[idx][TIMESTAMP_COL]
            act_wid = df.iloc[idx + h][WITHDRAWALS_COL]
            act_dep = df.iloc[idx + h][DEPOSITS_COL]
            act_net = act_wid - act_dep

            pt = svc.forecast(as_of).get_point(h)
            covered.append(act_net <= pt.net_outflow_p95)

        emp_cov = float(np.mean(covered))
        se = float(np.sqrt(emp_cov * (1.0 - emp_cov) / eval_origins))
        coverage_results[h] = {"coverage": emp_cov, "se": se}

    # Verify that robust reserve buffer achieves >= 95% coverage on all horizons
    for h in horizons:
        assert coverage_results[h]["coverage"] >= 0.95, (
            f"Horizon {h}h robust coverage {coverage_results[h]['coverage']:.1%} fell below 95% requirement!"
        )
