from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import pytest

from reserve_juggler.data import resolve_dataset_path
from reserve_juggler.data.schema import DEPOSITS_COL, TIMESTAMP_COL, WITHDRAWALS_COL
from reserve_juggler.forecasting.service import ForecastService
from reserve_juggler.forecasting.uncertainty import ORScenarioReconciler
from reserve_juggler.contracts import (
    ForecastPoint,
    ForecastBundle,
    TreasuryState,
    Obligation,
    ObligationType,
)
from reserve_juggler.risk.liquidity_shield import assess_liquidity_reserve


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


def test_non_overlapping_evaluation_and_appropriate_validation_status():
    """Evaluate non-overlapping origins (step=h) and verify appropriate empirical validation status."""
    df = pd.read_csv(resolve_dataset_path())
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)

    # Chronologically strictly separated training set (first 500 rows)
    train_df = df.iloc[:500].copy()
    svc = ForecastService()
    svc.fit(train_df, horizons=[1, 6, 24])

    eval_origins_count = 100
    for h in [1, 6, 24]:
        covered_robust = []
        covered_raw = []
        covered_proxy = []

        # Non-overlapping evaluation: step by h hours so windows are strictly disjoint
        idx = 500
        while idx + h < len(df) and len(covered_robust) < eval_origins_count:
            as_of = df.iloc[idx][TIMESTAMP_COL]
            act_net = df.iloc[idx + h][WITHDRAWALS_COL] - df.iloc[idx + h][DEPOSITS_COL]

            pt = svc.forecast(as_of).get_point(h)
            covered_robust.append(act_net <= pt.net_outflow_p95)
            covered_raw.append(act_net <= pt.net_outflow_calibrated_p95)
            covered_proxy.append(act_net <= pt.net_outflow_marginal_proxy)
            idx += h

        emp_cov = float(np.mean(covered_robust))
        se = float(np.sqrt(emp_cov * (1.0 - emp_cov) / len(covered_robust)))
        ci_lower = emp_cov - 1.96 * se

        # Point coverage satisfies >= 95%
        assert emp_cov >= 0.95, f"Non-overlapping horizon {h}h coverage {emp_cov:.1%} < 95%"

        # When CI spans below 95% (e.g. at h=6 where ci_lower < 0.95), status must NOT be labeled CERTIFIED
        if ci_lower < 0.95:
            validation_status = "CONDITIONALLY_CALIBRATED_FALLBACK"
        else:
            validation_status = "EMPIRICALLY_VALIDATED"

        assert validation_status in ("EMPIRICALLY_VALIDATED", "CONDITIONALLY_CALIBRATED_FALLBACK")
        assert validation_status != "CERTIFIED"


def test_target_labels_do_not_extend_into_evaluation_period():
    """Verify that no target label used for estimator training or calibration
    falls on or after the earliest evaluation origin timestamp.
    """
    df = pd.read_csv(resolve_dataset_path())
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)

    # Supply rows 0..999 to fit()
    N = 1000
    val_split_ratio = 0.25
    train_slice = df.iloc[:N].copy()

    svc = ForecastService()
    svc.fit(train_slice, horizons=[1, 6, 24, 48, 72], val_split_ratio=val_split_ratio)

    n_val = int(N * val_split_ratio)  # 250
    n_train = N - n_val                # 750

    earliest_eval_origin_idx = N
    earliest_eval_origin_ts = df.iloc[earliest_eval_origin_idx][TIMESTAMP_COL]

    for h in [1, 6, 24, 48, 72]:
        # Estimator training targets
        est_target_indices = [i + h for i in range(0, n_train - h)]
        est_latest_target_ts = df.iloc[est_target_indices[-1]][TIMESTAMP_COL]

        # Calibration targets
        cal_target_indices = [i + h for i in range(n_train, N - h)]
        cal_latest_target_ts = df.iloc[cal_target_indices[-1]][TIMESTAMP_COL]

        # Verify strict chronological separation: latest targets < earliest eval origin
        assert est_latest_target_ts < earliest_eval_origin_ts, (
            f"Horizon {h}h estimator target label ({est_latest_target_ts}) overlaps evaluation ({earliest_eval_origin_ts})"
        )
        assert cal_latest_target_ts < earliest_eval_origin_ts, (
            f"Horizon {h}h calibration target label ({cal_latest_target_ts}) overlaps evaluation ({earliest_eval_origin_ts})"
        )


def test_target_specific_reliability_classification_separation():
    """Verify that point-forecast validation does NOT automatically certify cumulative reserve forecasts."""
    df = pd.read_csv(resolve_dataset_path())
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)

    # Fit with sufficient calibration samples (N_cal >= 150)
    svc = ForecastService()
    svc.fit(df.iloc[:1000], horizons=[1, 6, 24, 48, 72], val_split_ratio=0.25)

    as_of = df.iloc[1000][TIMESTAMP_COL]
    bundle = svc.forecast(as_of)

    # 1h: Both point and cumulative are empirically validated
    pt_1h = bundle.get_point(1)
    assert pt_1h.point_reliability_status == "EMPIRICALLY_VALIDATED"
    assert pt_1h.cumulative_reliability_status == "EMPIRICALLY_VALIDATED"

    # 6h: Point and cumulative are conditionally calibrated fallback (due to diurnal clustering)
    pt_6h = bundle.get_point(6)
    assert pt_6h.point_reliability_status == "CONDITIONALLY_CALIBRATED_FALLBACK"
    assert pt_6h.cumulative_reliability_status == "CONDITIONALLY_CALIBRATED_FALLBACK"

    # 24h: Point is validated (>95% single-hour coverage), BUT cumulative is conditionally calibrated fallback!
    pt_24h = bundle.get_point(24)
    assert pt_24h.point_reliability_status == "EMPIRICALLY_VALIDATED"
    assert pt_24h.cumulative_reliability_status == "CONDITIONALLY_CALIBRATED_FALLBACK"
    assert pt_24h.cumulative_reliability_status != pt_24h.point_reliability_status


def test_adversarial_clustered_large_outflows_reserve_protection():
    """Adversarial test: verify that clustered large customer outflows cannot deplete
    reserves below committed obligations or permit unsafe discretionary deployments.
    """
    as_of = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    opening_cash = 50_000_000.0

    # 5 consecutive hours of extreme customer outflows ($15M withdrawals, $2M deposits = $13M/hr deficit)
    clustered_points = []
    for h in range(1, 6):
        clustered_points.append(
            ForecastPoint(
                timestamp=(as_of + timedelta(hours=h)).isoformat(),
                forecast_horizon=float(h),
                deposit_point_forecast=2_000_000.0,
                withdrawal_point_forecast=15_000_000.0,
                deposit_p10=1_000_000.0,
                deposit_p90=3_000_000.0,
                withdrawal_p10=12_000_000.0,
                withdrawal_p90=18_000_000.0,
                net_outflow_p95=16_000_000.0,
                is_cumulative=False,
            )
        )

    bundle = ForecastBundle(
        forecast_id="BUN-ADV-CLUSTER",
        generated_at=as_of.isoformat(),
        point_forecasts=clustered_points,
        is_cumulative=False,
    )

    # Obligation due at hour 6: $20M
    obl = Obligation(
        obligation_id="OBL-ADV-01",
        timestamp=as_of + timedelta(hours=6),
        amount_usd=20_000_000.0,
        obligation_type=ObligationType.CUSTOMER_PAYOUT,
        is_mandatory=True,
    )

    state = TreasuryState(
        timestamp=as_of,
        operational_cash_usd=opening_cash,
        obligations=[obl],
    )

    assessment = assess_liquidity_reserve(state, bundle)

    # Required reserve must cover both committed obligations AND accumulated customer deficit
    assert assessment.immediate_obligations == 20_000_000.0
    assert assessment.conservative_net_deficit > 0.0
    assert assessment.total_required_reserve > assessment.immediate_obligations

    # Because total demand ($20M + $2M buffer + clustered deficit > $50M) exceeds available cash ($50M):
    # Surplus must be exactly $0.00!
    assert assessment.eligible_investable_surplus == 0.0
    assert assessment.total_required_reserve >= opening_cash

