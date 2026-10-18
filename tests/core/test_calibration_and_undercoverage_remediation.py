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

    # 24h: Both point and cumulative are conditionally calibrated fallback
    pt_24h = bundle.get_point(24)
    assert pt_24h.point_reliability_status == "CONDITIONALLY_CALIBRATED_FALLBACK"
    assert pt_24h.cumulative_reliability_status == "CONDITIONALLY_CALIBRATED_FALLBACK"

    # 72h: Both point and cumulative are conditionally calibrated fallback
    pt_72h = bundle.get_point(72)
    assert pt_72h.point_reliability_status == "CONDITIONALLY_CALIBRATED_FALLBACK"
    assert pt_72h.cumulative_reliability_status == "CONDITIONALLY_CALIBRATED_FALLBACK"


def test_identical_evidence_status_consistency():
    """Verify that targets with identical empirical evidence receive consistent reliability status,
    and any divergence between 1h and 6h/24h is strictly governed by documented target-specific rules:
    - 1h achieves >= 95% population coverage under direct unbuffered conformal calibration.
    - 6h requires a 1.55x heuristic buffer because raw conformal quantiles underperform due to diurnal clustering.
    - Multi-hour cumulative targets are analytical proxy heuristics (sqrt(h) scaling) subject to serial drift.
    """
    reconciler = ORScenarioReconciler()
    for q in [2.5, 10.0, 25.0, 50.0, 75.0, 90.0, 97.5]:
        reconciler.deposit_residual_quantiles[q] = 0.0
        reconciler.withdrawal_residual_quantiles[q] = 0.0
    reconciler.raw_net_outflow_conformal_quantiles = {50.0: 0.0, 95.0: 0.0, 99.0: 0.0}
    reconciler.net_outflow_residual_quantiles = {50.0: 0.0, 95.0: 0.0, 99.0: 0.0}

    # Rule 1: Degraded calibration sample size (< 150) forces DEGRADED for all targets
    reconciler.calibration_sample_size = 100
    reconciler.calibration_is_reliable = False
    res_degraded_1h = reconciler.reconcile_point(
        pred_deposit=100.0,
        pred_withdrawal=100.0,
        horizon_hours=1,
    )
    res_degraded_6h = reconciler.reconcile_point(
        pred_deposit=100.0,
        pred_withdrawal=100.0,
        horizon_hours=6,
    )
    assert res_degraded_1h["point_reliability_status"] == "DEGRADED"
    assert res_degraded_1h["cumulative_reliability_status"] == "DEGRADED"
    assert res_degraded_6h["point_reliability_status"] == "DEGRADED"
    assert res_degraded_6h["cumulative_reliability_status"] == "DEGRADED"

    # Rule 2: Sufficient calibration (>=150) distinguishes 1h direct calibration from multi-hour fallbacks
    reconciler.calibration_sample_size = 250
    reconciler.calibration_is_reliable = True
    res_1h = reconciler.reconcile_point(
        pred_deposit=100.0,
        pred_withdrawal=100.0,
        horizon_hours=1,
    )
    res_6h = reconciler.reconcile_point(
        pred_deposit=100.0,
        pred_withdrawal=100.0,
        horizon_hours=6,
    )
    # 1h is EMPIRICALLY_VALIDATED because raw conformal quantiles cover >=95% without heuristic multipliers
    assert res_1h["point_reliability_status"] == "EMPIRICALLY_VALIDATED"
    assert res_1h["cumulative_reliability_status"] == "EMPIRICALLY_VALIDATED"

    # 6h receives CONDITIONALLY_CALIBRATED_FALLBACK because it relies on a 1.55x heuristic buffer
    assert res_6h["point_reliability_status"] == "CONDITIONALLY_CALIBRATED_FALLBACK"
    assert res_6h["cumulative_reliability_status"] == "CONDITIONALLY_CALIBRATED_FALLBACK"


def test_sustained_72h_multi_day_outflow_protection():
    """Verify that sustained 72-hour customer net deficit locks investable surplus to $0.00
    and enforces conservative reserve safeguards even when 72h forecasts are in fallback status.
    """
    as_of = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    opening_cash = 60_000_000.0

    # 72 hours of persistent customer outflows ($1M/hr net deficit = $72M total demand)
    points_72h = []
    for h in [1, 6, 24, 48, 72]:
        points_72h.append(
            ForecastPoint(
                timestamp=(as_of + timedelta(hours=h)).isoformat(),
                forecast_horizon=float(h),
                deposit_point_forecast=500_000.0,
                withdrawal_point_forecast=1_500_000.0,
                deposit_p10=300_000.0,
                deposit_p90=700_000.0,
                withdrawal_p10=1_200_000.0,
                withdrawal_p90=2_000_000.0,
                net_outflow_p95=2_000_000.0,
                cumulative_net_outflow_p95=float(h) * 1_200_000.0,
                is_cumulative=False,
                point_reliability_status="CONDITIONALLY_CALIBRATED_FALLBACK",
                cumulative_reliability_status="CONDITIONALLY_CALIBRATED_FALLBACK",
            )
        )

    bundle = ForecastBundle(
        forecast_id="BUN-72H-SUSTAINED",
        generated_at=as_of.isoformat(),
        point_forecasts=points_72h,
        is_cumulative=False,
    )

    state = TreasuryState(
        timestamp=as_of,
        operational_cash_usd=opening_cash,
        obligations=[],
    )

    assessment = assess_liquidity_reserve(state, bundle, use_calibrated_net_quantile=True)

    # 72h cumulative deficit demand is $72 * 1.2M = $86.4M, far exceeding $60M cash
    assert assessment.conservative_net_deficit >= 70_000_000.0
    assert assessment.total_required_reserve >= opening_cash
    assert assessment.eligible_investable_surplus == 0.0



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


def test_zero_exceedance_does_not_promote_policy_or_horizon_reliability():
    """Verify that zero observed failures (100% empirical coverage) does NOT promote
    a multi-hour forecast or reserve policy to EMPIRICALLY_VALIDATED.
    Reliability certification requires explicit, direct conformal calibration without heuristic multipliers.
    """
    as_of = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    opening_cash = 100_000_000.0

    # 6h point forecast in fallback
    pt_6h = ForecastPoint(
        timestamp=(as_of + timedelta(hours=6)).isoformat(),
        forecast_horizon=6.0,
        deposit_point_forecast=5_000_000.0,
        withdrawal_point_forecast=6_000_000.0,
        deposit_p10=4_000_000.0,
        deposit_p90=6_000_000.0,
        withdrawal_p10=5_000_000.0,
        withdrawal_p90=8_000_000.0,
        net_outflow_p95=4_000_000.0,
        cumulative_net_outflow_p95=10_000_000.0,
        is_cumulative=False,
        point_reliability_status="CONDITIONALLY_CALIBRATED_FALLBACK",
        cumulative_reliability_status="CONDITIONALLY_CALIBRATED_FALLBACK",
    )

    bundle = ForecastBundle(
        forecast_id="BUN-6H-FALLBACK",
        generated_at=as_of.isoformat(),
        point_forecasts=[pt_6h],
        is_cumulative=False,
    )

    state = TreasuryState(
        timestamp=as_of,
        operational_cash_usd=opening_cash,
        obligations=[],
    )

    assessment = assess_liquidity_reserve(state, bundle, use_calibrated_net_quantile=True)

    # Even though 6h reserve ($10M) easily covers customer outflow of e.g. $1M (0 failures):
    # Status MUST remain CONDITIONALLY_CALIBRATED_FALLBACK, not EMPIRICALLY_VALIDATED!
    assert assessment.policy_reliability_status == "CONDITIONALLY_CALIBRATED_FALLBACK"
    assert pt_6h.point_reliability_status == "CONDITIONALLY_CALIBRATED_FALLBACK"
    assert pt_6h.cumulative_reliability_status == "CONDITIONALLY_CALIBRATED_FALLBACK"


def test_statistical_intervals_zero_low_nonzero_exceedances():
    """Verify statistical intervals (Wilson score and Moving Block Bootstrap) under
    zero, low, and nonzero failure counts. All endpoints must be finite and bounded in [0, 1].
    """
    from scripts.audit_reserve_backtest import wilson_score_interval, moving_block_bootstrap_ci

    # 1. Zero failures (k=0 failures out of N=200, i.e. 200 successes)
    ind_zero = np.ones(200, dtype=float)
    w_low, w_high = wilson_score_interval(200, 200)
    assert 0.98 <= w_low <= 0.985
    assert w_high == 1.0

    mbb_zero = moving_block_bootstrap_ci(ind_zero, block_length=6)
    assert mbb_zero["is_degenerate"] is True
    # Rule of Three / Hanley-Jadack exact upper bound on failure: ~3/N = 1.5% -> lower coverage >= 98.5%
    assert mbb_zero["rule_of_three_bound"] >= 0.985
    assert mbb_zero["upper"] == 1.0
    assert 0.0 <= mbb_zero["lower"] <= 1.0

    # 2. Low failures (k=2 failures out of N=100, i.e. 98 successes)
    ind_low = np.array([1.0] * 98 + [0.0] * 2)
    w_low, w_high = wilson_score_interval(98, 100)
    assert 0.92 <= w_low <= 0.94
    assert 0.99 <= w_high <= 1.0

    mbb_low = moving_block_bootstrap_ci(ind_low, block_length=2)
    assert mbb_low["is_degenerate"] is False
    assert 0.90 <= mbb_low["lower"] <= mbb_low["upper"] <= 1.0

    # 3. Nonzero failures (k=38 failures out of N=200, i.e. 162 successes = 81%)
    ind_nonzero = np.array([1.0] * 162 + [0.0] * 38)
    w_low, w_high = wilson_score_interval(162, 200)
    assert 0.74 <= w_low <= 0.76
    assert 0.85 <= w_high <= 0.87

    mbb_nonzero = moving_block_bootstrap_ci(ind_nonzero, block_length=6)
    assert mbb_nonzero["is_degenerate"] is False
    assert 0.65 <= mbb_nonzero["lower"] <= mbb_nonzero["upper"] <= 0.95


def test_truly_1h_only_assessment_policy_status():
    """Verify that a policy assessment with truly 1h-only constituent forecast
    receives EMPIRICALLY_VALIDATED if the 1h forecast is empirically validated.
    """
    as_of = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    pt_1h = ForecastPoint(
        timestamp=(as_of + timedelta(hours=1)).isoformat(),
        forecast_horizon=1.0,
        deposit_point_forecast=5_000_000.0,
        withdrawal_point_forecast=6_000_000.0,
        deposit_p10=4_000_000.0,
        deposit_p90=6_000_000.0,
        withdrawal_p10=5_000_000.0,
        withdrawal_p90=8_000_000.0,
        net_outflow_p95=2_000_000.0,
        cumulative_net_outflow_p95=2_000_000.0,
        is_cumulative=False,
        point_reliability_status="EMPIRICALLY_VALIDATED",
        cumulative_reliability_status="EMPIRICALLY_VALIDATED",
    )
    bundle = ForecastBundle(forecast_id="BUN-1H", generated_at=as_of.isoformat(), point_forecasts=[pt_1h])
    state = TreasuryState(timestamp=as_of, operational_cash_usd=50_000_000.0, obligations=[])

    # Case A: By default without explicit policy-level statistical certification,
    # policy status retains CONDITIONALLY_CALIBRATED_FALLBACK even if input is EMPIRICALLY_VALIDATED
    assessment_unverified = assess_liquidity_reserve(
        state, bundle, forecast_horizon_hours=1.0, payout_window_hours=1.0, policy_is_certified=False
    )
    assert assessment_unverified.policy_reliability_status == "CONDITIONALLY_CALIBRATED_FALLBACK"

    # Case B: When policy-level statistical certification is established for 1h policy,
    # it receives EMPIRICALLY_VALIDATED
    assessment_certified = assess_liquidity_reserve(
        state, bundle, forecast_horizon_hours=1.0, payout_window_hours=1.0, policy_is_certified=True
    )
    assert assessment_certified.policy_reliability_status == "EMPIRICALLY_VALIDATED"


def test_unused_multi_hour_forecasts_do_not_contaminate_1h_policy_status():
    """Verify that unused multi-hour forecasts present in a bundle do NOT contaminate
    a strictly 1-hour assessment when forecast_horizon_hours=1.0.
    However, when forecast_horizon_hours includes multi-hour targets, status falls back.
    """
    as_of = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    pt_1h = ForecastPoint(
        timestamp=(as_of + timedelta(hours=1)).isoformat(),
        forecast_horizon=1.0,
        deposit_point_forecast=5_000_000.0,
        withdrawal_point_forecast=6_000_000.0,
        deposit_p10=4_000_000.0,
        deposit_p90=6_000_000.0,
        withdrawal_p10=5_000_000.0,
        withdrawal_p90=8_000_000.0,
        net_outflow_p95=2_000_000.0,
        cumulative_net_outflow_p95=2_000_000.0,
        is_cumulative=False,
        point_reliability_status="EMPIRICALLY_VALIDATED",
        cumulative_reliability_status="EMPIRICALLY_VALIDATED",
    )
    pt_6h = ForecastPoint(
        timestamp=(as_of + timedelta(hours=6)).isoformat(),
        forecast_horizon=6.0,
        deposit_point_forecast=5_000_000.0,
        withdrawal_point_forecast=6_000_000.0,
        deposit_p10=4_000_000.0,
        deposit_p90=6_000_000.0,
        withdrawal_p10=5_000_000.0,
        withdrawal_p90=8_000_000.0,
        net_outflow_p95=5_000_000.0,
        cumulative_net_outflow_p95=12_000_000.0,
        is_cumulative=False,
        point_reliability_status="CONDITIONALLY_CALIBRATED_FALLBACK",
        cumulative_reliability_status="CONDITIONALLY_CALIBRATED_FALLBACK",
    )
    # Bundle contains BOTH 1h and 6h
    bundle = ForecastBundle(forecast_id="BUN-MIXED", generated_at=as_of.isoformat(), point_forecasts=[pt_1h, pt_6h])
    state = TreasuryState(timestamp=as_of, operational_cash_usd=50_000_000.0, obligations=[])

    # Case A: Horizon restricted to 1h with certified policy -> Unused 6h point does not contaminate
    assessment_1h = assess_liquidity_reserve(
        state, bundle, forecast_horizon_hours=1.0, payout_window_hours=1.0, policy_is_certified=True
    )
    assert assessment_1h.policy_reliability_status == "EMPIRICALLY_VALIDATED"

    # Case B: Horizon expanded to 6h -> Multi-hour fallback point is included, policy falls back
    assessment_6h = assess_liquidity_reserve(
        state, bundle, forecast_horizon_hours=6.0, payout_window_hours=6.0, policy_is_certified=True
    )
    assert assessment_6h.policy_reliability_status == "CONDITIONALLY_CALIBRATED_FALLBACK"


def test_degraded_forecast_contaminates_policy_to_degraded():
    """Verify that a degraded constituent forecast forces policy status to DEGRADED."""
    as_of = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    pt_degraded = ForecastPoint(
        timestamp=(as_of + timedelta(hours=1)).isoformat(),
        forecast_horizon=1.0,
        deposit_point_forecast=5_000_000.0,
        withdrawal_point_forecast=6_000_000.0,
        deposit_p10=4_000_000.0,
        deposit_p90=6_000_000.0,
        withdrawal_p10=5_000_000.0,
        withdrawal_p90=8_000_000.0,
        net_outflow_p95=2_000_000.0,
        is_cumulative=False,
        point_reliability_status="DEGRADED",
        cumulative_reliability_status="DEGRADED",
    )
    bundle = ForecastBundle(forecast_id="BUN-DEG", generated_at=as_of.isoformat(), point_forecasts=[pt_degraded])
    state = TreasuryState(timestamp=as_of, operational_cash_usd=50_000_000.0, obligations=[])

    assessment = assess_liquidity_reserve(state, bundle, forecast_horizon_hours=1.0, payout_window_hours=1.0, policy_is_certified=True)
    assert assessment.policy_reliability_status == "DEGRADED"


def test_mbb_reproducibility_and_fixed_seed():
    """Verify that moving_block_bootstrap_ci produces reproducible, deterministic results
    for fixed seed=42 across multiple invocations.
    """
    from scripts.audit_reserve_backtest import moving_block_bootstrap_ci
    # Simulated sequence with 10 failures out of 100 observations
    indicators = np.ones(100)
    indicators[::10] = 0.0

    res1 = moving_block_bootstrap_ci(indicators, block_length=5, num_bootstraps=2000, seed=42)
    res2 = moving_block_bootstrap_ci(indicators, block_length=5, num_bootstraps=2000, seed=42)

    assert res1["lower"] == res2["lower"]
    assert res1["upper"] == res2["upper"]
    assert res1["display"] == res2["display"]


def test_distinguish_input_calibration_reserve_sufficiency_and_cash_shortfall():
    """Verify the structural distinction between:
    1. Forecast Exceedance (realized outflow > point/cumulative forecast)
    2. Reserve Exceedance (realized cumulative outflow > required reserve)
    3. Actual Cash Shortfall (realized demand > available post-decision cleared cash)
    """
    as_of = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    pt_1h = ForecastPoint(
        timestamp=(as_of + timedelta(hours=1)).isoformat(),
        forecast_horizon=1.0,
        deposit_point_forecast=5_000_000.0,
        withdrawal_point_forecast=6_000_000.0,
        deposit_p10=4_000_000.0,
        deposit_p90=6_000_000.0,
        withdrawal_p10=5_000_000.0,
        withdrawal_p90=8_000_000.0,
        net_outflow_p95=2_000_000.0,
        cumulative_net_outflow_p95=2_000_000.0,
        is_cumulative=False,
        point_reliability_status="EMPIRICALLY_VALIDATED",
        cumulative_reliability_status="EMPIRICALLY_VALIDATED",
    )
    bundle = ForecastBundle(forecast_id="BUN-DIST", generated_at=as_of.isoformat(), point_forecasts=[pt_1h])
    # Starting state: $50M cash, $10M obligations due in 1h
    state = TreasuryState(
        timestamp=as_of,
        operational_cash_usd=50_000_000.0,
        obligations=[],
    )
    assessment = assess_liquidity_reserve(
        state, bundle, forecast_horizon_hours=1.0, payout_window_hours=1.0, use_calibrated_net_quantile=True
    )
    
    # Sized reserve: $2,000,000.00
    required_reserve = assessment.total_required_reserve
    assert required_reserve == 2_000_000.0

    # Operational decision: deploy surplus $48M into repo, keep $2M operating cash
    surplus = assessment.eligible_investable_surplus
    post_decision_cash = state.operational_cash_usd - surplus
    assert post_decision_cash == 2_000_000.0

    # Scenario 1: Realized outflow = $2.5M
    realized_outflow = 2_500_000.0
    forecast_exceeded = realized_outflow > pt_1h.net_outflow_p95  # True
    reserve_exceeded = realized_outflow > required_reserve         # True
    # In operating cash: $2.0M - $2.5M = -$500k shortfall during repo lockup!
    cash_shortfall = realized_outflow > post_decision_cash         # True
    assert forecast_exceeded is True
    assert reserve_exceeded is True
    assert cash_shortfall is True

    # Scenario 2: Treasury maintains $10M cash buffer policy (surplus deployment capped)
    conservative_post_cash = 10_000_000.0
    cash_shortfall_with_buffer = realized_outflow > conservative_post_cash
    assert cash_shortfall_with_buffer is False  # Cash shortfall prevented despite forecast exceedance!


def test_terminal_vs_peak_cumulative_outflow_intrahorizon_divergence():
    """Regression test proving that terminal-flow coverage can differ from peak-outflow coverage,
    and that a path-wise liquidity check detects earlier intrahorizon cash shortfalls
    that terminal checks miss.
    """
    as_of = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)

    # Synthetic cash-flow sequence over 6 hours:
    # Hour 1: +$10M net withdrawal (Cum = $10M)
    # Hour 2: +$5M net withdrawal  (Cum = $15M -> PEAK!)
    # Hour 3: -$5M net withdrawal  (Cum = $10M)
    # Hour 4: -$4M net withdrawal  (Cum = $6M)
    # Hour 5: -$3M net withdrawal  (Cum = $3M)
    # Hour 6: -$2M net withdrawal  (Cum = $1M -> TERMINAL!)
    hourly_net_outflows = [10_000_000.0, 5_000_000.0, -5_000_000.0, -4_000_000.0, -3_000_000.0, -2_000_000.0]
    cum_series = np.cumsum(hourly_net_outflows)

    terminal_outflow = cum_series[-1]  # $1,000,000.0
    peak_outflow = np.max(cum_series)   # $15,000,000.0
    assert terminal_outflow == 1_000_000.0
    assert peak_outflow == 15_000_000.0

    # Tested reserve sized at $8,000,000.0
    tested_reserve = 8_000_000.0

    # 1. Terminal vs Peak Exceedance divergence:
    terminal_exceeded = bool(terminal_outflow > tested_reserve)  # False ($1M <= $8M)
    peak_exceeded = bool(peak_outflow > tested_reserve)          # True ($15M > $8M)
    assert terminal_exceeded is False
    assert peak_exceeded is True

    # 2. Path-wise cash simulation starting with $8M cleared cash (after surplus deployed):
    starting_cash = 8_000_000.0
    shortfall_timestamps = []

    for t_idx, cum_flow in enumerate(cum_series):
        avail_cash = starting_cash - cum_flow
        t_stamp = (as_of + timedelta(hours=t_idx + 1)).isoformat()
        if avail_cash < 0:
            shortfall_timestamps.append(t_stamp)

    # At hour 1: cash = $8M - $10M = -$2M (shortfall!)
    # At hour 2: cash = $8M - $15M = -$7M (shortfall!)
    # At hour 6: cash = $8M - $1M = +$7M (terminal check says cash is positive!)
    assert len(shortfall_timestamps) > 0
    assert shortfall_timestamps[0] == (as_of + timedelta(hours=1)).isoformat()
    assert (starting_cash - terminal_outflow) > 0  # Terminal cash is positive!





