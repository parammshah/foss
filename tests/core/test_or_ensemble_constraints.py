"""Unit tests for Operations Research (OR) ensemble constraints and simplex projections."""

import numpy as np
import pytest
from reserve_juggler.forecasting.or_ensemble import ORConstrainedEnsemble, project_to_simplex
from reserve_juggler.forecasting.uncertainty import ORScenarioReconciler


def test_project_to_simplex():
    # Test random arbitrary vectors
    rng = np.random.default_rng(42)
    for _ in range(20):
        v = rng.normal(0, 5, 5)
        w = project_to_simplex(v)
        # All non-negative
        assert np.all(w >= -1e-8)
        # Sums to 1
        assert pytest.approx(np.sum(w)) == 1.0


def test_or_constrained_ensemble_weights():
    """Verify that ensemble solves for weights w >= 0 and sum(w) == 1 that minimize MSE."""
    rng = np.random.default_rng(123)
    N = 200
    y_true = rng.normal(20, 4, N)

    # Model 1: Good model (MSE ~ 1.0)
    pred_1 = y_true + rng.normal(0, 1.0, N)
    # Model 2: Biased model (MSE ~ 9.0)
    pred_2 = y_true + 3.0 + rng.normal(0, 1.0, N)
    # Model 3: Mediocre model (MSE ~ 4.0)
    pred_3 = y_true + rng.normal(0, 2.0, N)

    Y_hat = np.column_stack([pred_1, pred_2, pred_3])

    ensemble = ORConstrainedEnsemble()
    ensemble.fit(Y_hat, y_true, model_names=["good", "biased", "mediocre"])

    weights = ensemble.weights
    assert weights is not None
    # 1. Non-negativity constraint
    assert np.all(weights >= 0.0), f"Weights violate non-negativity: {weights}"
    # 2. Sum to one constraint
    assert pytest.approx(np.sum(weights), abs=1e-5) == 1.0
    # 3. Best model should receive higher weight than heavily biased model
    assert weights[0] > weights[1], "Good model must receive higher weight than biased model!"

    # 4. Ensemble MSE should be competitive or better than individual models
    ens_pred = ensemble.predict(Y_hat)
    ens_mse = np.mean((y_true - ens_pred) ** 2)
    assert ens_mse < np.mean((y_true - pred_2) ** 2)


def test_scenario_reconciler_properties():
    reconciler = ORScenarioReconciler()
    rng = np.random.default_rng(42)
    N = 100
    dep_true = rng.uniform(15_000_000, 25_000_000, N)
    dep_pred = dep_true + rng.normal(0, 1_000_000, N)

    wid_true = rng.uniform(14_000_000, 26_000_000, N)
    wid_pred = wid_true + rng.normal(0, 1_000_000, N)

    reconciler.fit_residuals(dep_true, dep_pred, wid_true, wid_pred)

    point = reconciler.reconcile_point(pred_deposit=20_000_000.0, pred_withdrawal=19_000_000.0, horizon_hours=6)

    # Valid non-negative quantities
    assert point["deposit_p10"] >= 0.0
    assert point["withdrawal_p10"] >= 0.0
    # Quantile monotonicity
    assert point["deposit_p10"] <= point["deposit_p50"] <= point["deposit_p90"]
    assert point["withdrawal_p10"] <= point["withdrawal_p50"] <= point["withdrawal_p90"]
    # Net outflow identity: Net Outflow = Withdrawals - Deposits
    assert point["net_outflow_p50"] == pytest.approx(point["withdrawal_p50"] - point["deposit_p50"])
    # Tail shocks monotonicity
    assert point["net_outflow_p50"] <= point["net_outflow_p95"] <= point["net_outflow_p99"]
    # Multi-hour cumulative risk bounds
    assert point["cumulative_net_outflow_p95"] <= point["cumulative_net_outflow_p99"]
