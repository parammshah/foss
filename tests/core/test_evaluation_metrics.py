"""Unit tests for quantitative evaluation metrics: MAE, WAPE, Quantile Loss, PICP, and Calibration."""

import numpy as np
import pytest
from reserve_juggler.evaluation.metrics import (
    evaluate_forecast_suite,
    mean_absolute_error,
    mean_prediction_interval_width,
    prediction_interval_coverage_probability,
    quantile_loss,
    weighted_absolute_percentage_error,
)


def test_mae_and_wape():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 330.0])

    # Absolute errors: 10, 10, 30. Sum = 50. MAE = 50/3 = 16.6667
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(50.0 / 3.0)
    # WAPE = 50 / (100+200+300) = 50 / 600 = 0.083333
    assert weighted_absolute_percentage_error(y_true, y_pred) == pytest.approx(50.0 / 600.0)


def test_quantile_pinball_loss():
    # If y = 10, pred = 8, error = 2 (underprediction)
    # For q = 0.9: loss = 0.9 * 2 = 1.8
    assert quantile_loss([10.0], [8.0], 0.9) == pytest.approx(1.8)

    # If y = 10, pred = 12, error = -2 (overprediction)
    # For q = 0.9: loss = (0.9 - 1) * (-2) = -0.1 * -2 = 0.2
    assert quantile_loss([10.0], [12.0], 0.9) == pytest.approx(0.2)


def test_picp_coverage_and_width():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    lb = np.array([8.0, 15.0, 25.0, 45.0])   # 40 is outside (40 < 45)
    ub = np.array([12.0, 25.0, 35.0, 50.0])

    # 3 out of 4 are inside: PICP = 0.75
    assert prediction_interval_coverage_probability(y_true, lb, ub) == 0.75

    # Width: (4 + 10 + 10 + 5) / 4 = 29 / 4 = 7.25
    assert mean_prediction_interval_width(lb, ub) == 7.25


def test_evaluate_forecast_suite():
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([11.0, 19.0, 31.0, 39.0, 50.0])
    y_p10 = y_pred - 5.0
    y_p90 = y_pred + 5.0

    res = evaluate_forecast_suite(y_true, y_pred, y_p10, y_p90)
    assert "mae" in res
    assert "wape" in res
    assert "picp_80" in res
    assert "calibration_error_80" in res
    assert res["picp_80"] == 1.0  # All inside interval
