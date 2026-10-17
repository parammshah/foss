"""Quantitative forecast evaluation metrics: MAE, WAPE, Quantile Loss, PICP, and Calibration."""

from __future__ import annotations

from typing import Dict, Union
import numpy as np


def mean_absolute_error(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> float:
    """Mean Absolute Error (MAE): (1/N) * sum |y - y_hat|."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean(np.abs(y - p)))


def weighted_absolute_percentage_error(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> float:
    """Weighted Absolute Percentage Error (WAPE): sum |y - y_hat| / sum(y).

    Preferred over MAPE in financial cash-flow analysis because it avoids division by zero
    and weights errors by flow volume.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    denom = np.sum(np.abs(y))
    if denom == 0.0:
        return 0.0
    return float(np.sum(np.abs(y - p)) / denom)


def quantile_loss(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    quantile: float,
) -> float:
    """Quantile (Pinball) Loss for probabilistic forecast evaluation at level q in (0, 1).

    Loss = (1/N) * sum( max(q * (y - y_hat), (q - 1) * (y - y_hat)) )
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError("Quantile must be strictly between 0 and 1.")

    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    err = y - p
    return float(np.mean(np.maximum(quantile * err, (quantile - 1.0) * err)))


def prediction_interval_coverage_probability(
    y_true: np.ndarray | list[float],
    lower_bound: np.ndarray | list[float],
    upper_bound: np.ndarray | list[float],
) -> float:
    """Prediction Interval Coverage Probability (PICP).

    Fraction of true realizations that fall inside [lower_bound, upper_bound].
    For an 80% interval (P10 to P90), ideal nominal PICP is 0.80.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    lb = np.asarray(lower_bound, dtype=float).ravel()
    ub = np.asarray(upper_bound, dtype=float).ravel()

    inside = (y >= lb) & (y <= ub)
    return float(np.mean(inside))


def mean_prediction_interval_width(
    lower_bound: np.ndarray | list[float],
    upper_bound: np.ndarray | list[float],
) -> float:
    """Mean Prediction Interval Width (MPIW): measures sharpness of uncertainty bounds."""
    lb = np.asarray(lower_bound, dtype=float).ravel()
    ub = np.asarray(upper_bound, dtype=float).ravel()
    return float(np.mean(ub - lb))


def evaluate_forecast_suite(
    y_true: np.ndarray | list[float],
    y_pred_point: np.ndarray | list[float],
    y_p10: Optional[np.ndarray | list[float]] = None,
    y_p90: Optional[np.ndarray | list[float]] = None,
) -> Dict[str, float]:
    """Compute standard metrics bundle for point and interval forecasts."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred_point, dtype=float).ravel()

    results = {
        "mae": mean_absolute_error(y, p),
        "wape": weighted_absolute_percentage_error(y, p),
    }

    if y_p10 is not None and y_p90 is not None:
        lb = np.asarray(y_p10, dtype=float).ravel()
        ub = np.asarray(y_p90, dtype=float).ravel()
        results["picp_80"] = prediction_interval_coverage_probability(y, lb, ub)
        results["mpiw_80"] = mean_prediction_interval_width(lb, ub)
        results["calibration_error_80"] = abs(results["picp_80"] - 0.80)
        results["pinball_p10"] = quantile_loss(y, lb, 0.10)
        results["pinball_p90"] = quantile_loss(y, ub, 0.90)

    return results
