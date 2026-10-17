"""Operations Research (OR) quantile scenario estimation and cash-flow accounting reconciliation."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np


class ORScenarioReconciler:
    """Computes quantile prediction intervals and reconciles multi-hour net-outflow shock scenarios.

    Ensures that generated scenarios maintain valid financial quantities:
    - Gross deposits >= 0, gross withdrawals >= 0.
    - Quantile monotonicity: P10 <= P50 <= P90.
    - Net Outflow conservation: Net Outflow = Withdrawals - Deposits.
    - Multi-hour cumulative stress scenarios account for temporal autocorrelation.
    """

    def __init__(self) -> None:
        self.deposit_residual_quantiles: Dict[float, float] = {
            2.5: 0.0, 10.0: 0.0, 25.0: 0.0, 50.0: 0.0, 75.0: 0.0, 90.0: 0.0, 97.5: 0.0
        }
        self.withdrawal_residual_quantiles: Dict[float, float] = {
            2.5: 0.0, 10.0: 0.0, 25.0: 0.0, 50.0: 0.0, 75.0: 0.0, 90.0: 0.0, 97.5: 0.0
        }
        self.net_outflow_residual_quantiles: Dict[float, float] = {50.0: 0.0, 95.0: 0.0, 99.0: 0.0}
        self.net_autocorr_factor: float = 1.0
        self.net_std: float = 1_000_000.0
        self.calibration_sample_size: int = 0

    def fit_residuals(
        self,
        true_deposits: np.ndarray,
        pred_deposits: np.ndarray,
        true_withdrawals: np.ndarray,
        pred_withdrawals: np.ndarray,
    ) -> ORScenarioReconciler:
        """Fit empirical residual distributions on validation window."""
        res_dep = true_deposits - pred_deposits
        res_wid = true_withdrawals - pred_withdrawals
        res_net = (true_withdrawals - true_deposits) - (pred_withdrawals - pred_deposits)
        self.calibration_sample_size = len(res_dep)

        # Deposit quantiles
        for q in [2.5, 10.0, 25.0, 50.0, 75.0, 90.0, 97.5]:
            self.deposit_residual_quantiles[q] = float(np.percentile(res_dep, q))

        # Withdrawal quantiles
        for q in [2.5, 10.0, 25.0, 50.0, 75.0, 90.0, 97.5]:
            self.withdrawal_residual_quantiles[q] = float(np.percentile(res_wid, q))

        # Net outflow quantiles
        self.net_outflow_residual_quantiles[50.0] = float(np.percentile(res_net, 50))
        self.net_outflow_residual_quantiles[95.0] = float(np.percentile(res_net, 95))
        self.net_outflow_residual_quantiles[99.0] = float(np.percentile(res_net, 99))

        self.net_std = float(np.std(res_net))

        # Estimate lag-1 autocorrelation for cumulative horizon scaling
        if len(res_net) > 2:
            c1 = np.corrcoef(res_net[:-1], res_net[1:])[0, 1]
            rho1 = max(0.0, float(c1)) if not np.isnan(c1) else 0.0
            self.net_autocorr_factor = float(np.sqrt(1.0 + 2.0 * rho1))
        else:
            self.net_autocorr_factor = 1.0

        return self

    def reconcile_point(
        self,
        pred_deposit: float,
        pred_withdrawal: float,
        horizon_hours: int,
    ) -> Dict[str, float]:
        """Construct coherent reconciled quantiles for a single forecast point."""
        # 1. Deposit quantiles
        d025 = max(0.0, pred_deposit + self.deposit_residual_quantiles[2.5])
        d10 = max(0.0, pred_deposit + self.deposit_residual_quantiles[10.0])
        d25 = max(0.0, pred_deposit + self.deposit_residual_quantiles[25.0])
        d50 = max(0.0, pred_deposit + self.deposit_residual_quantiles[50.0])
        d75 = max(0.0, pred_deposit + self.deposit_residual_quantiles[75.0])
        d90 = max(0.0, pred_deposit + self.deposit_residual_quantiles[90.0])
        d975 = max(0.0, pred_deposit + self.deposit_residual_quantiles[97.5])

        # Strict Monotonicity adjustment
        d10 = max(d025, d10)
        d25 = max(d10, d25)
        d50 = max(d25, d50)
        d75 = max(d50, d75)
        d90 = max(d75, d90)
        d975 = max(d90, d975)

        # 2. Withdrawal quantiles
        w025 = max(0.0, pred_withdrawal + self.withdrawal_residual_quantiles[2.5])
        w10 = max(0.0, pred_withdrawal + self.withdrawal_residual_quantiles[10.0])
        w25 = max(0.0, pred_withdrawal + self.withdrawal_residual_quantiles[25.0])
        w50 = max(0.0, pred_withdrawal + self.withdrawal_residual_quantiles[50.0])
        w75 = max(0.0, pred_withdrawal + self.withdrawal_residual_quantiles[75.0])
        w90 = max(0.0, pred_withdrawal + self.withdrawal_residual_quantiles[90.0])
        w975 = max(0.0, pred_withdrawal + self.withdrawal_residual_quantiles[97.5])

        # Strict Monotonicity adjustment
        w10 = max(w025, w10)
        w25 = max(w10, w25)
        w50 = max(w25, w50)
        w75 = max(w50, w75)
        w90 = max(w75, w90)
        w975 = max(w90, w975)

        # 3. Net Outflow point quantiles (Net Outflow = Withdrawals - Deposits)
        net_50 = w50 - d50
        net_95 = net_50 + max(0.0, self.net_outflow_residual_quantiles[95.0])
        net_99 = net_50 + max(net_95 - net_50, self.net_outflow_residual_quantiles[99.0])

        # 4. Multi-hour cumulative net outflow buffer
        # Cumulative shock scales with sqrt(horizon) * autocorrelation factor
        h_factor = np.sqrt(float(horizon_hours)) * self.net_autocorr_factor
        z95 = 1.645
        z99 = 2.326

        cum_net_95 = (net_50 * horizon_hours) + (z95 * self.net_std * h_factor)
        cum_net_99 = (net_50 * horizon_hours) + (z99 * self.net_std * h_factor)

        # Cumulative P99 >= Cumulative P95
        cum_net_99 = max(cum_net_99, cum_net_95)

        return {
            "deposit_p025": float(d025),
            "deposit_p10": float(d10),
            "deposit_p25": float(d25),
            "deposit_p50": float(d50),
            "deposit_p75": float(d75),
            "deposit_p90": float(d90),
            "deposit_p975": float(d975),
            "withdrawal_p025": float(w025),
            "withdrawal_p10": float(w10),
            "withdrawal_p25": float(w25),
            "withdrawal_p50": float(w50),
            "withdrawal_p75": float(w75),
            "withdrawal_p90": float(w90),
            "withdrawal_p975": float(w975),
            "net_outflow_p50": float(net_50),
            "net_outflow_p95": float(net_95),
            "net_outflow_p99": float(net_99),
            "cumulative_net_outflow_p95": float(cum_net_95),
            "cumulative_net_outflow_p99": float(cum_net_99),
            "calibration_sample_size": self.calibration_sample_size,
        }
