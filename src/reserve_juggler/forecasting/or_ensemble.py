"""Operations Research (OR) constrained forecast combination ensemble.

Solves a convex constrained optimization problem to find optimal model weights:
    min_{w} (1/2) * || y - Y_hat * w ||_2^2 + (gamma/2) * || w - (1/M) ||_2^2
    subject to:
        w_i >= 0  for all i = 1..M
        sum(w_i) == 1.0
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of vector v onto the probability simplex:
    w >= 0, sum(w) = 1.
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = float(cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    return w / np.sum(w)


class ORConstrainedEnsemble:
    """Operations Research constrained optimization solver for forecast combination."""

    def __init__(self, regularization: float = 1e-4) -> None:
        self.regularization = regularization
        self.weights: Optional[np.ndarray] = None
        self.model_names: List[str] = []
        self.validation_loss: float = 0.0

    def fit(
        self,
        Y_hat: np.ndarray,
        y_true: np.ndarray,
        model_names: Optional[List[str]] = None,
    ) -> ORConstrainedEnsemble:
        """Solves the constrained quadratic optimization problem.

        Args:
            Y_hat: Matrix of shape (N, M) containing out-of-sample predictions from M candidate models.
            y_true: Ground truth target vector of shape (N,).
            model_names: Optional labels for each candidate model.
        """
        N, M = Y_hat.shape
        self.model_names = model_names or [f"model_{i}" for i in range(M)]

        if M == 1:
            self.weights = np.array([1.0])
            self.validation_loss = float(np.mean((y_true - Y_hat[:, 0]) ** 2))
            return self

        # Scale by target magnitude to ensure numerical stability in quadratic optimization
        scale = max(1.0, float(np.mean(np.abs(y_true))))
        y_norm = y_true / scale
        Y_norm = Y_hat / scale

        # Objective function: Normalized MSE + L2 shrinkage towards uniform weights
        uniform_weights = np.ones(M) / float(M)

        def objective(w: np.ndarray) -> float:
            pred = Y_norm @ w
            mse = 0.5 * np.mean((y_norm - pred) ** 2)
            shrinkage = 0.5 * self.regularization * np.sum((w - uniform_weights) ** 2)
            return mse + shrinkage

        def gradient(w: np.ndarray) -> np.ndarray:
            pred = Y_norm @ w
            grad_mse = (Y_norm.T @ (pred - y_norm)) / float(N)
            grad_shrinkage = self.regularization * (w - uniform_weights)
            return grad_mse + grad_shrinkage


        # Initial point: uniform distribution
        w0 = uniform_weights.copy()
        bounds = [(0.0, 1.0) for _ in range(M)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(
            objective,
            w0,
            method="SLSQP",
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-9},
        )

        if res.success:
            w_opt = res.x
            # Clean up slight numerical noise
            w_opt = np.maximum(0.0, w_opt)
            self.weights = w_opt / np.sum(w_opt)
        else:
            # Projected gradient fallback
            w = w0.copy()
            lr = 0.1
            for _ in range(300):
                grad = gradient(w)
                w = project_to_simplex(w - lr * grad)
            self.weights = w

        # Verify exact OR constraints
        assert np.all(self.weights >= -1e-8), "Negative weights violate OR constraints"
        assert abs(np.sum(self.weights) - 1.0) < 1e-5, "Weights must sum to 1.0"
        self.weights = np.clip(self.weights, 0.0, 1.0)
        self.weights = self.weights / np.sum(self.weights)

        # Record validation loss
        ens_pred = Y_hat @ self.weights
        self.validation_loss = float(np.mean((y_true - ens_pred) ** 2))
        return self

    def predict(self, Y_hat: np.ndarray) -> np.ndarray:
        """Combine predictions using optimal OR weights."""
        if self.weights is None:
            raise ValueError("Ensemble is not fitted.")
        return Y_hat @ self.weights

    def get_weights_summary(self) -> Dict[str, float]:
        """Return dictionary mapping model names to their optimal OR weights."""
        if self.weights is None:
            return {}
        return {name: float(w) for name, w in zip(self.model_names, self.weights)}
