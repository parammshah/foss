"""CPU-friendly Machine Learning regression model for liquidity flow forecasting."""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


class RidgeMLModel:
    """Fast, deterministic, CPU-friendly Ridge Regression with analytical solution.

    Computes optimal regression weights:
        beta = (X_std^T X_std + alpha * I)^{-1} X_std^T y
    Features are standardized using training statistics only (preventing leakage).
    Predictions for deposits and withdrawals are strictly non-negative.
    """

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = float(alpha)
        self.feature_names: List[str] = []
        self.weights: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> RidgeMLModel:
        """Fit model on feature matrix X and target series y."""
        self.feature_names = list(X.columns)
        X_mat = np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
        y_vec = np.asarray(y, dtype=float).ravel()

        n_samples, n_features = X_mat.shape
        if n_samples == 0:
            self.weights = np.zeros(n_features)
            self.intercept = 0.0
            self.feature_means = np.zeros(n_features)
            self.feature_stds = np.ones(n_features)
            return self


        # 1. Standardize features
        self.feature_means = np.mean(X_mat, axis=0)
        self.feature_stds = np.std(X_mat, axis=0)
        # Avoid division by zero
        self.feature_stds = np.where(self.feature_stds < 1e-8, 1.0, self.feature_stds)

        X_std = (X_mat - self.feature_means) / self.feature_stds

        # Center target
        y_mean = float(np.mean(y_vec))
        y_centered = y_vec - y_mean

        # 2. Analytical Ridge solve: (X^T X + alpha * I)^{-1} X^T y
        XtX = X_std.T @ X_std
        reg = self.alpha * np.eye(n_features)
        A = XtX + reg
        b = X_std.T @ y_centered

        try:
            self.weights = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self.weights = np.linalg.pinv(A) @ b

        self.intercept = y_mean
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict for input features. Output is clipped to non-negative values."""
        if self.weights is None or self.feature_means is None or self.feature_stds is None:
            raise ValueError("Model is not fitted yet.")

        # Reindex / ensure correct feature columns
        X_mat = np.nan_to_num(X[self.feature_names].values, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
        X_std = (X_mat - self.feature_means) / self.feature_stds

        raw_preds = X_std @ self.weights + self.intercept
        # Deposits and withdrawals cannot be negative
        return np.maximum(0.0, raw_preds)
