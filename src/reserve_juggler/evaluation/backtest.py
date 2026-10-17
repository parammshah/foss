"""Chronological rolling-origin evaluation and baseline comparison without lookahead leakage."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from reserve_juggler.contracts import (
    EventPublisher,
    LearningEvent,
    NoOpEventPublisher,
    ensure_utc,
)
from reserve_juggler.data.feature_engineering import FeatureEngineer
from reserve_juggler.data.schema import (
    DEPOSITS_COL,
    TIMESTAMP_COL,
    WITHDRAWALS_COL,
)
from reserve_juggler.evaluation.metrics import evaluate_forecast_suite
from reserve_juggler.forecasting.baseline import (
    RecentAverageBaseline,
    TimeOfWeekSeasonalBaseline,
)
from reserve_juggler.forecasting.ml_model import RidgeMLModel
from reserve_juggler.forecasting.or_ensemble import ORConstrainedEnsemble
from reserve_juggler.forecasting.uncertainty import ORScenarioReconciler


@dataclass
class BacktestHorizonComparison:
    horizon_hours: int
    target_variable: str
    recent_baseline_mae: float
    recent_baseline_wape: float
    seasonal_baseline_mae: float
    seasonal_baseline_wape: float
    ridge_ml_mae: float
    ridge_ml_wape: float
    or_ensemble_mae: float
    or_ensemble_wape: float
    or_ensemble_weights: Dict[str, float]
    sample_count: int
    picp_80: Optional[float] = None
    pinball_p10: Optional[float] = None
    pinball_p90: Optional[float] = None
    calibration_error_80: Optional[float] = None


@dataclass
class BacktestReport:
    start_timestamp: datetime
    end_timestamp: datetime
    total_evaluations: int
    horizon_results: List[BacktestHorizonComparison] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 98,
            f"CHRONOLOGICAL ROLLING-ORIGIN BACKTEST RESULTS ({self.start_timestamp} -> {self.end_timestamp})",
            "=" * 98,
            f"{'Target':<14} {'Horizon':<8} {'Recent MAE':<14} {'Seasonal MAE':<14} {'ML MAE':<14} {'OR Ensemble MAE':<16} {'WAPE':<8} {'PICP 80%':<10}",
            "-" * 98,
        ]
        for h in self.horizon_results:
            picp_str = f"{h.picp_80:.1%}" if h.picp_80 is not None else "N/A"
            lines.append(
                f"{h.target_variable:<14} {h.horizon_hours:<8} "
                f"${h.recent_baseline_mae:<13,.0f} "
                f"${h.seasonal_baseline_mae:<13,.0f} "
                f"${h.ridge_ml_mae:<13,.0f} "
                f"${h.or_ensemble_mae:<15,.0f} "
                f"{h.or_ensemble_wape:<7.2%} "
                f"{picp_str:<10}"
            )
        lines.append("=" * 98)
        return "\n".join(lines)


class RollingOriginBacktest:
    """Executes chronological rolling-origin backtesting across all candidate models."""

    def __init__(
        self,
        event_publisher: Optional[EventPublisher] = None,
        feature_engineer: Optional[FeatureEngineer] = None,
    ) -> None:
        self.publisher = event_publisher or NoOpEventPublisher()
        self.feature_engineer = feature_engineer or FeatureEngineer()

    def run(
        self,
        df: pd.DataFrame,
        target_cols: Optional[List[str]] = None,
        horizons: Optional[List[int]] = None,
        initial_train_size: int = 500,
        roll_step_hours: int = 24,
        max_evaluations: int = 20,
    ) -> BacktestReport:
        """Run chronological backtest without lookahead leakage."""
        targets = target_cols or [DEPOSITS_COL, WITHDRAWALS_COL]
        test_horizons = horizons or [1, 6, 24, 48, 72]

        clean_df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(clean_df[TIMESTAMP_COL]):
            clean_df[TIMESTAMP_COL] = pd.to_datetime(clean_df[TIMESTAMP_COL], utc=True)
        clean_df = clean_df.sort_values(by=TIMESTAMP_COL).reset_index(drop=True)

        n_total = len(clean_df)
        max_h = max(test_horizons)

        # Check origin points
        origin_indices: List[int] = []
        curr = initial_train_size
        while curr + max_h < n_total and len(origin_indices) < max_evaluations:
            origin_indices.append(curr)
            curr += roll_step_hours

        if not origin_indices:
            raise ValueError(f"Dataset has {n_total} rows, insufficient for initial train size {initial_train_size} + max horizon {max_h}.")

        featured_all = self.feature_engineer.create_features(clean_df)
        feature_cols = self.feature_engineer.get_feature_column_names(featured_all)

        horizon_comparisons: List[BacktestHorizonComparison] = []

        for target in targets:
            for h in test_horizons:
                y_trues: List[float] = []
                p_recents: List[float] = []
                p_seasonals: List[float] = []
                p_mls: List[float] = []
                p_ensembles: List[float] = []
                p_p10s: List[float] = []
                p_p90s: List[float] = []
                latest_weights: Dict[str, float] = {}

                for origin_idx in origin_indices:
                    origin_ts = clean_df.loc[origin_idx, TIMESTAMP_COL]
                    target_idx = origin_idx + h
                    actual_val = float(clean_df.loc[target_idx, target])
                    target_ts = clean_df.loc[target_idx, TIMESTAMP_COL]

                    # Slice strictly up to origin (no future data leakage!)
                    train_data = clean_df.iloc[:origin_idx].reset_index(drop=True)
                    train_features = featured_all.iloc[:origin_idx].reset_index(drop=True)

                    # 1. Recent Baseline
                    recent_model = RecentAverageBaseline(method="rolling_24h").fit(train_data[target])
                    pred_recent = recent_model.predict(h)

                    # 2. Seasonal Baseline
                    seasonal_model = TimeOfWeekSeasonalBaseline().fit(train_data, target)
                    pred_seasonal = float(seasonal_model.predict(pd.Timestamp(target_ts))[0])

                    # 3. Ridge ML Model
                    y_shifted = train_data[target].shift(-h).dropna()
                    X_aligned = train_features.loc[y_shifted.index, feature_cols]
                    ml_model = RidgeMLModel(alpha=10.0).fit(X_aligned, y_shifted)
                    origin_feat = featured_all.iloc[[origin_idx]][feature_cols]
                    pred_ml = float(ml_model.predict(origin_feat)[0])

                    # 4. OR Constrained Ensemble
                    val_window = min(96, max(12, len(y_shifted) // 4))
                    val_idx = y_shifted.index[-val_window:]

                    val_y = y_shifted.loc[val_idx].values
                    val_X = train_features.loc[val_idx, feature_cols]
                    val_ts = train_data.loc[val_idx, TIMESTAMP_COL] + pd.Timedelta(hours=h)

                    val_pred_rec = train_data[target].shift(1).rolling(24, min_periods=1).mean().bfill().loc[val_idx].values
                    val_pred_sea = seasonal_model.predict(pd.DatetimeIndex(val_ts))
                    val_pred_ml = ml_model.predict(val_X)

                    Y_hat_val = np.column_stack([val_pred_rec, val_pred_sea, val_pred_ml])
                    ensemble = ORConstrainedEnsemble().fit(
                        Y_hat=Y_hat_val,
                        y_true=val_y,
                        model_names=["recent_avg", "seasonal", "ridge_ml"],
                    )
                    pred_ens = float(ensemble.predict(np.array([[pred_recent, pred_seasonal, pred_ml]]))[0])
                    latest_weights = ensemble.get_weights_summary()

                    # Uncertainty quantiles
                    val_ens_preds = ensemble.predict(Y_hat_val)
                    residuals = val_y - val_ens_preds
                    p10_res = float(np.percentile(residuals, 10))
                    p90_res = float(np.percentile(residuals, 90))
                    p10_val = max(0.0, pred_ens + p10_res)
                    p90_val = max(p10_val, pred_ens + p90_res)

                    y_trues.append(actual_val)
                    p_recents.append(pred_recent)
                    p_seasonals.append(pred_seasonal)
                    p_mls.append(pred_ml)
                    p_ensembles.append(pred_ens)
                    p_p10s.append(p10_val)
                    p_p90s.append(p90_val)

                    # Publish evaluation event
                    evt = LearningEvent(
                        event_id=f"eval_{uuid.uuid4().hex[:8]}",
                        event_type="forecast_evaluated",
                        source_module="reserve_juggler.evaluation.backtest",
                        event_timestamp=target_ts,
                        recording_timestamp=datetime.now(timezone.utc),
                        payload={
                            "origin_timestamp": origin_ts.isoformat(),
                            "target_timestamp": target_ts.isoformat(),
                            "horizon_hours": h,
                            "target": target,
                            "actual_usd": actual_val,
                            "recent_pred_usd": pred_recent,
                            "seasonal_pred_usd": pred_seasonal,
                            "ml_pred_usd": pred_ml,
                            "ensemble_pred_usd": pred_ens,
                            "p10_usd": p10_val,
                            "p90_usd": p90_val,
                        },
                    )
                    self.publisher.publish(evt)

                # Compute metrics across roll origins
                m_rec = evaluate_forecast_suite(y_trues, p_recents)
                m_sea = evaluate_forecast_suite(y_trues, p_seasonals)
                m_ml = evaluate_forecast_suite(y_trues, p_mls)
                m_ens = evaluate_forecast_suite(y_trues, p_ensembles, p_p10s, p_p90s)

                comp = BacktestHorizonComparison(
                    horizon_hours=h,
                    target_variable=target,
                    recent_baseline_mae=m_rec["mae"],
                    recent_baseline_wape=m_rec["wape"],
                    seasonal_baseline_mae=m_sea["mae"],
                    seasonal_baseline_wape=m_sea["wape"],
                    ridge_ml_mae=m_ml["mae"],
                    ridge_ml_wape=m_ml["wape"],
                    or_ensemble_mae=m_ens["mae"],
                    or_ensemble_wape=m_ens["wape"],
                    or_ensemble_weights=latest_weights,
                    sample_count=len(y_trues),
                    picp_80=m_ens.get("picp_80"),
                    pinball_p10=m_ens.get("pinball_p10"),
                    pinball_p90=m_ens.get("pinball_p90"),
                    calibration_error_80=m_ens.get("calibration_error_80"),
                )
                horizon_comparisons.append(comp)

        return BacktestReport(
            start_timestamp=clean_df.loc[origin_indices[0], TIMESTAMP_COL],
            end_timestamp=clean_df.loc[origin_indices[-1] + max_h, TIMESTAMP_COL],
            total_evaluations=len(origin_indices),
            horizon_results=horizon_comparisons,
        )
