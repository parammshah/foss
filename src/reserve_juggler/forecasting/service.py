"""Forecast service coordinating hybrid ML + OR forecasting models and delivering ForecastBundle."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from reserve_juggler.contracts import (
    DataQualityStatus,
    EventPublisher,
    ForecastBundle,
    ForecastPoint,
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
from reserve_juggler.forecasting.baseline import (
    RecentAverageBaseline,
    TimeOfWeekSeasonalBaseline,
)
from reserve_juggler.forecasting.ml_model import RidgeMLModel
from reserve_juggler.forecasting.or_ensemble import ORConstrainedEnsemble
from reserve_juggler.forecasting.uncertainty import ORScenarioReconciler

DEFAULT_HORIZONS = [1, 6, 24, 48, 72]


class HorizonModelSuite:
    """Manages baseline, ML, and OR ensemble models for a single target horizon."""

    def __init__(self, horizon_hours: int) -> None:
        self.horizon_hours = horizon_hours

        # Models for deposits
        self.dep_recent = RecentAverageBaseline(method="rolling_24h")
        self.dep_seasonal = TimeOfWeekSeasonalBaseline()
        self.dep_ml = RidgeMLModel(alpha=10.0)
        self.dep_or_ensemble = ORConstrainedEnsemble()

        # Models for withdrawals
        self.wid_recent = RecentAverageBaseline(method="rolling_24h")
        self.wid_seasonal = TimeOfWeekSeasonalBaseline()
        self.wid_ml = RidgeMLModel(alpha=10.0)
        self.wid_or_ensemble = ORConstrainedEnsemble()

        # Reconciler
        self.reconciler = ORScenarioReconciler()

        self.is_fitted: bool = False


class ForecastService:
    """Canonical service for generating probabilistic multi-horizon liquidity forecasts.

    Coordinates feature engineering, statistical baselines, CPU-friendly ML,
    and Operations Research constrained combination and scenario reconciliation.
    """

    def __init__(
        self,
        event_publisher: Optional[EventPublisher] = None,
        feature_engineer: Optional[FeatureEngineer] = None,
    ) -> None:
        self.publisher = event_publisher or NoOpEventPublisher()
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.suites: Dict[int, HorizonModelSuite] = {}
        self.model_version: str = "RJ-Hybrid-ML-OR-v1.0"
        self._historical_df: Optional[pd.DataFrame] = None

    def fit(
        self,
        historical_df: pd.DataFrame,
        horizons: Optional[List[int]] = None,
        val_split_ratio: float = 0.2,
    ) -> ForecastService:
        """Fit all models on historical dataset using train/validation split for OR combination."""
        df = historical_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[TIMESTAMP_COL]):
            df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
        df = df.sort_values(by=TIMESTAMP_COL).reset_index(drop=True)
        self._historical_df = df

        target_horizons = horizons or DEFAULT_HORIZONS
        n_total = len(df)
        n_val = max(4, int(n_total * val_split_ratio))
        if n_val >= n_total - 2:
            n_val = max(2, n_total // 4)
        train_df = df.iloc[:-n_val].reset_index(drop=True)
        val_df = df.iloc[-n_val:].reset_index(drop=True)


        # Generate features with zero lookahead leakage
        featured_all = self.feature_engineer.create_features(df)
        featured_train = featured_all.iloc[:-n_val].reset_index(drop=True)
        featured_val = featured_all.iloc[-n_val:].reset_index(drop=True)
        feature_cols = self.feature_engineer.get_feature_column_names(featured_all)

        for h in target_horizons:
            suite = HorizonModelSuite(h)

            # Fit baselines on train
            suite.dep_recent.fit(train_df[DEPOSITS_COL])
            suite.dep_seasonal.fit(train_df, DEPOSITS_COL)
            suite.wid_recent.fit(train_df[WITHDRAWALS_COL])
            suite.wid_seasonal.fit(train_df, WITHDRAWALS_COL)

            # Prepare ML targets shifted by horizon h
            # At row t, target is value at t+h
            y_dep_train = train_df[DEPOSITS_COL].shift(-h).dropna()
            X_dep_train = featured_train.loc[y_dep_train.index, feature_cols]
            suite.dep_ml.fit(X_dep_train, y_dep_train)

            y_wid_train = train_df[WITHDRAWALS_COL].shift(-h).dropna()
            X_wid_train = featured_train.loc[y_wid_train.index, feature_cols]
            suite.wid_ml.fit(X_wid_train, y_wid_train)

            # Generate validation predictions for OR ensemble fitting
            y_dep_val = val_df[DEPOSITS_COL].shift(-h).dropna()
            val_idx = y_dep_val.index
            X_val = featured_val.loc[val_idx, feature_cols]
            target_ts_val = val_df.loc[val_idx, TIMESTAMP_COL] + pd.Timedelta(hours=h)

            pred_dep_recent = val_df[DEPOSITS_COL].shift(1).rolling(24, min_periods=1).mean().bfill().loc[val_idx].values
            pred_dep_seasonal = suite.dep_seasonal.predict(pd.DatetimeIndex(target_ts_val))
            pred_dep_ml = suite.dep_ml.predict(X_val)

            Y_hat_dep = np.column_stack([pred_dep_recent, pred_dep_seasonal, pred_dep_ml])
            suite.dep_or_ensemble.fit(
                Y_hat=Y_hat_dep,
                y_true=y_dep_val.values,
                model_names=["recent_avg", "seasonal", "ridge_ml"],
            )

            y_wid_val = val_df[WITHDRAWALS_COL].shift(-h).dropna()
            pred_wid_recent = val_df[WITHDRAWALS_COL].shift(1).rolling(24, min_periods=1).mean().bfill().loc[val_idx].values
            pred_wid_seasonal = suite.wid_seasonal.predict(pd.DatetimeIndex(target_ts_val))

            pred_wid_ml = suite.wid_ml.predict(X_val)

            Y_hat_wid = np.column_stack([pred_wid_recent, pred_wid_seasonal, pred_wid_ml])
            suite.wid_or_ensemble.fit(
                Y_hat=Y_hat_wid,
                y_true=y_wid_val.values,
                model_names=["recent_avg", "seasonal", "ridge_ml"],
            )

            # Fit OR uncertainty quantiles on validation ensemble residuals
            ens_pred_dep = suite.dep_or_ensemble.predict(Y_hat_dep)
            ens_pred_wid = suite.wid_or_ensemble.predict(Y_hat_wid)
            suite.reconciler.fit_residuals(
                true_deposits=y_dep_val.values,
                pred_deposits=ens_pred_dep,
                true_withdrawals=y_wid_val.values,
                pred_withdrawals=ens_pred_wid,
            )

            suite.is_fitted = True
            self.suites[h] = suite

        return self

    def forecast(
        self,
        as_of_timestamp: datetime,
        horizons_hours: Optional[List[int]] = None,
    ) -> ForecastBundle:
        """Primary forecast method returning canonical ForecastBundle.

        Args:
            as_of_timestamp: UTC datetime indicating current forecast origin.
            horizons_hours: List of horizons in hours (defaults to [1, 6, 24, 48, 72]).
        """
        as_of_utc = ensure_utc(as_of_timestamp)
        horizons = horizons_hours or list(self.suites.keys()) or DEFAULT_HORIZONS

        if self._historical_df is None or not self.suites:
            raise ValueError("ForecastService must be fit() with historical data before forecasting.")

        # Cut off historical data strictly at as_of_timestamp (Zero future leakage!)
        ts_col = self._historical_df[TIMESTAMP_COL]
        hist_subset = self._historical_df[ts_col <= as_of_utc].copy()
        if len(hist_subset) == 0:
            hist_subset = self._historical_df.iloc[:24].copy()

        # Compute features strictly up to as-of time
        featured_hist = self.feature_engineer.create_features(hist_subset)
        feature_cols = self.feature_engineer.get_feature_column_names(featured_hist)
        latest_features = featured_hist.iloc[[-1]][feature_cols]

        points: List[ForecastPoint] = []
        model_versions: Dict[str, str] = {}
        eval_meta: Dict[str, Any] = {}

        for h in horizons:
            suite = self.suites.get(h)
            if suite is None or not suite.is_fitted:
                # If horizon was not pre-fitted, reuse closest available suite
                available_h = min(self.suites.keys(), key=lambda k: abs(k - h))
                suite = self.suites[available_h]

            target_ts = as_of_utc + timedelta(hours=h)

            # 1. Candidate predictions
            # Deposits
            p_dep_recent = suite.dep_recent.predict(h)
            p_dep_seasonal = float(suite.dep_seasonal.predict(pd.Timestamp(target_ts))[0])
            p_dep_ml = float(suite.dep_ml.predict(latest_features)[0])
            Y_dep = np.array([[p_dep_recent, p_dep_seasonal, p_dep_ml]])
            point_dep = float(suite.dep_or_ensemble.predict(Y_dep)[0])

            # Withdrawals
            p_wid_recent = suite.wid_recent.predict(h)
            p_wid_seasonal = float(suite.wid_seasonal.predict(pd.Timestamp(target_ts))[0])
            p_wid_ml = float(suite.wid_ml.predict(latest_features)[0])
            Y_wid = np.array([[p_wid_recent, p_wid_seasonal, p_wid_ml]])
            point_wid = float(suite.wid_or_ensemble.predict(Y_wid)[0])

            # 2. OR scenario reconciliation & quantiles
            reconciled = suite.reconciler.reconcile_point(
                pred_deposit=point_dep,
                pred_withdrawal=point_wid,
                horizon_hours=h,
            )

            # Record model weights for metadata
            dep_w = suite.dep_or_ensemble.get_weights_summary()
            wid_w = suite.wid_or_ensemble.get_weights_summary()
            eval_meta[f"h{h}_dep_weights"] = dep_w
            eval_meta[f"h{h}_wid_weights"] = wid_w

            pt = ForecastPoint(
                as_of_timestamp=as_of_utc,
                target_timestamp=target_ts,
                horizon_hours=h,
                deposit_p10=reconciled["deposit_p10"],
                deposit_p50=reconciled["deposit_p50"],
                deposit_p90=reconciled["deposit_p90"],
                withdrawal_p10=reconciled["withdrawal_p10"],
                withdrawal_p50=reconciled["withdrawal_p50"],
                withdrawal_p90=reconciled["withdrawal_p90"],
                net_outflow_p50=reconciled["net_outflow_p50"],
                net_outflow_p95=reconciled["net_outflow_p95"],
                net_outflow_p99=reconciled["net_outflow_p99"],
                cumulative_net_outflow_p95=reconciled["cumulative_net_outflow_p95"],
                cumulative_net_outflow_p99=reconciled["cumulative_net_outflow_p99"],
                reliability_score=0.95,
                model_version=f"{self.model_version}-h{h}",
            )
            points.append(pt)
            model_versions[f"horizon_{h}"] = pt.model_version

        bundle = ForecastBundle(
            forecast_timestamp=as_of_utc,
            points=points,
            scenario_references={
                "base_scenario": "expected_reconciled",
                "stress_scenario_p95": "adverse_outflow_shock",
                "stress_scenario_p99": "extreme_outflow_tail",
            },
            data_quality_status=DataQualityStatus.OK,
            model_versions=model_versions,
            evaluation_metadata=eval_meta,
        )

        # Emit forecast learning event
        now = datetime.now(timezone.utc)
        evt = LearningEvent(
            event_id=f"fc_{uuid.uuid4().hex[:8]}",
            event_type="forecast_generated",
            source_module="reserve_juggler.forecasting.service",
            event_timestamp=as_of_utc,
            recording_timestamp=now,
            payload={
                "horizons": horizons,
                "points_count": len(points),
                "as_of": as_of_utc.isoformat(),
            },
            model_version=self.model_version,
        )
        self.publisher.publish(evt)

        return bundle
