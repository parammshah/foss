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
    NET_FLOW_COL,
    REPO_SWEEP_MATURE_COL,
    REPO_SWEEP_OPEN_COL,
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
    unique_target_timestamps_count: int = 0
    # Multi-level interval calibration
    picp_50: Optional[float] = None
    mpiw_50: Optional[float] = None
    picp_80: Optional[float] = None
    mpiw_80: Optional[float] = None
    picp_95: Optional[float] = None
    mpiw_95: Optional[float] = None
    pinball_p10: Optional[float] = None
    pinball_p90: Optional[float] = None
    winkler_80: Optional[float] = None
    calibration_error_80: Optional[float] = None
    calibration_sample_size: int = 0
    # Financial regime evaluations
    regime_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestReport:
    start_timestamp: datetime
    end_timestamp: datetime
    total_evaluations: int
    holdout_hours: int = 0
    holdout_start_timestamp: Optional[datetime] = None
    horizon_results: List[BacktestHorizonComparison] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 108,
            f"CHRONOLOGICAL ROLLING-ORIGIN BACKTEST RESULTS ({self.start_timestamp} -> {self.end_timestamp})",
            f"Total Origins: {self.total_evaluations} | Final Holdout Reserved: {self.holdout_hours} Hours",
            "=" * 108,
            f"{'Target':<14} {'Horizon':<8} {'Recent MAE':<14} {'Seasonal MAE':<14} {'ML MAE':<14} {'OR Ensemble MAE':<16} {'WAPE':<8} {'PICP 80%':<10} {'PICP 95%':<10}",
            "-" * 115,
        ]
        for h in self.horizon_results:
            p80_str = f"{h.picp_80:.1%}" if h.picp_80 is not None else "N/A"
            p95_str = f"{h.picp_95:.1%}" if h.picp_95 is not None else "N/A"
            lines.append(
                f"{h.target_variable:<14} {h.horizon_hours:<8} "
                f"${h.recent_baseline_mae:<13,.0f} "
                f"${h.seasonal_baseline_mae:<13,.0f} "
                f"${h.ridge_ml_mae:<13,.0f} "
                f"${h.or_ensemble_mae:<15,.0f} "
                f"{h.or_ensemble_wape:<7.2%} "
                f"{p80_str:<10} "
                f"{p95_str:<10}"
            )
        lines.append("=" * 115)
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
        initial_train_size: int = 1000,
        roll_step_hours: int = 24,
        max_evaluations: int = 60,
        holdout_hours: int = 720,
    ) -> BacktestReport:
        """Run chronological backtest with multi-origin expansion, regime checks, and holdout reserve."""
        targets = target_cols or [DEPOSITS_COL, WITHDRAWALS_COL]
        test_horizons = horizons or [1, 6, 24, 48, 72]

        clean_df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(clean_df[TIMESTAMP_COL]):
            clean_df[TIMESTAMP_COL] = pd.to_datetime(clean_df[TIMESTAMP_COL], utc=True)
        clean_df = clean_df.sort_values(by=TIMESTAMP_COL).reset_index(drop=True)

        n_total = len(clean_df)
        max_h = max(test_horizons)

        # Reserve final holdout period (untouched by model training or tuning)
        effective_limit = n_total - max(0, holdout_hours)
        if effective_limit <= initial_train_size + max_h:
            effective_limit = n_total

        # Collect chronological origin points
        origin_indices: List[int] = []
        curr = initial_train_size
        while curr + max_h < effective_limit and len(origin_indices) < max_evaluations:
            origin_indices.append(curr)
            curr += roll_step_hours

        if not origin_indices:
            raise ValueError(
                f"Dataset has {n_total} rows, insufficient for initial train size {initial_train_size} + "
                f"holdout {holdout_hours} + max horizon {max_h}."
            )

        # Precompute features (strict row t features derived from <= t-1)
        featured_all = self.feature_engineer.create_features(clean_df)
        feature_cols = self.feature_engineer.get_feature_column_names(featured_all)

        # Establish fixed regime thresholds strictly from the initial training window
        init_train = clean_df.iloc[:initial_train_size]
        high_wid_thresh = float(init_train[WITHDRAWALS_COL].quantile(0.90)) if WITHDRAWALS_COL in init_train.columns else 1e9
        large_def_thresh = float(init_train[NET_FLOW_COL].quantile(0.10)) if NET_FLOW_COL in init_train.columns else -1e9

        horizon_comparisons: List[BacktestHorizonComparison] = []

        for target in targets:
            for h in test_horizons:
                y_trues: List[float] = []
                p_recents: List[float] = []
                p_seasonals: List[float] = []
                p_mls: List[float] = []
                p_ensembles: List[float] = []

                # Interval quantiles
                p_p025s: List[float] = []
                p_p10s: List[float] = []
                p_p25s: List[float] = []
                p_p75s: List[float] = []
                p_p90s: List[float] = []
                p_p975s: List[float] = []

                target_timestamps: List[datetime] = []
                regime_flags: List[Dict[str, bool]] = []
                latest_weights: Dict[str, float] = {}
                last_cal_size: int = 0

                for origin_idx in origin_indices:
                    origin_ts = clean_df.loc[origin_idx, TIMESTAMP_COL]
                    target_idx = origin_idx + h
                    actual_val = float(clean_df.loc[target_idx, target])
                    target_ts = clean_df.loc[target_idx, TIMESTAMP_COL]
                    target_timestamps.append(target_ts)

                    # Determine regime conditions at target observation
                    act_wid = float(clean_df.loc[target_idx, WITHDRAWALS_COL]) if WITHDRAWALS_COL in clean_df.columns else 0.0
                    act_net = float(clean_df.loc[target_idx, NET_FLOW_COL]) if NET_FLOW_COL in clean_df.columns else 0.0
                    is_sweep = bool(clean_df.loc[target_idx, REPO_SWEEP_OPEN_COL]) if REPO_SWEEP_OPEN_COL in clean_df.columns else False
                    is_mature = bool(clean_df.loc[target_idx, REPO_SWEEP_MATURE_COL]) if REPO_SWEEP_MATURE_COL in clean_df.columns else False

                    regime_flags.append({
                        "high_withdrawal": act_wid >= high_wid_thresh,
                        "large_negative_net_flow": act_net <= large_def_thresh,
                        "repo_sweep_window": is_sweep,
                        "repo_maturity_settlement": is_mature,
                    })

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
                    val_window = min(96, max(24, len(y_shifted) // 4))
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

                    # 5. Chronological Multi-Level Uncertainty Calibration
                    val_ens_preds = ensemble.predict(Y_hat_val)
                    residuals = val_y - val_ens_preds
                    last_cal_size = len(residuals)

                    res_p025 = float(np.percentile(residuals, 2.5))
                    res_p10 = float(np.percentile(residuals, 10.0))
                    res_p25 = float(np.percentile(residuals, 25.0))
                    res_p75 = float(np.percentile(residuals, 75.0))
                    res_p90 = float(np.percentile(residuals, 90.0))
                    res_p975 = float(np.percentile(residuals, 97.5))

                    lb025 = max(0.0, pred_ens + res_p025)
                    lb10 = max(lb025, pred_ens + res_p10)
                    lb25 = max(lb10, pred_ens + res_p25)
                    ub75 = max(lb25, pred_ens + res_p75)
                    ub90 = max(ub75, pred_ens + res_p90)
                    ub975 = max(ub90, pred_ens + res_p975)

                    y_trues.append(actual_val)
                    p_recents.append(pred_recent)
                    p_seasonals.append(pred_seasonal)
                    p_mls.append(pred_ml)
                    p_ensembles.append(pred_ens)

                    p_p025s.append(lb025)
                    p_p10s.append(lb10)
                    p_p25s.append(lb25)
                    p_p75s.append(ub75)
                    p_p90s.append(ub90)
                    p_p975s.append(ub975)

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
                            "p10_usd": lb10,
                            "p90_usd": ub90,
                        },
                    )
                    self.publisher.publish(evt)

                # Compute overall metrics across roll origins
                m_rec = evaluate_forecast_suite(y_trues, p_recents)
                m_sea = evaluate_forecast_suite(y_trues, p_seasonals)
                m_ml = evaluate_forecast_suite(y_trues, p_mls)
                m_ens = evaluate_forecast_suite(
                    y_true=y_trues,
                    y_pred_point=p_ensembles,
                    y_p10=p_p10s,
                    y_p90=p_p90s,
                    y_p25=p_p25s,
                    y_p75=p_p75s,
                    y_p025=p_p025s,
                    y_p975=p_p975s,
                )

                # Compute regime-specific metrics
                regime_results: Dict[str, Any] = {}
                for r_name in ["high_withdrawal", "large_negative_net_flow", "repo_sweep_window", "repo_maturity_settlement"]:
                    r_indices = [idx for idx, flags in enumerate(regime_flags) if flags.get(r_name, False)]
                    n_r = len(r_indices)
                    if n_r >= 10:
                        y_r = [y_trues[i] for i in r_indices]
                        p_r = [p_ensembles[i] for i in r_indices]
                        lb_r = [p_p10s[i] for i in r_indices]
                        ub_r = [p_p90s[i] for i in r_indices]
                        m_r = evaluate_forecast_suite(y_r, p_r, y_p10=lb_r, y_p90=ub_r)
                        regime_results[r_name] = {
                            "status": "VALID",
                            "sample_count": n_r,
                            "mae": m_r["mae"],
                            "wape": m_r["wape"],
                            "picp_80": m_r.get("picp_80"),
                        }
                    else:
                        regime_results[r_name] = {
                            "status": "Insufficient evidence (N < 10)",
                            "sample_count": n_r,
                        }

                unique_ts_count = len(set(target_timestamps))

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
                    unique_target_timestamps_count=unique_ts_count,
                    picp_50=m_ens.get("picp_50"),
                    mpiw_50=m_ens.get("mpiw_50"),
                    picp_80=m_ens.get("picp_80"),
                    mpiw_80=m_ens.get("mpiw_80"),
                    picp_95=m_ens.get("picp_95"),
                    mpiw_95=m_ens.get("mpiw_95"),
                    pinball_p10=m_ens.get("pinball_p10"),
                    pinball_p90=m_ens.get("pinball_p90"),
                    winkler_80=m_ens.get("winkler_80"),
                    calibration_error_80=m_ens.get("calibration_error_80"),
                    calibration_sample_size=last_cal_size,
                    regime_metrics=regime_results,
                )
                horizon_comparisons.append(comp)

        holdout_ts = clean_df.loc[n_total - holdout_hours, TIMESTAMP_COL] if holdout_hours > 0 and holdout_hours < n_total else None

        return BacktestReport(
            start_timestamp=clean_df.loc[origin_indices[0], TIMESTAMP_COL],
            end_timestamp=clean_df.loc[origin_indices[-1] + max_h, TIMESTAMP_COL],
            total_evaluations=len(origin_indices),
            holdout_hours=holdout_hours,
            holdout_start_timestamp=holdout_ts,
            horizon_results=horizon_comparisons,
        )
