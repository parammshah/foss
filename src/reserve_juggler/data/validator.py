"""Dataset validation module for checking temporal integrity, numerical validity, and financial reconciliation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from reserve_juggler.contracts import (
    DataQualityStatus,
    EventPublisher,
    LearningEvent,
    NoOpEventPublisher,
)
from reserve_juggler.data.schema import (
    DEPOSITS_COL,
    MANDATORY_NUMERIC_COLUMNS,
    NET_FLOW_COL,
    PERCENTAGE_COLUMNS,
    REPO_SWEEP_MATURE_COL,
    REPO_SWEEP_OPEN_COL,
    TIMESTAMP_COL,
    WITHDRAWALS_COL,
)


@dataclass
class ValidationIssue:
    issue_type: str
    severity: str  # "INFO", "WARNING", or "ERROR"
    message: str
    row_indices: List[int] = field(default_factory=list)
    affected_columns: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    is_valid: bool
    status: DataQualityStatus
    issues: List[ValidationIssue] = field(default_factory=list)
    total_rows: int = 0
    total_columns: int = 0
    columns: List[str] = field(default_factory=list)
    start_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    sampling_frequency: str = "unknown"
    missing_hours_count: int = 0
    reconciliation_discrepancy_count: int = 0
    duplicate_count: int = 0
    imputed_count: int = 0
    constant_columns: List[str] = field(default_factory=list)
    column_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "ERROR" for i in self.issues)

    def summary(self) -> str:
        lines = [
            "=" * 78,
            "RESERVE JUGGLER — DATASET VALIDATION REPORT",
            "=" * 78,
            f"Validation Status:           {self.status.value}",
            f"Total Rows:                  {self.total_rows:,}",
            f"Total Columns:               {self.total_columns}",
            f"Observed Time Range:         {self.start_timestamp} to {self.end_timestamp}",
            f"Sampling Frequency:          {self.sampling_frequency}",
            f"Missing Hours Count:         {self.missing_hours_count}",
            f"Duplicate Timestamps:        {self.duplicate_count}",
            f"Reconciliation Discrepancies: {self.reconciliation_discrepancy_count}",
            f"Constant Parameter Columns:  {', '.join(self.constant_columns) if self.constant_columns else 'None'}",
            f"Total Issues Flagged:        {len(self.issues)}",
            "-" * 78,
            "COLUMN PROFILE SUMMARY:",
            f"{'Column Name':<36} {'Type':<8} {'Min':<10} {'Mean':<12} {'Max':<10} {'Nulls':<6}",
            "-" * 78,
        ]
        for col, stats in self.column_summaries.items():
            t = str(stats.get("dtype", "unknown"))[:7]
            mn = f"{stats.get('min', 'N/A'):.2f}" if isinstance(stats.get("min"), (int, float)) else str(stats.get("min", "N/A"))[:9]
            me = f"{stats.get('mean', 'N/A'):.2f}" if isinstance(stats.get("mean"), (int, float)) else str(stats.get("mean", "N/A"))[:11]
            mx = f"{stats.get('max', 'N/A'):.2f}" if isinstance(stats.get("max"), (int, float)) else str(stats.get("max", "N/A"))[:9]
            nl = str(stats.get("nulls", 0))
            lines.append(f"{col:<36} {t:<8} {mn:<10} {me:<12} {mx:<10} {nl:<6}")
        lines.append("=" * 78)
        return "\n".join(lines)


class DatasetValidator:
    """Validates raw and processed treasury datasets."""

    def __init__(
        self,
        event_publisher: Optional[EventPublisher] = None,
        net_flow_tolerance_usd: float = 1.0,
    ) -> None:
        self.publisher = event_publisher or NoOpEventPublisher()
        self.net_flow_tolerance = net_flow_tolerance_usd

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Run comprehensive validation checks across the dataframe."""
        issues: List[ValidationIssue] = []
        total_rows = len(df)
        total_cols = len(df.columns)
        col_summaries: Dict[str, Dict[str, Any]] = {}
        constant_cols: List[str] = []

        if total_rows == 0:
            issue = ValidationIssue(
                issue_type="EMPTY_DATASET",
                severity="ERROR",
                message="Dataset has 0 rows.",
            )
            return ValidationReport(
                is_valid=False,
                status=DataQualityStatus.CRITICAL_ERROR,
                issues=[issue],
                total_rows=0,
                total_columns=total_cols,
                columns=list(df.columns),
            )

        # 1. Timestamp existence and datetime type
        if TIMESTAMP_COL not in df.columns:
            issues.append(
                ValidationIssue(
                    issue_type="MISSING_TIMESTAMP_COLUMN",
                    severity="ERROR",
                    message=f"Mandatory column '{TIMESTAMP_COL}' missing.",
                )
            )
            return ValidationReport(
                is_valid=False,
                status=DataQualityStatus.CRITICAL_ERROR,
                issues=issues,
                total_rows=total_rows,
                total_columns=total_cols,
                columns=list(df.columns),
            )

        try:
            ts_series = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
        except Exception as e:
            issues.append(
                ValidationIssue(
                    issue_type="INVALID_TIMESTAMP_FORMAT",
                    severity="ERROR",
                    message=f"Could not parse timestamps to datetime: {e}",
                )
            )
            return ValidationReport(
                is_valid=False,
                status=DataQualityStatus.CRITICAL_ERROR,
                issues=issues,
                total_rows=total_rows,
                total_columns=total_cols,
                columns=list(df.columns),
            )

        # 2. Duplicate timestamps
        duplicates = df[ts_series.duplicated()].index.tolist()
        duplicate_count = len(duplicates)
        if duplicate_count > 0:
            issues.append(
                ValidationIssue(
                    issue_type="DUPLICATE_TIMESTAMPS",
                    severity="ERROR",
                    message=f"Found {duplicate_count} duplicate timestamp rows.",
                    row_indices=duplicates[:20],
                    affected_columns=[TIMESTAMP_COL],
                )
            )

        # 3. Chronological sorting
        is_sorted = ts_series.is_monotonic_increasing
        if not is_sorted:
            issues.append(
                ValidationIssue(
                    issue_type="OUT_OF_ORDER_TIMESTAMPS",
                    severity="ERROR",
                    message="Timestamps are not strictly monotonically increasing.",
                    affected_columns=[TIMESTAMP_COL],
                )
            )

        # 4. Hourly sequence gaps & frequency check
        missing_hours_count = 0
        observed_freq = "unknown"
        if is_sorted and duplicate_count == 0:
            diffs = ts_series.diff().dropna()
            unique_diffs = diffs.unique()
            one_hour = pd.Timedelta(hours=1)
            if len(unique_diffs) == 1 and unique_diffs[0] == one_hour:
                observed_freq = "1 Hour (Continuous)"
            else:
                observed_freq = f"Irregular: {unique_diffs}"

            gap_mask = ts_series.diff() > one_hour
            if gap_mask.any():
                gap_indices = df[gap_mask].index.tolist()
                for idx in gap_indices:
                    gap_hours = int(ts_series.diff().loc[idx].total_seconds() / 3600) - 1
                    missing_hours_count += gap_hours
                issues.append(
                    ValidationIssue(
                        issue_type="HOURLY_SEQUENCE_GAP",
                        severity="WARNING",
                        message=f"Detected {missing_hours_count} missing hours across {len(gap_indices)} gaps in timeline.",
                        row_indices=gap_indices[:20],
                        affected_columns=[TIMESTAMP_COL],
                    )
                )

        # 5. Missing / non-finite values in numeric columns & column statistics
        for col in df.columns:
            s = df[col]
            null_count = int(s.isnull().sum())
            unique_count = int(s.nunique())
            stats: Dict[str, Any] = {
                "dtype": str(s.dtype),
                "nulls": null_count,
                "unique": unique_count,
            }

            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                num_s = pd.to_numeric(s, errors="coerce")
                nan_mask = num_s.isna() | np.isinf(num_s)
                if nan_mask.any():
                    nan_indices = df[nan_mask].index.tolist()
                    issues.append(
                        ValidationIssue(
                            issue_type="NON_FINITE_VALUES",
                            severity="ERROR",
                            message=f"Column '{col}' contains {len(nan_indices)} NaN or Inf values.",
                            row_indices=nan_indices[:20],
                            affected_columns=[col],
                        )
                    )
                stats["min"] = float(num_s.min()) if len(num_s) > 0 else 0.0
                stats["max"] = float(num_s.max()) if len(num_s) > 0 else 0.0
                stats["mean"] = float(num_s.mean()) if len(num_s) > 0 else 0.0
                stats["std"] = float(num_s.std()) if len(num_s) > 0 else 0.0

                # Check if column is constant
                if unique_count == 1 and total_rows > 1:
                    constant_cols.append(col)
                    issues.append(
                        ValidationIssue(
                            issue_type="CONSTANT_COLUMN",
                            severity="INFO",
                            message=f"Column '{col}' is constant across all {total_rows} rows (value={stats['min']}). Represents simulation parameter.",
                            affected_columns=[col],
                        )
                    )
            elif pd.api.types.is_bool_dtype(s):
                stats["min"] = bool(s.min())
                stats["max"] = bool(s.max())
                stats["true_count"] = int(s.sum())
                stats["false_count"] = int((~s).sum())
            else:
                stats["min"] = str(s.iloc[0]) if len(s) > 0 else "N/A"
                stats["max"] = str(s.iloc[-1]) if len(s) > 0 else "N/A"

            col_summaries[col] = stats

        # 6. Non-negative deposit and withdrawal validity
        for cash_col in [DEPOSITS_COL, WITHDRAWALS_COL]:
            if cash_col in df.columns:
                s = pd.to_numeric(df[cash_col], errors="coerce")
                neg_mask = s < 0.0
                if neg_mask.any():
                    neg_indices = df[neg_mask].index.tolist()
                    issues.append(
                        ValidationIssue(
                            issue_type="NEGATIVE_CASH_FLOW",
                            severity="ERROR",
                            message=f"Column '{cash_col}' contains {len(neg_indices)} negative flow values.",
                            row_indices=neg_indices[:20],
                            affected_columns=[cash_col],
                        )
                    )

        # 7. Unit conversion check: rate percentages vs decimals
        is_pct_scale = False
        for probe_col in ["tier1_repo_rate_apy", "tier2_bond_yield_pct"]:
            if probe_col in df.columns:
                val = pd.to_numeric(df[probe_col], errors="coerce").mean()
                if val > 1.0:
                    is_pct_scale = True
                    break

        if is_pct_scale:
            for rate_col in PERCENTAGE_COLUMNS.intersection(df.columns):
                s = pd.to_numeric(df[rate_col], errors="coerce").dropna()
                issues.append(
                    ValidationIssue(
                        issue_type="UNNORMALIZED_PERCENTAGE",
                        severity="WARNING",
                        message=f"Column '{rate_col}' is expressed in percentage points (e.g. {s.mean():.3f}). Must be normalized to decimal fraction at boundary.",
                        affected_columns=[rate_col],
                    )
                )

        # 8. Cash flow reconciliation: net_flow = deposits - withdrawals
        reconciliation_discrepancy_count = 0
        if NET_FLOW_COL in df.columns and DEPOSITS_COL in df.columns and WITHDRAWALS_COL in df.columns:
            deps = pd.to_numeric(df[DEPOSITS_COL], errors="coerce")
            wids = pd.to_numeric(df[WITHDRAWALS_COL], errors="coerce")
            net = pd.to_numeric(df[NET_FLOW_COL], errors="coerce")
            calc_net = deps - wids
            discrepancy = (net - calc_net).abs() > self.net_flow_tolerance
            if discrepancy.any():
                disc_indices = df[discrepancy].index.tolist()
                reconciliation_discrepancy_count = len(disc_indices)
                issues.append(
                    ValidationIssue(
                        issue_type="NET_FLOW_RECONCILIATION_MISMATCH",
                        severity="WARNING",
                        message=f"Found {reconciliation_discrepancy_count} rows where {NET_FLOW_COL} != {DEPOSITS_COL} - {WITHDRAWALS_COL} (tolerance=${self.net_flow_tolerance:.2f}).",
                        row_indices=disc_indices[:20],
                        affected_columns=[NET_FLOW_COL, DEPOSITS_COL, WITHDRAWALS_COL],
                    )
                )

        # 9. Repo sweep open/close sanity check
        if REPO_SWEEP_OPEN_COL in df.columns and REPO_SWEEP_MATURE_COL in df.columns:
            open_s = df[REPO_SWEEP_OPEN_COL].astype(bool)
            mature_s = df[REPO_SWEEP_MATURE_COL].astype(bool)
            both_active = (open_s & mature_s).any()
            if both_active:
                both_indices = df[open_s & mature_s].index.tolist()
                issues.append(
                    ValidationIssue(
                        issue_type="INVALID_REPO_STATE",
                        severity="WARNING",
                        message="Repo sweep open and mature flags are simultaneously True.",
                        row_indices=both_indices[:10],
                        affected_columns=[REPO_SWEEP_OPEN_COL, REPO_SWEEP_MATURE_COL],
                    )
                )

        # Determine overall status
        has_errors = any(i.severity == "ERROR" for i in issues)
        has_warnings = any(i.severity == "WARNING" for i in issues)

        if has_errors:
            status = DataQualityStatus.CRITICAL_ERROR
        elif has_warnings or missing_hours_count > 0:
            status = DataQualityStatus.WARNING
        else:
            status = DataQualityStatus.OK

        report = ValidationReport(
            is_valid=not has_errors,
            status=status,
            issues=issues,
            total_rows=total_rows,
            total_columns=total_cols,
            columns=list(df.columns),
            start_timestamp=ts_series.iloc[0].to_pydatetime() if total_rows > 0 else None,
            end_timestamp=ts_series.iloc[-1].to_pydatetime() if total_rows > 0 else None,
            sampling_frequency=observed_freq,
            missing_hours_count=missing_hours_count,
            reconciliation_discrepancy_count=reconciliation_discrepancy_count,
            duplicate_count=duplicate_count,
            constant_columns=constant_cols,
            column_summaries=col_summaries,
        )

        now = datetime.now(timezone.utc)
        evt = LearningEvent(
            event_id=f"val_{int(now.timestamp())}_{total_rows}",
            event_type="dataset_validated",
            source_module="reserve_juggler.data.validator",
            event_timestamp=now,
            recording_timestamp=now,
            payload={
                "status": status.value,
                "is_valid": report.is_valid,
                "total_rows": total_rows,
                "total_columns": total_cols,
                "sampling_frequency": observed_freq,
                "constant_columns": constant_cols,
                "issues_count": len(issues),
            },
        )
        self.publisher.publish(evt)

        return report
