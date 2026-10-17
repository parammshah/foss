"""Dataset validation module for checking temporal integrity, numerical validity, and financial reconciliation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
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
    RATE_COLUMNS,
    REPO_SWEEP_MATURE_COL,
    REPO_SWEEP_OPEN_COL,
    TIMESTAMP_COL,
    WITHDRAWALS_COL,
)


@dataclass
class ValidationIssue:
    issue_type: str
    severity: str  # "WARNING" or "ERROR"
    message: str
    row_indices: List[int] = field(default_factory=list)
    affected_columns: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    is_valid: bool
    status: DataQualityStatus
    issues: List[ValidationIssue] = field(default_factory=list)
    total_rows: int = 0
    start_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    missing_hours_count: int = 0
    reconciliation_discrepancy_count: int = 0
    duplicate_count: int = 0
    imputed_count: int = 0

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "ERROR" for i in self.issues)

    def summary(self) -> str:
        lines = [
            f"Validation Status: {self.status.value}",
            f"Total Rows: {self.total_rows}",
            f"Time Range: {self.start_timestamp} to {self.end_timestamp}",
            f"Missing Hours: {self.missing_hours_count}",
            f"Duplicate Timestamps: {self.duplicate_count}",
            f"Reconciliation Discrepancies: {self.reconciliation_discrepancy_count}",
            f"Total Issues Flagged: {len(self.issues)}",
        ]
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
        """Run full validation checks across the dataframe."""
        issues: List[ValidationIssue] = []
        total_rows = len(df)

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
            )

        # Parse timestamps to UTC
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

        # 4. Hourly sequence gaps
        missing_hours_count = 0
        if is_sorted and duplicate_count == 0:
            diffs = ts_series.diff()
            one_hour = pd.Timedelta(hours=1)
            # Find where diff > 1 hour
            gap_mask = diffs > one_hour
            if gap_mask.any():
                gap_indices = df[gap_mask].index.tolist()
                for idx in gap_indices:
                    gap_hours = int(diffs.loc[idx].total_seconds() / 3600) - 1
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

        # 5. Missing / non-finite values in mandatory numeric columns
        for col in MANDATORY_NUMERIC_COLUMNS:
            if col not in df.columns:
                issues.append(
                    ValidationIssue(
                        issue_type="MISSING_COLUMN",
                        severity="ERROR",
                        message=f"Mandatory numeric column '{col}' is missing.",
                        affected_columns=[col],
                    )
                )
            else:
                series = pd.to_numeric(df[col], errors="coerce")
                nan_mask = series.isna() | np.isinf(series)
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
        for rate_col in RATE_COLUMNS.intersection(df.columns):
            s = pd.to_numeric(df[rate_col], errors="coerce").dropna()
            # If values > 1.0 (e.g. 4.8 instead of 0.048), flag warning for unit conversion
            if (s > 1.0).any():
                pct_indices = df[df[rate_col] > 1.0].index.tolist()
                issues.append(
                    ValidationIssue(
                        issue_type="UNNORMALIZED_PERCENTAGE",
                        severity="WARNING",
                        message=f"Column '{rate_col}' appears to have percentage values > 1.0 (e.g. {s.max():.2f}). Must be normalized to decimal fraction.",
                        row_indices=pct_indices[:10],
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
            start_timestamp=ts_series.iloc[0].to_pydatetime() if total_rows > 0 else None,
            end_timestamp=ts_series.iloc[-1].to_pydatetime() if total_rows > 0 else None,
            missing_hours_count=missing_hours_count,
            reconciliation_discrepancy_count=reconciliation_discrepancy_count,
            duplicate_count=duplicate_count,
        )

        # Publish data quality event
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
                "missing_hours": missing_hours_count,
                "issues_count": len(issues),
            },
        )
        self.publisher.publish(evt)

        return report
