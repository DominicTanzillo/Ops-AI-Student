"""Data quality validator for the demand-forecast pipeline.

Three checks, all CRITICAL severity (per design decision):
  1. value_ranges    - trip_count sentinel / negative / extreme-outlier values
  2. holiday_labels  - is_holiday=1 on a date not in the US Federal calendar
  3. duplicates      - duplicate (PULocationID, time_bucket) keys

Severity rationale: a deliberate, narrow set of "if any of these fire, the
data is broken and predictions will be wrong" checks. Drift-style softer
signals belong in Week 4 monitoring, not here.

Scope note - lag-column tampering not detected here:
  An earlier draft included a `check_lag_contamination` that compared per-zone
  lag_1week vs trip_count correlation against a historical baseline. Empirical
  testing showed ~28% false-positive rate on clean 17-day historical slices
  (16 of 57 zones falsely flagged) because Pearson correlation on small
  windows has high natural variance. The check was removed as unreliable at
  this layer; lag tampering should be caught by Week 4's drift monitoring,
  which operates on longer rolling windows and statistical tests built for
  detecting feature drift over time.

Usage:
    from validation.check_data_quality import DataQualityValidator
    validator = DataQualityValidator()
    result = validator.validate(df)
    if not result["passed"]:
        # at least one CRITICAL check failed
        for issue in result["issues"]:
            logger.error("data quality issue: %s", issue)
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    # package-relative import (when run as a module)
    from .holiday_calendar import is_real_holiday
except ImportError:
    # fallback for direct-script use
    from holiday_calendar import is_real_holiday  # type: ignore

logger = logging.getLogger(__name__)

# Tunables - kept here as module constants so they're easy to find / change
TRIP_COUNT_SENTINELS = {-5, -1, 9999, 99999}
TRIP_COUNT_MAX_HISTORICAL = 310  # from baseline drill-down
TRIP_COUNT_NEGATIVE_FLOOR = 0    # trip counts cannot be negative


@dataclass
class Issue:
    """One detected data quality issue."""
    check: str
    severity: str  # "critical" | "warning" | "info"
    description: str
    details: dict[str, Any] = field(default_factory=dict)
    affected_rows: int = 0


class DataQualityValidator:
    """Runs the three data quality checks on a DataFrame and returns structured results."""

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------
    def validate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run all checks. Return dict with passed/issues/summary."""
        issues: list[Issue] = []
        issues.extend(self.check_value_ranges(df))
        issues.extend(self.check_holiday_labels(df))
        issues.extend(self.check_duplicates(df))

        critical = [i for i in issues if i.severity == "critical"]
        return {
            "passed": len(critical) == 0,
            "checks_run": ["value_ranges", "holiday_labels", "duplicates"],
            "issues": [asdict(i) for i in issues],
            "summary": self._summary(issues),
        }

    # ------------------------------------------------------------------
    # individual checks
    # ------------------------------------------------------------------
    def check_value_ranges(self, df: pd.DataFrame) -> list[Issue]:
        """Detect sentinel values, negatives, and out-of-range maxes in trip_count."""
        if "trip_count" not in df.columns:
            return [Issue("value_ranges", "critical", "column trip_count is missing")]

        issues: list[Issue] = []
        tc = df["trip_count"]
        sentinel_mask = tc.isin(TRIP_COUNT_SENTINELS)

        # one issue per sentinel value present (so logs are searchable per value)
        for sv in sorted(TRIP_COUNT_SENTINELS):
            sub = df[tc == sv]
            if len(sub) > 0:
                issues.append(Issue(
                    check="value_ranges",
                    severity="critical",
                    description=f"trip_count sentinel value {sv} detected",
                    details={
                        "value": int(sv),
                        "count": int(len(sub)),
                        "zones": sorted(sub["PULocationID"].astype(int).unique().tolist())[:20],
                    },
                    affected_rows=int(len(sub)),
                ))

        # negatives NOT already counted as sentinels
        neg_mask = (tc < TRIP_COUNT_NEGATIVE_FLOOR) & ~sentinel_mask
        if neg_mask.any():
            sub = df[neg_mask]
            issues.append(Issue(
                check="value_ranges",
                severity="critical",
                description="trip_count negative values outside sentinel set",
                details={
                    "count": int(len(sub)),
                    "min": float(sub["trip_count"].min()),
                },
                affected_rows=int(len(sub)),
            ))

        # extreme outliers NOT already counted as sentinels
        outlier_mask = (tc > TRIP_COUNT_MAX_HISTORICAL) & ~sentinel_mask
        if outlier_mask.any():
            sub = df[outlier_mask]
            issues.append(Issue(
                check="value_ranges",
                severity="critical",
                description=f"trip_count > historical max ({TRIP_COUNT_MAX_HISTORICAL})",
                details={
                    "count": int(len(sub)),
                    "max_observed": float(sub["trip_count"].max()),
                    "historical_max": TRIP_COUNT_MAX_HISTORICAL,
                },
                affected_rows=int(len(sub)),
            ))

        return issues

    def check_holiday_labels(self, df: pd.DataFrame) -> list[Issue]:
        """Detect is_holiday=1 on dates not in the US Federal calendar."""
        for col in ("is_holiday", "time_bucket"):
            if col not in df.columns:
                return [Issue("holiday_labels", "critical", f"column {col} is missing")]

        ts = pd.to_datetime(df["time_bucket"], errors="coerce")
        dates = ts.dt.date
        flagged_mask = (df["is_holiday"] == 1)
        # apply once over unique dates for speed instead of per-row
        unique_dates = dates[flagged_mask].dropna().unique()
        real_lookup = {d: is_real_holiday(d) for d in unique_dates}
        is_real = dates.map(real_lookup).fillna(False).astype(bool)

        mislabel_mask = flagged_mask & ~is_real
        if mislabel_mask.any():
            sub = df[mislabel_mask]
            sub_dates = sorted({str(d) for d in dates[mislabel_mask].dropna().unique()})
            return [Issue(
                check="holiday_labels",
                severity="critical",
                description="is_holiday=1 on dates that are not real US Federal holidays",
                details={
                    "affected_dates": sub_dates,
                    "date_count": len(sub_dates),
                    "affected_rows": int(len(sub)),
                },
                affected_rows=int(len(sub)),
            )]
        return []

    def check_duplicates(self, df: pd.DataFrame) -> list[Issue]:
        """Detect duplicate (PULocationID, time_bucket) rows."""
        for col in ("PULocationID", "time_bucket"):
            if col not in df.columns:
                return [Issue("duplicates", "critical", f"column {col} is missing")]

        keys = df.groupby(["PULocationID", "time_bucket"]).size()
        dup_keys = keys[keys > 1]
        extra = int((dup_keys - 1).sum())
        if extra == 0:
            return []

        affected_zones = sorted(int(z) for z in dup_keys.index.get_level_values("PULocationID").unique())
        return [Issue(
            check="duplicates",
            severity="critical",
            description="duplicate (PULocationID, time_bucket) keys detected",
            details={
                "duplicate_key_count": int(len(dup_keys)),
                "extra_rows_beyond_first": extra,
                "affected_zone_count": len(affected_zones),
                "affected_zones": affected_zones[:25],
                "time_range": [
                    str(dup_keys.index.get_level_values("time_bucket").min()),
                    str(dup_keys.index.get_level_values("time_bucket").max()),
                ],
            },
            affected_rows=extra,
        )]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _summary(issues: list[Issue]) -> dict[str, int]:
        """Roll up issue counts by check name for a metrics endpoint or dashboard."""
        out: dict[str, int] = {}
        for i in issues:
            out[f"{i.check}_count"] = out.get(f"{i.check}_count", 0) + 1
            out[f"{i.check}_rows"] = out.get(f"{i.check}_rows", 0) + i.affected_rows
        return out


# ----------------------------------------------------------------------
# CLI: run validator against a parquet file and print JSON to stdout
# ----------------------------------------------------------------------
def main():
    import argparse
    p = argparse.ArgumentParser(description="Run data quality validation on a parquet file.")
    p.add_argument("parquet_path", help="path to parquet file to validate")
    p.add_argument("--fail-on-critical", action="store_true",
                   help="exit non-zero if any CRITICAL issue is found (for CI gating)")
    args = p.parse_args()

    df = pd.read_parquet(args.parquet_path)
    v = DataQualityValidator()
    result = v.validate(df)

    print(json.dumps(result, indent=2, default=str))
    if args.fail_on_critical and not result["passed"]:
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
