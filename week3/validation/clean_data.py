"""Apply fixes for data quality issues at load time (graceful degradation).

Pattern: detect (via check_data_quality.py) -> fix (here) -> alert (structured logs).

Used by week3/backend/data.py at startup to clean the loaded DataFrame.
- Drops rows with invalid trip_count (sentinels / negatives / > historical max)
- Deduplicates by (PULocationID, time_bucket), keeping the first occurrence
- Sets is_holiday=0 on dates that are not real US Federal holidays

If the total rows-to-drop exceeds max_drop_rate (default 5%), raises
DataLoadTooBadError so the service refuses to start with badly degraded data
(per the "fail-loud on uncertainty" design choice).

Every fix emits a structured WARNING log line, per the design principle
"silent failures are the top fear; degradation must be loud."
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

try:
    from .check_data_quality import (
        TRIP_COUNT_MAX_HISTORICAL,
        TRIP_COUNT_NEGATIVE_FLOOR,
        TRIP_COUNT_SENTINELS,
    )
    from .holiday_calendar import is_real_holiday
except ImportError:
    from check_data_quality import (  # type: ignore
        TRIP_COUNT_MAX_HISTORICAL,
        TRIP_COUNT_NEGATIVE_FLOOR,
        TRIP_COUNT_SENTINELS,
    )
    from holiday_calendar import is_real_holiday  # type: ignore

logger = logging.getLogger(__name__)


class DataLoadTooBadError(Exception):
    """Raised when too many rows would have to be dropped to clean the data."""


def clean_dataframe(
    df: pd.DataFrame,
    max_drop_rate: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a cleaned DataFrame and a report of what was changed.

    Raises DataLoadTooBadError if drop_rate exceeds max_drop_rate.
    """
    original_rows = len(df)
    report: dict[str, Any] = {
        "original_rows": original_rows,
        "fixes_applied": [],
        "rows_dropped": 0,
        "rows_modified": 0,
    }

    out = df.copy()

    # ---------- FIX 1: drop invalid trip_count rows ----------
    if "trip_count" in out.columns:
        invalid_mask = (
            out["trip_count"].isin(TRIP_COUNT_SENTINELS)
            | (out["trip_count"] < TRIP_COUNT_NEGATIVE_FLOOR)
            | (out["trip_count"] > TRIP_COUNT_MAX_HISTORICAL)
        )
        n_invalid = int(invalid_mask.sum())
        if n_invalid > 0:
            out = out[~invalid_mask].reset_index(drop=True)
            report["fixes_applied"].append({
                "fix": "drop_invalid_trip_count",
                "rows_dropped": n_invalid,
                "rule": (
                    f"trip_count in {sorted(TRIP_COUNT_SENTINELS)} "
                    f"or < {TRIP_COUNT_NEGATIVE_FLOOR} "
                    f"or > {TRIP_COUNT_MAX_HISTORICAL}"
                ),
            })
            report["rows_dropped"] += n_invalid
            logger.warning(
                "data_quality_fix=drop_invalid_trip_count rows_dropped=%d original_rows=%d",
                n_invalid, original_rows,
            )

    # ---------- FIX 2: deduplicate by (PULocationID, time_bucket) ----------
    if "PULocationID" in out.columns and "time_bucket" in out.columns:
        before = len(out)
        out = out.drop_duplicates(
            subset=["PULocationID", "time_bucket"], keep="first"
        ).reset_index(drop=True)
        n_dropped = before - len(out)
        if n_dropped > 0:
            report["fixes_applied"].append({
                "fix": "deduplicate",
                "rows_dropped": n_dropped,
                "rule": "keep first by (PULocationID, time_bucket)",
            })
            report["rows_dropped"] += n_dropped
            logger.warning(
                "data_quality_fix=deduplicate rows_dropped=%d before=%d after=%d",
                n_dropped, before, len(out),
            )

    # ---------- FIX 3: correct mislabeled is_holiday flags ----------
    if "is_holiday" in out.columns and "time_bucket" in out.columns:
        ts = pd.to_datetime(out["time_bucket"], errors="coerce")
        dates = ts.dt.date
        flagged = out["is_holiday"] == 1
        unique_dates = dates[flagged].dropna().unique()
        real_lookup = {d: is_real_holiday(d) for d in unique_dates}
        is_real = dates.map(real_lookup).fillna(False).astype(bool)
        mislabel_mask = flagged & ~is_real
        n_fixed = int(mislabel_mask.sum())
        if n_fixed > 0:
            out.loc[mislabel_mask, "is_holiday"] = 0
            affected_dates = sorted({str(d) for d in dates[mislabel_mask].dropna().unique()})
            report["fixes_applied"].append({
                "fix": "correct_holiday_labels",
                "rows_modified": n_fixed,
                "affected_dates": affected_dates,
                "rule": "set is_holiday=0 on dates not in US Federal holiday calendar",
            })
            report["rows_modified"] += n_fixed
            logger.warning(
                "data_quality_fix=correct_holiday_labels rows_modified=%d affected_dates=%s",
                n_fixed, affected_dates,
            )

    # ---------- finalize ----------
    drop_rate = report["rows_dropped"] / max(original_rows, 1)
    report["drop_rate"] = round(drop_rate, 6)
    report["final_rows"] = len(out)

    if drop_rate > max_drop_rate:
        raise DataLoadTooBadError(
            f"data_quality_fix=FAIL drop_rate={drop_rate:.2%} exceeds max_drop_rate={max_drop_rate:.2%} "
            f"(dropped {report['rows_dropped']} of {original_rows} rows). "
            f"Refusing to start with badly degraded data. fixes_applied={report['fixes_applied']}"
        )

    return out, report
