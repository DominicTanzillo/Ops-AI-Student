"""Tests for DataQualityValidator.

Per-check: positive (triggers), negative (clean passes), edge (missing column / single row).
Plus 2 integration smoke tests against the actual parquet files.

Run from repo root:
    pytest week3/validation/test_data_quality.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# allow `from validation.check_data_quality import ...` when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation.check_data_quality import DataQualityValidator, TRIP_COUNT_MAX_HISTORICAL


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def validator():
    return DataQualityValidator()


@pytest.fixture
def clean_df():
    """A small, internally-consistent clean dataframe with all expected columns."""
    # 3 zones x 4 time buckets = 12 rows, 15-min slots starting 2025-09-01 00:00
    times = pd.date_range("2025-09-01 00:00", periods=4, freq="15min")
    zones = [4, 13, 24]
    rows = []
    rng = np.random.default_rng(42)
    for z in zones:
        for t in times:
            rows.append({
                "PULocationID": z,
                "time_bucket": t,
                "trip_count": int(rng.integers(0, 20)),
                "is_holiday": 0,
                "lag_1week": float(rng.integers(0, 20)),
            })
    return pd.DataFrame(rows)


# ======================================================================
# check_value_ranges
# ======================================================================
class TestValueRanges:
    def test_positive_sentinel_minus_5(self, validator, clean_df):
        df = clean_df.copy()
        df.loc[0, "trip_count"] = -5
        issues = validator.check_value_ranges(df)
        assert any(i.check == "value_ranges" and i.details.get("value") == -5 for i in issues)

    def test_positive_sentinel_99999(self, validator, clean_df):
        df = clean_df.copy()
        df.loc[0, "trip_count"] = 99999
        issues = validator.check_value_ranges(df)
        assert any(i.details.get("value") == 99999 for i in issues)
        # affected_rows for that sentinel = 1
        sentinel_issues = [i for i in issues if i.details.get("value") == 99999]
        assert sentinel_issues[0].affected_rows == 1

    def test_positive_extreme_outlier(self, validator, clean_df):
        df = clean_df.copy()
        df.loc[0, "trip_count"] = TRIP_COUNT_MAX_HISTORICAL + 50  # 360, not a sentinel
        issues = validator.check_value_ranges(df)
        assert any("historical max" in i.description for i in issues)

    def test_negative_clean(self, validator, clean_df):
        """Clean dataframe should yield no value_range issues."""
        issues = validator.check_value_ranges(clean_df)
        assert issues == []

    def test_edge_missing_column(self, validator, clean_df):
        df = clean_df.drop(columns=["trip_count"])
        issues = validator.check_value_ranges(df)
        assert len(issues) == 1
        assert "missing" in issues[0].description.lower()

    def test_edge_zero_is_valid(self, validator, clean_df):
        """trip_count=0 is legitimate (rare zones at 3am), not a sentinel. No issue should fire."""
        df = clean_df.copy()
        df["trip_count"] = 0
        issues = validator.check_value_ranges(df)
        assert issues == [], f"trip_count=0 should be valid; got: {[i.description for i in issues]}"


# ======================================================================
# check_holiday_labels
# ======================================================================
class TestHolidayLabels:
    def test_positive_mislabeled_date(self, validator, clean_df):
        """is_holiday=1 on a Tuesday that isn't a real holiday should fire."""
        df = clean_df.copy()
        df["is_holiday"] = 1  # 2025-09-01 is Labor Day (a holiday); but the OTHER rows in clean_df are Sep 1 too
        # use 2025-09-02 instead, which is NOT a holiday
        df["time_bucket"] = pd.date_range("2025-09-02 00:00", periods=len(df), freq="15min")
        issues = validator.check_holiday_labels(df)
        assert len(issues) == 1
        assert issues[0].severity == "critical"
        assert "2025-09-02" in issues[0].details["affected_dates"]

    def test_negative_real_holiday_flagged_ok(self, validator, clean_df):
        """is_holiday=1 on Christmas Day should NOT fire."""
        df = clean_df.copy()
        df["is_holiday"] = 1
        df["time_bucket"] = pd.date_range("2025-12-25 00:00", periods=len(df), freq="15min")
        issues = validator.check_holiday_labels(df)
        assert issues == []

    def test_negative_clean(self, validator, clean_df):
        """Default clean_df has is_holiday=0 everywhere → no issue."""
        issues = validator.check_holiday_labels(clean_df)
        assert issues == []

    def test_edge_missing_is_holiday_column(self, validator, clean_df):
        df = clean_df.drop(columns=["is_holiday"])
        issues = validator.check_holiday_labels(df)
        assert len(issues) == 1
        assert "missing" in issues[0].description.lower()

    def test_edge_only_real_and_mislabeled_dates(self, validator, clean_df):
        """A df where SOME holiday flags are correct (Christmas) and SOME wrong (random Tuesday).
        Should fire only for the wrong dates."""
        df1 = clean_df.copy()
        df1["is_holiday"] = 1
        df1["time_bucket"] = pd.date_range("2025-12-25 00:00", periods=len(df1), freq="15min")
        df2 = clean_df.copy()
        df2["is_holiday"] = 1
        df2["time_bucket"] = pd.date_range("2025-09-02 00:00", periods=len(df2), freq="15min")  # not a holiday
        df = pd.concat([df1, df2], ignore_index=True)
        issues = validator.check_holiday_labels(df)
        assert len(issues) == 1
        assert "2025-09-02" in issues[0].details["affected_dates"]
        assert "2025-12-25" not in issues[0].details["affected_dates"]


# ======================================================================
# check_duplicates
# ======================================================================
class TestDuplicates:
    def test_positive_one_duplicate(self, validator, clean_df):
        df = pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True)
        issues = validator.check_duplicates(df)
        assert len(issues) == 1
        assert issues[0].affected_rows == 1
        assert issues[0].details["duplicate_key_count"] == 1

    def test_positive_multiple_duplicates(self, validator, clean_df):
        df = pd.concat([clean_df, clean_df.iloc[:3]], ignore_index=True)
        issues = validator.check_duplicates(df)
        assert len(issues) == 1
        assert issues[0].affected_rows == 3

    def test_negative_clean(self, validator, clean_df):
        assert validator.check_duplicates(clean_df) == []

    def test_edge_missing_column(self, validator, clean_df):
        df = clean_df.drop(columns=["PULocationID"])
        issues = validator.check_duplicates(df)
        assert len(issues) == 1
        assert "missing" in issues[0].description.lower()

    def test_edge_single_row(self, validator):
        df = pd.DataFrame({"PULocationID": [1], "time_bucket": [pd.Timestamp("2025-09-01")]})
        # single row cannot have a duplicate
        assert validator.check_duplicates(df) == []


# ======================================================================
# integration: validate() across all checks
# ======================================================================
class TestIntegration:
    def test_clean_df_passes(self, validator, clean_df):
        result = validator.validate(clean_df)
        assert result["passed"] is True
        assert result["issues"] == []
        assert set(result["checks_run"]) == {"value_ranges", "holiday_labels", "duplicates"}

    def test_dirty_df_fails(self, validator, clean_df):
        """Inject one issue per check; assert all three fire."""
        df = clean_df.copy()
        df.loc[0, "trip_count"] = 99999
        df.loc[1, "is_holiday"] = 1  # Sep 1 in clean_df IS Labor Day, so change date first
        df.loc[1, "time_bucket"] = pd.Timestamp("2025-09-02 00:30")  # not a holiday
        df = pd.concat([df, df.iloc[[2]]], ignore_index=True)  # duplicate
        result = validator.validate(df)
        assert result["passed"] is False
        checks_with_issues = {i["check"] for i in result["issues"]}
        assert "value_ranges" in checks_with_issues
        assert "holiday_labels" in checks_with_issues
        assert "duplicates" in checks_with_issues


# ======================================================================
# smoke tests against real parquet data (skipped if files not present)
# ======================================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORRUPTED_PARQUET = DATA_DIR / "demand_enriched_corrupted.parquet"


@pytest.fixture(scope="module")
def full_corrupted_df():
    if not CORRUPTED_PARQUET.exists():
        pytest.skip(f"{CORRUPTED_PARQUET} not present locally")
    df = pd.read_parquet(CORRUPTED_PARQUET)
    df["time_bucket"] = pd.to_datetime(df["time_bucket"])
    return df


class TestRealData:
    def test_smoke_bad_batch_fails(self, validator, full_corrupted_df):
        """The Jan 16 - Feb 1 corrupted batch must fail validation with at least value_ranges + holiday_labels issues."""
        new = full_corrupted_df[
            (full_corrupted_df["time_bucket"] >= "2026-01-16")
            & (full_corrupted_df["time_bucket"] < "2026-02-02")
        ]
        result = validator.validate(new)
        assert result["passed"] is False
        checks_with_issues = {i["check"] for i in result["issues"]}
        assert "value_ranges" in checks_with_issues
        assert "holiday_labels" in checks_with_issues

    def test_smoke_clean_slice_passes(self, validator, full_corrupted_df):
        """A clean historical slice (Sep 1-17, 2025) far from any documented corruption must pass."""
        clean = full_corrupted_df[
            (full_corrupted_df["time_bucket"] >= "2025-09-01")
            & (full_corrupted_df["time_bucket"] < "2025-09-18")
        ]
        result = validator.validate(clean)
        assert result["passed"] is True, (
            f"Clean historical slice should pass; got issues: {[i['description'] for i in result['issues']]}"
        )
