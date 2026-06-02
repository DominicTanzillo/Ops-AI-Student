"""Tests for the 8 monitoring metrics + the 4-pattern drift detector.

Two layers:
  - Unit tests per metric: a tiny synthetic baseline + new DataFrame,
    one positive case (alert fires) and one negative case (clean data),
    so we can verify the metric reports the right structure and direction.
  - Integration test: runs compute_metrics + detect_drift end-to-end
    against the real parquet (skipped if not present in the test env).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from metric_template import (
    MetricComputer,
    _poisson_deviance,
    _psi,
    _mape,
)

REPO = HERE.parents[1]
REAL_PARQUET = REPO / "week4" / "data" / "demand_enriched_week4.parquet"
REAL_MODEL = REPO / "week2" / "model" / "lgbm_demand_model.txt"


# ----------------------------------------------------------------------
# Synthetic data builders
# ----------------------------------------------------------------------
def _build_baseline(n_per_zone: int = 200, zones=(1, 2, 3, 4)) -> pd.DataFrame:
    """A clean synthetic baseline: 4 zones across 2 boroughs, 24 hours, two weekends."""
    rng = np.random.default_rng(0)
    rows = []
    start = pd.Timestamp("2026-01-01")
    for z in zones:
        borough = 0 if z <= 2 else 1
        for i in range(n_per_zone):
            ts = start + pd.Timedelta(minutes=15 * i)
            rows.append({
                "PULocationID": z,
                "time_bucket": ts,
                "trip_count": int(rng.poisson(lam=10)),
                "hour": ts.hour,
                "is_weekend": int(ts.weekday() >= 5),
                "borough_id": borough,
                "lag_1day": float(rng.normal(10, 2)),
                "lag_1week": float(rng.normal(10, 2)),
                "roll_mean_1day": float(rng.normal(10, 2)),
            })
    df = pd.DataFrame(rows)
    df["yhat"] = df["trip_count"].astype(float) + 0.1  # near-perfect baseline predictions
    return df


def _build_new_clean(baseline: pd.DataFrame, n_per_zone: int = 50) -> pd.DataFrame:
    """A clean new window drawn from the same distribution as baseline."""
    rng = np.random.default_rng(1)
    rows = []
    start = pd.Timestamp("2026-02-02")
    for z in sorted(baseline["PULocationID"].unique()):
        borough = int(baseline.loc[baseline["PULocationID"] == z, "borough_id"].iloc[0])
        for i in range(n_per_zone):
            ts = start + pd.Timedelta(minutes=15 * i)
            rows.append({
                "PULocationID": z,
                "time_bucket": ts,
                "trip_count": int(rng.poisson(lam=10)),
                "hour": ts.hour,
                "is_weekend": int(ts.weekday() >= 5),
                "borough_id": borough,
                "lag_1day": float(rng.normal(10, 2)),
                "lag_1week": float(rng.normal(10, 2)),
                "roll_mean_1day": float(rng.normal(10, 2)),
            })
    return pd.DataFrame(rows)


def _build_new_drifted(baseline: pd.DataFrame, n_per_zone: int = 50,
                       shift_lambda: float = 25.0,
                       inject_dupes: bool = False,
                       null_fraction: float = 0.0) -> pd.DataFrame:
    """A drifted new window: trip_count drawn from much higher Poisson."""
    rng = np.random.default_rng(2)
    rows = []
    start = pd.Timestamp("2026-02-02")
    for z in sorted(baseline["PULocationID"].unique()):
        borough = int(baseline.loc[baseline["PULocationID"] == z, "borough_id"].iloc[0])
        for i in range(n_per_zone):
            ts = start + pd.Timedelta(minutes=15 * i)
            rows.append({
                "PULocationID": z,
                "time_bucket": ts,
                "trip_count": int(rng.poisson(lam=shift_lambda)),
                "hour": ts.hour,
                "is_weekend": int(ts.weekday() >= 5),
                "borough_id": borough,
                "lag_1day": float(rng.normal(25, 4)),
                "lag_1week": float(rng.normal(25, 4)),
                "roll_mean_1day": float(rng.normal(25, 4)),
            })
    df = pd.DataFrame(rows)
    if inject_dupes:
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    if null_fraction > 0:
        idx = rng.choice(len(df), int(len(df) * null_fraction), replace=False)
        df.loc[idx, "trip_count"] = np.nan
    return df


# ----------------------------------------------------------------------
# Stat helper unit tests
# ----------------------------------------------------------------------
def test_poisson_deviance_zero_when_perfect():
    y = np.array([3.0, 1.0, 7.0, 0.0, 4.0])
    assert _poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-6)


def test_poisson_deviance_grows_with_error():
    y = np.array([3.0, 1.0, 7.0, 0.0, 4.0])
    near = _poisson_deviance(y, y + 0.1)
    far = _poisson_deviance(y, y * 2 + 1)
    assert far > near


def test_psi_zero_on_identical_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert _psi(a, a) == pytest.approx(0.0, abs=1e-6)


def test_psi_fires_on_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(2, 1, 5000)
    assert _psi(a, b) > 0.25


def test_mape_skips_zero_actuals():
    y = np.array([0.0, 0.0, 10.0])
    yhat = np.array([5.0, 5.0, 12.0])
    # Should only score the one non-zero row -> |10-12|/10 = 0.2
    assert _mape(y, yhat) == pytest.approx(0.2)


# ----------------------------------------------------------------------
# Metric unit tests (positive + negative case for each)
# ----------------------------------------------------------------------
@pytest.fixture
def baseline_df():
    return _build_baseline()


@pytest.fixture
def computer(baseline_df):
    return MetricComputer(baseline_df)


def test_metric_1_accuracy_baseline_returns_dict(computer, baseline_df):
    new = _build_new_clean(baseline_df)
    preds = new["trip_count"].astype(float).values
    actuals = new["trip_count"].astype(float).values
    out = computer.metric_1_accuracy(new, preds, actuals)
    assert isinstance(out, dict)
    for k in ("poisson_deviance", "rmse", "mape", "actual_mean", "pred_mean"):
        assert k in out


def test_metric_1_accuracy_negative_when_perfect(computer, baseline_df):
    new = _build_new_clean(baseline_df)
    preds = new["trip_count"].astype(float).values
    actuals = new["trip_count"].astype(float).values
    out = computer.metric_1_accuracy(new, preds, actuals)
    assert out["poisson_deviance"] == pytest.approx(0.0, abs=1e-6)


def test_metric_1_accuracy_positive_on_bad_predictions(computer, baseline_df):
    new = _build_new_clean(baseline_df)
    actuals = new["trip_count"].astype(float).values
    bad_preds = actuals + 5  # consistently over-predict
    out = computer.metric_1_accuracy(new, bad_preds, actuals)
    assert out["poisson_deviance"] > 0.5


def test_metric_2_accuracy_by_zone_segments(computer, baseline_df):
    new = _build_new_clean(baseline_df)
    preds = new["trip_count"].astype(float).values
    actuals = new["trip_count"].astype(float).values
    out = computer.metric_2_accuracy_by_zone(new, preds, actuals)
    assert set(out.keys()) >= {"per_zone", "per_borough", "per_borough_weekend"}
    assert len(out["per_zone"]) == new["PULocationID"].nunique()


def test_metric_3_null_rates_clean(computer, baseline_df):
    new = _build_new_clean(baseline_df)
    out = computer.metric_3_null_rates(new)
    assert out["trip_count"] == 0.0


def test_metric_3_null_rates_fires_on_nulls(computer, baseline_df):
    new = _build_new_drifted(baseline_df, null_fraction=0.05)
    out = computer.metric_3_null_rates(new)
    assert out["trip_count"] > 0.02


def test_metric_4_ks_no_alert_on_clean(computer, baseline_df):
    new = _build_new_clean(baseline_df)
    out = computer.metric_4_ks_test(new)
    assert out["global"]["pvalue"] >= 0.01


def test_metric_4_ks_fires_on_distribution_shift(computer, baseline_df):
    new = _build_new_drifted(baseline_df, shift_lambda=30)
    out = computer.metric_4_ks_test(new)
    assert out["global"]["pvalue"] < 0.01


def test_metric_5_psi_low_on_clean(computer, baseline_df):
    new = _build_new_clean(baseline_df)
    out = computer.metric_5_psi(new)
    # all global PSI values should be small for unchanged distributions
    assert all(v < 0.20 for v in out["global"].values())


def test_metric_5_psi_fires_on_lag_drift(computer, baseline_df):
    new = _build_new_drifted(baseline_df, shift_lambda=30)
    out = computer.metric_5_psi(new)
    # at least one lag feature should fire critical
    assert any(v >= 0.25 for v in out["global"].values())


def test_metric_6_prediction_distribution_reports_collapse(computer):
    preds_collapsed = np.full(1000, 5.0)
    out = computer.metric_6_prediction_distribution(preds_collapsed)
    assert out["std"] < 0.01
    if "baseline_std" in out:
        assert out["collapsed"] is True


def test_metric_6_prediction_distribution_clean(computer):
    rng = np.random.default_rng(0)
    preds = rng.normal(10, 3, 1000)
    out = computer.metric_6_prediction_distribution(preds)
    assert out["std"] > 1.0


def test_metric_7_data_freshness_fresh(computer, baseline_df):
    new = _build_new_clean(baseline_df)
    ref = new["time_bucket"].max() + pd.Timedelta(minutes=10)
    out = computer.metric_7_data_freshness(new, reference_time=ref)
    assert out["stale"] is False
    assert out["age_minutes"] < 120


def test_metric_7_data_freshness_stale(computer, baseline_df):
    new = _build_new_clean(baseline_df)
    ref = new["time_bucket"].max() + pd.Timedelta(hours=5)
    out = computer.metric_7_data_freshness(new, reference_time=ref)
    assert out["stale"] is True


def test_metric_8_duplicate_rate_zero_clean(computer, baseline_df):
    new = _build_new_clean(baseline_df)
    out = computer.metric_8_duplicate_rate(new)
    assert out["count"] == 0


def test_metric_8_duplicate_rate_fires(computer, baseline_df):
    new = _build_new_drifted(baseline_df, inject_dupes=True)
    out = computer.metric_8_duplicate_rate(new)
    assert out["count"] > 0


def test_compute_all_metrics_returns_all_8(computer, baseline_df):
    new = _build_new_clean(baseline_df)
    preds = new["trip_count"].astype(float).values
    actuals = new["trip_count"].astype(float).values
    out = computer.compute_all_metrics(new, predictions=preds, actuals=actuals)
    expected = {
        "metric_1_accuracy", "metric_2_accuracy_by_zone",
        "metric_3_null_rates", "metric_4_ks_test", "metric_5_psi",
        "metric_6_prediction_distribution",
        "metric_7_data_freshness", "metric_8_duplicate_rate",
    }
    assert set(out.keys()) == expected


# ----------------------------------------------------------------------
# Integration test against the real upstream parquet
# ----------------------------------------------------------------------
@pytest.mark.skipif(not REAL_PARQUET.exists(),
                    reason="real parquet absent (CI sparse-checkout will provide)")
@pytest.mark.skipif(not REAL_MODEL.exists(),
                    reason="real model absent in this checkout")
def test_real_parquet_fires_expected_patterns():
    """Smoke test: against the real Week 4 parquet, the four planted drift
    patterns should each surface in the expected metric / segment."""
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(REAL_MODEL))
    features = booster.feature_name()
    df = pd.read_parquet(REAL_PARQUET)
    baseline = df[df["time_bucket"] < pd.Timestamp("2026-01-16")].copy()
    new = df[(df["time_bucket"] >= pd.Timestamp("2026-02-02"))
             & (df["time_bucket"] < pd.Timestamp("2026-03-01"))].copy()

    baseline["yhat"] = booster.predict(baseline[features])
    new_preds = booster.predict(new[features])
    new_actuals = new["trip_count"].astype(float).values

    mc = MetricComputer(baseline)
    metrics = mc.compute_all_metrics(new, predictions=new_preds, actuals=new_actuals)

    # 1. global Poisson deviance should fire critical (Level 2 measured +34%)
    assert metrics["metric_1_accuracy"]["deviance_rel_delta"] > 0.20, \
        "global concept drift should fire critical"

    # 2. Manhattan weekend should be the worst per-segment.
    # Segment keys encode (borough_id, is_weekend) as "b{B}_w{W}".
    seg = metrics["metric_2_accuracy_by_zone"]["per_borough_weekend"]
    worst_key = max(seg.items(),
                    key=lambda kv: kv[1].get("deviance_rel_delta", -1))[0]
    assert worst_key == "b0_w1", \
        f"expected Manhattan-weekend (b0_w1) as worst segment, got {worst_key}"

    # 3. Manhattan lag PSI should be critical, Brooklyn lag PSI should NOT
    bps = metrics["metric_5_psi"]["per_borough"]
    assert any(v >= 0.25 for v in bps[0].values()), \
        "Manhattan lag features should show critical PSI"
    assert all(v < 0.25 for v in bps[2].values()), \
        "Brooklyn lag features should NOT show critical PSI (deflation was Manhattan-only)"

    # 4. KS test should fire on per-hour cuts (temporal peak shift)
    ks = metrics["metric_4_ks_test"]["per_hour"]
    significant_hours = [h for h, s in ks.items() if s["pvalue"] < 0.01]
    assert len(significant_hours) >= 3, \
        "temporal peak shift should produce multiple hour-level KS criticals"
