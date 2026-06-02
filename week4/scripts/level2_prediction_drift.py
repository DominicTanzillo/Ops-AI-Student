"""Week 4 - Level 2 (prediction-level) drift analysis.

Loads the LightGBM model from week2/, scores three time windows of the
week4 parquet, and reports RMSE / MAPE at the cuts required to evaluate
the locked metric design:

  baseline     : 2023-01-01 to 2026-01-15  (pre-corruption, 3+ years, clean)
  week3_corrupt: 2026-01-16 to 2026-02-01  (corrupted by Week 3 mechanisms)
  week4_drift  : 2026-02-02 to 2026-02-28  (drifted by 4 planted patterns)

Reports per window:
  - global RMSE / MAE / MAPE / Poisson deviance
  - per-borough RMSE
  - per-zone RMSE (worst 10 + best 10)
  - per (borough x is_weekend) RMSE   <- catches manhattan_weekend_concept_drift
  - per hour-of-day RMSE              <- catches temporal_peak_shift
  - prediction distribution stats (mean, std, p50, p95) for KS prep

Output: prints tables; writes week4/data/level2_metrics.json for the writeup.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO / "week2" / "model" / "lgbm_demand_model.txt"
PARQUET = REPO / "week4" / "data" / "demand_enriched_week4.parquet"
OUT_JSON = REPO / "week4" / "data" / "level2_metrics.json"

WINDOWS = {
    "baseline":      (pd.Timestamp("2023-01-01"), pd.Timestamp("2026-01-16")),  # half-open
    "week3_corrupt": (pd.Timestamp("2026-01-16"), pd.Timestamp("2026-02-02")),
    "week4_drift":   (pd.Timestamp("2026-02-02"), pd.Timestamp("2026-03-01")),
}

BOROUGH_NAMES = {0: "Manhattan", 1: "Queens", 2: "Brooklyn", 3: "Bronx", 4: "Staten Island"}


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))


def mape(y, yhat):
    # MAPE undefined at y=0; report on non-zero rows only
    m = y > 0
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(y[m] - yhat[m]) / y[m]))


def poisson_dev(y, yhat):
    # 2 * sum( y*log(y/yhat) - (y - yhat) ) per LightGBM Poisson
    yhat = np.clip(yhat, 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / yhat), 0.0) - (y - yhat)
    return float(2.0 * np.mean(term))


def summarize(y, yhat):
    return {
        "n": int(len(y)),
        "rmse": rmse(y, yhat),
        "mae": mae(y, yhat),
        "mape": mape(y, yhat),
        "poisson_dev": poisson_dev(y, yhat),
        "pred_mean": float(yhat.mean()),
        "pred_std": float(yhat.std()),
        "pred_p50": float(np.percentile(yhat, 50)),
        "pred_p95": float(np.percentile(yhat, 95)),
        "actual_mean": float(y.mean()),
    }


def main():
    print("=" * 92)
    print(f"LOAD model: {MODEL_PATH.relative_to(REPO)}")
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    features = booster.feature_name()
    print(f"  {len(features)} features, {booster.num_trees()} trees, objective=poisson")

    print(f"\nLOAD parquet: {PARQUET.relative_to(REPO)}")
    df = pd.read_parquet(PARQUET)
    print(f"  {len(df):,} rows, time_bucket range {df['time_bucket'].min()} -> {df['time_bucket'].max()}")

    print("\nSCORE all rows once, then slice per window")
    X = df[features]
    df = df.assign(yhat=booster.predict(X))
    y = df["trip_count"].astype(float).values
    yhat = df["yhat"].values

    out = {"windows": {}}

    for name, (start, end) in WINDOWS.items():
        m = (df["time_bucket"] >= start) & (df["time_bucket"] < end)
        sub = df[m]
        if len(sub) == 0:
            continue
        ys = sub["trip_count"].astype(float).values
        ps = sub["yhat"].values

        print("\n" + "=" * 92)
        print(f"WINDOW: {name}  [{start.date()}, {end.date()})   n={len(sub):,}")
        print("=" * 92)
        g = summarize(ys, ps)
        print(f"  GLOBAL    n={g['n']:>9,d}  rmse={g['rmse']:.3f}  mae={g['mae']:.3f}  "
              f"mape={g['mape']:.3f}  poisson_dev={g['poisson_dev']:.3f}  "
              f"actual_mean={g['actual_mean']:.3f}  pred_mean={g['pred_mean']:.3f}")

        # per-borough
        per_borough = {}
        print("  PER-BOROUGH:")
        for bid, g2 in sub.groupby("borough_id"):
            s = summarize(g2["trip_count"].astype(float).values, g2["yhat"].values)
            per_borough[int(bid)] = s
            print(f"    borough={int(bid)} ({BOROUGH_NAMES.get(int(bid),'?'):>13}) "
                  f"n={s['n']:>9,d}  rmse={s['rmse']:.3f}  mape={s['mape']:.3f}  "
                  f"actual_mean={s['actual_mean']:.3f}")

        # per (borough x is_weekend) - manhattan_weekend_concept_drift cut
        per_borough_weekend = {}
        print("  PER (borough x is_weekend):")
        for (bid, w), g2 in sub.groupby(["borough_id", "is_weekend"]):
            s = summarize(g2["trip_count"].astype(float).values, g2["yhat"].values)
            per_borough_weekend[f"b{int(bid)}_w{int(w)}"] = s
            print(f"    b={int(bid)} w={int(w)} n={s['n']:>9,d}  rmse={s['rmse']:.3f}  "
                  f"mape={s['mape']:.3f}  actual_mean={s['actual_mean']:.3f}")

        # per hour-of-day - temporal_peak_shift cut
        per_hour = {}
        print("  PER hour-of-day (worst 6 by RMSE):")
        hour_summaries = {}
        for h, g2 in sub.groupby("hour"):
            s = summarize(g2["trip_count"].astype(float).values, g2["yhat"].values)
            hour_summaries[int(h)] = s
        per_hour = hour_summaries
        for h, s in sorted(hour_summaries.items(), key=lambda kv: -kv[1]["rmse"])[:6]:
            print(f"    hour={h:>2}  n={s['n']:>8,d}  rmse={s['rmse']:.3f}  "
                  f"actual_mean={s['actual_mean']:.3f}  pred_mean={s['pred_mean']:.3f}")

        # per-zone - worst 10 by RMSE (outer_borough_baseline_scramble suspect)
        per_zone_top = {}
        print("  PER-ZONE worst 10 by RMSE:")
        zone_summaries = {}
        for z, g2 in sub.groupby("PULocationID"):
            s = summarize(g2["trip_count"].astype(float).values, g2["yhat"].values)
            zone_summaries[int(z)] = s
        for z, s in sorted(zone_summaries.items(), key=lambda kv: -kv[1]["rmse"])[:10]:
            per_zone_top[z] = s
            print(f"    zone={z:>4}  n={s['n']:>7,d}  rmse={s['rmse']:.3f}  "
                  f"actual_mean={s['actual_mean']:.3f}  pred_mean={s['pred_mean']:.3f}")

        out["windows"][name] = {
            "start": str(start), "end": str(end),
            "global": g,
            "per_borough": per_borough,
            "per_borough_weekend": per_borough_weekend,
            "per_hour": per_hour,
            "per_zone_worst10": per_zone_top,
        }

    # Relative deltas baseline -> week4_drift, for the writeup headline numbers
    if "baseline" in out["windows"] and "week4_drift" in out["windows"]:
        b = out["windows"]["baseline"]["global"]
        d = out["windows"]["week4_drift"]["global"]
        print("\n" + "=" * 92)
        print("HEADLINE DELTA: baseline -> week4_drift (Feb 2-28)")
        print("=" * 92)
        for k in ("rmse", "mae", "mape", "poisson_dev"):
            if not (np.isnan(b[k]) or np.isnan(d[k])):
                rel = (d[k] - b[k]) / b[k] * 100
                print(f"  {k:>14}: {b[k]:.4f} -> {d[k]:.4f}   ({rel:+.1f}%)")

        # per-borough deltas
        print("\n  PER-BOROUGH RMSE: baseline -> week4_drift")
        for bid in sorted(out["windows"]["baseline"]["per_borough"].keys()):
            bb = out["windows"]["baseline"]["per_borough"][bid]
            dd = out["windows"]["week4_drift"]["per_borough"].get(bid)
            if dd:
                rel = (dd["rmse"] - bb["rmse"]) / bb["rmse"] * 100
                print(f"    b={bid} ({BOROUGH_NAMES.get(int(bid),'?'):>13}): "
                      f"{bb['rmse']:.3f} -> {dd['rmse']:.3f}   ({rel:+.1f}%)")

        # per (borough x is_weekend) deltas - the concept drift probe
        print("\n  PER (borough x is_weekend) RMSE: baseline -> week4_drift")
        for k in sorted(out["windows"]["baseline"]["per_borough_weekend"].keys()):
            bb = out["windows"]["baseline"]["per_borough_weekend"][k]
            dd = out["windows"]["week4_drift"]["per_borough_weekend"].get(k)
            if dd:
                rel = (dd["rmse"] - bb["rmse"]) / bb["rmse"] * 100
                print(f"    {k}: {bb['rmse']:.3f} -> {dd['rmse']:.3f}   ({rel:+.1f}%)")

    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT_JSON.relative_to(REPO)}")


if __name__ == "__main__":
    main()
