"""Drift pattern detector - statistical investigation focused on the 4-pattern
deliverable required by the Week 4 README.

compute_metrics.py runs the 8 monitoring metrics for routine CI gating.
detect_drift.py runs a deeper, more readable analysis specifically aimed
at identifying the distinct drift patterns and their statistical signatures.

Patterns this looks for:
  1. Temporal peak shift           - per-hour-of-day KS on trip_count
  2. Borough-level feature drift   - PSI on lag features per-borough
  3. Per-zone actual-vs-pred divergence - identifies worst segments and
                                          flags zones whose actual_mean
                                          shifted >40% (likely scramble)
  4. Borough x is_weekend concept drift - Poisson deviance per slice
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

HERE = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE))
    from metric_template import _poisson_deviance, _psi, _mape  # type: ignore
else:
    from .metric_template import _poisson_deviance, _psi, _mape

BOROUGH_NAMES = {0: "Manhattan", 1: "Queens", 2: "Brooklyn",
                 3: "Bronx", 4: "Staten Island"}
LAG_FEATURES = ["lag_1day", "lag_1week", "roll_mean_1day"]


def detect_temporal_peak_shift(baseline: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Pattern 1: per-hour-of-day KS test on trip_count.

    Two adjacent hours showing much higher KS statistics than the rest is the
    signature of a peak-time demand shift.
    """
    per_hour = {}
    for h in sorted(new["hour"].unique()):
        b = baseline.loc[baseline["hour"] == h, "trip_count"].values
        n = new.loc[new["hour"] == h, "trip_count"].values
        if len(b) > 0 and len(n) > 0:
            s, p = ks_2samp(b, n)
            per_hour[int(h)] = {
                "ks_statistic": float(s), "ks_pvalue": float(p),
                "baseline_mean": float(b.mean()), "new_mean": float(n.mean()),
            }
    worst = sorted(per_hour.items(), key=lambda kv: -kv[1]["ks_statistic"])[:5]
    return {
        "per_hour": per_hour,
        "worst_5_hours_by_ks": [(h, s) for h, s in worst],
        "interpretation": (
            "Two adjacent hours with KS statistic >2x the median => "
            "demand profile shifted (peak suppression or boost)."
        ),
    }


def detect_borough_feature_drift(baseline: pd.DataFrame, new: pd.DataFrame) -> dict:
    """Pattern 2: PSI on lag features per borough."""
    out = {}
    for b in sorted(new["borough_id"].unique()):
        b = int(b)
        bsub = baseline[baseline["borough_id"] == b]
        nsub = new[new["borough_id"] == b]
        if len(bsub) == 0:
            continue
        psi_per_feat = {f: _psi(bsub[f].values, nsub[f].values) for f in LAG_FEATURES}
        valid = [v for v in psi_per_feat.values() if not np.isnan(v)]
        out[b] = {
            "borough_name": BOROUGH_NAMES.get(b, "?"),
            "n_rows_baseline": int(len(bsub)),
            "n_rows_new": int(len(nsub)),
            "psi": psi_per_feat,
            "max_psi": max(valid) if valid else float("nan"),
        }
    return {
        "per_borough": out,
        "interpretation": (
            "PSI > 0.25 on lag features in ONE borough but not others => "
            "feature pipeline for that borough is upstream-broken or drifted."
        ),
    }


def detect_per_zone_divergence(
    baseline: pd.DataFrame, new: pd.DataFrame,
    baseline_preds: np.ndarray, new_preds: np.ndarray,
) -> dict:
    """Pattern 3: per-zone deviance delta + actual-mean shift.

    Combined signal: if a zone's actual_mean shifted >40% AND its deviance
    behaves abnormally, it's a candidate for upstream data tampering.
    """
    new = new.assign(yhat=new_preds, y=new["trip_count"].astype(float))
    baseline = baseline.assign(yhat=baseline_preds, y=baseline["trip_count"].astype(float))
    rows = []
    for z, gn in new.groupby("PULocationID"):
        gb = baseline[baseline["PULocationID"] == z]
        if len(gb) == 0:
            continue
        ndev = _poisson_deviance(gn["y"].values, gn["yhat"].values)
        bdev = _poisson_deviance(gb["y"].values, gb["yhat"].values)
        n_actual = float(gn["y"].mean())
        b_actual = float(gb["y"].mean())
        actual_drop = (n_actual - b_actual) / max(b_actual, 1e-9)
        dev_delta = (ndev - bdev) / max(bdev, 1e-9) if bdev > 1e-9 else float("nan")
        rows.append({
            "zone": int(z),
            "borough": int(gn["borough_id"].iloc[0]),
            "n": int(len(gn)),
            "actual_mean_baseline": b_actual,
            "actual_mean_new": n_actual,
            "actual_mean_drop": actual_drop,
            "poisson_dev_baseline": bdev,
            "poisson_dev_new": ndev,
            "dev_rel_delta": dev_delta,
        })
    rows.sort(key=lambda r: -(r["dev_rel_delta"] if not np.isnan(r["dev_rel_delta"]) else -1))
    scramble_suspects = [
        r for r in rows
        if r["actual_mean_drop"] < -0.40 or r["actual_mean_drop"] > 1.5
    ]
    return {
        "all_zones": rows,
        "worst_10_by_dev_delta": rows[:10],
        "scramble_suspects": scramble_suspects,
        "interpretation": (
            "Worst-deviance zones are the broadest concept-drift signal. "
            "Scramble suspects (|actual_mean shift| > 40%) flag zones whose "
            "data was structurally altered upstream."
        ),
    }


def detect_segment_concept_drift(
    baseline: pd.DataFrame, new: pd.DataFrame,
    baseline_preds: np.ndarray, new_preds: np.ndarray,
) -> dict:
    """Pattern 4: per (borough x is_weekend) deviance + MAPE delta."""
    baseline = baseline.assign(yhat=baseline_preds, y=baseline["trip_count"].astype(float))
    new = new.assign(yhat=new_preds, y=new["trip_count"].astype(float))
    out = {}
    for (b, w), gn in new.groupby(["borough_id", "is_weekend"]):
        gb = baseline[(baseline["borough_id"] == int(b))
                      & (baseline["is_weekend"] == int(w))]
        if len(gb) == 0:
            continue
        ndev = _poisson_deviance(gn["y"].values, gn["yhat"].values)
        bdev = _poisson_deviance(gb["y"].values, gb["yhat"].values)
        nmape = _mape(gn["y"].values, gn["yhat"].values)
        bmape = _mape(gb["y"].values, gb["yhat"].values)
        out[f"b{int(b)}_w{int(w)}"] = {
            "borough": int(b),
            "borough_name": BOROUGH_NAMES.get(int(b), "?"),
            "is_weekend": int(w),
            "n": int(len(gn)),
            "actual_mean_baseline": float(gb["y"].mean()),
            "actual_mean_new": float(gn["y"].mean()),
            "deviance_baseline": bdev,
            "deviance_new": ndev,
            "deviance_rel_delta": (ndev - bdev) / max(bdev, 1e-9),
            "mape_baseline": bmape,
            "mape_new": nmape,
            "mape_abs_delta": nmape - bmape,
        }
    worst = max(out.values(), key=lambda r: r["deviance_rel_delta"])
    return {
        "per_borough_weekend": out,
        "worst_segment": worst,
        "interpretation": (
            "Concept drift hotspot: the segment with the largest deviance_rel_delta "
            "is where the demand pattern shifted in a way the model cannot predict."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="week4/data/demand_enriched_week4.parquet")
    p.add_argument("--model", default="week2/model/lgbm_demand_model.txt")
    p.add_argument("--baseline-end", default="2026-01-16")
    p.add_argument("--new-start", default="2026-02-02")
    p.add_argument("--new-end", default="2026-03-01")
    p.add_argument("--out", default="week4/data/drift_patterns.json")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[2]
    parquet = repo / args.parquet
    model_path = repo / args.model
    out_path = repo / args.out

    print(f"[detect_drift] loading model: {model_path}")
    booster = lgb.Booster(model_file=str(model_path))
    features = booster.feature_name()

    print(f"[detect_drift] loading parquet: {parquet}")
    df = pd.read_parquet(parquet)

    baseline_df = df[df["time_bucket"] < pd.Timestamp(args.baseline_end)].copy()
    new_df = df[
        (df["time_bucket"] >= pd.Timestamp(args.new_start))
        & (df["time_bucket"] < pd.Timestamp(args.new_end))
    ].copy()

    print(f"[detect_drift] scoring   baseline={len(baseline_df):,}  new={len(new_df):,}")
    baseline_preds = booster.predict(baseline_df[features])
    new_preds = booster.predict(new_df[features])

    print("[detect_drift] running 4-pattern detector\n")
    patterns = {
        "pattern_1_temporal_peak_shift": detect_temporal_peak_shift(baseline_df, new_df),
        "pattern_2_borough_feature_drift": detect_borough_feature_drift(baseline_df, new_df),
        "pattern_3_per_zone_divergence": detect_per_zone_divergence(
            baseline_df, new_df, baseline_preds, new_preds,
        ),
        "pattern_4_segment_concept_drift": detect_segment_concept_drift(
            baseline_df, new_df, baseline_preds, new_preds,
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(patterns, indent=2, default=str))

    print("=" * 92)
    print("PATTERN 1 - TEMPORAL PEAK SHIFT (per-hour KS on trip_count)")
    print("=" * 92)
    for h, stats in patterns["pattern_1_temporal_peak_shift"]["worst_5_hours_by_ks"]:
        print(f"  hour={h:>2}  ks_stat={stats['ks_statistic']:.4f}  "
              f"baseline_mean={stats['baseline_mean']:.2f} -> new_mean={stats['new_mean']:.2f}")

    print("\n" + "=" * 92)
    print("PATTERN 2 - BOROUGH LAG-FEATURE DRIFT (PSI per borough)")
    print("=" * 92)
    for b, stats in patterns["pattern_2_borough_feature_drift"]["per_borough"].items():
        psi_str = "  ".join(f"{f}={v:.3f}" for f, v in stats["psi"].items())
        print(f"  borough={b} ({stats['borough_name']:>13}): {psi_str}    max_psi={stats['max_psi']:.3f}")

    print("\n" + "=" * 92)
    print("PATTERN 3 - PER-ZONE DIVERGENCE (worst 10 + scramble suspects)")
    print("=" * 92)
    for r in patterns["pattern_3_per_zone_divergence"]["worst_10_by_dev_delta"]:
        print(f"  zone={r['zone']:>4}  borough={r['borough']}  "
              f"dev_delta={r['dev_rel_delta']:+.3f}  "
              f"actual_drop={r['actual_mean_drop']:+.1%}")
    sus = patterns["pattern_3_per_zone_divergence"]["scramble_suspects"]
    print(f"  Scramble suspects (|actual_mean shift| > 40%): {len(sus)} zones")
    for r in sus[:10]:
        print(f"    zone={r['zone']:>4}  borough={r['borough']}  "
              f"actual_mean {r['actual_mean_baseline']:.2f} -> {r['actual_mean_new']:.2f}  "
              f"({r['actual_mean_drop']:+.1%})")

    print("\n" + "=" * 92)
    print("PATTERN 4 - SEGMENT CONCEPT DRIFT (borough x is_weekend)")
    print("=" * 92)
    for seg, r in patterns["pattern_4_segment_concept_drift"]["per_borough_weekend"].items():
        print(f"  {seg} ({r['borough_name']:>13}, weekend={r['is_weekend']}):  "
              f"n={r['n']:>7,d}  dev_delta={r['deviance_rel_delta']:+.3f}  "
              f"mape_delta={r['mape_abs_delta']:+.3f}")
    w = patterns["pattern_4_segment_concept_drift"]["worst_segment"]
    print(f"  WORST SEGMENT: {w['borough_name']} weekend={w['is_weekend']}  "
          f"dev_delta={w['deviance_rel_delta']:+.3f}")

    print(f"\n[detect_drift] wrote {out_path.relative_to(repo)}")


if __name__ == "__main__":
    main()
