"""Week 4 - Level 1 (feature-level) + Level 3 (segment-level) drift exploration.

Splits week4 parquet into:
  baseline = time_bucket < 2026-02-02  (3+ years of pre-drift history)
  drift    = 2026-02-02 <= time_bucket <= 2026-02-28  (27 days of suspect data)

Then dumps:
  Feature-level: KS / PSI / Jensen-Shannon per numeric feature, ranked
  Segment-level: per-zone trip_count means, per-borough breakdowns,
                 per-hour-of-day patterns, per-day-of-week patterns
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

PARQUET = "week4/data/demand_enriched_week4.parquet"
DRIFT_START = pd.Timestamp("2026-02-02")
DRIFT_END = pd.Timestamp("2026-03-01")  # exclusive


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    e = pd.to_numeric(expected, errors="coerce").dropna().values
    a = pd.to_numeric(actual, errors="coerce").dropna().values
    if len(e) == 0 or len(a) == 0:
        return float("nan")
    edges = np.linspace(min(e.min(), a.min()), max(e.max(), a.max()) + 1e-9, bins + 1)
    eh, _ = np.histogram(e, bins=edges)
    ah, _ = np.histogram(a, bins=edges)
    e_pct = np.maximum(eh / eh.sum(), 1e-6)
    a_pct = np.maximum(ah / ah.sum(), 1e-6)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def js_div(expected: pd.Series, actual: pd.Series, bins: int = 20) -> float:
    e = pd.to_numeric(expected, errors="coerce").dropna().values
    a = pd.to_numeric(actual, errors="coerce").dropna().values
    if len(e) == 0 or len(a) == 0:
        return float("nan")
    edges = np.linspace(min(e.min(), a.min()), max(e.max(), a.max()) + 1e-9, bins + 1)
    eh, _ = np.histogram(e, bins=edges, density=True)
    ah, _ = np.histogram(a, bins=edges, density=True)
    # normalize to probabilities, jensenshannon returns the distance (sqrt of divergence) by default
    eh = eh / max(eh.sum(), 1e-12)
    ah = ah / max(ah.sum(), 1e-12)
    return float(jensenshannon(eh, ah, base=2))


def fmt(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


print("=" * 90)
print("LOAD + SPLIT")
print("=" * 90)
df = pd.read_parquet(PARQUET)
df["time_bucket"] = pd.to_datetime(df["time_bucket"])
baseline = df[df["time_bucket"] < DRIFT_START]
drift = df[(df["time_bucket"] >= DRIFT_START) & (df["time_bucket"] < DRIFT_END)]
print(f"baseline: {len(baseline):>10,} rows ({baseline['time_bucket'].min()} -> {baseline['time_bucket'].max()})")
print(f"drift:    {len(drift):>10,} rows ({drift['time_bucket'].min()} -> {drift['time_bucket'].max()})")
print(f"zones in baseline: {baseline['PULocationID'].nunique()}")
print(f"zones in drift:    {drift['PULocationID'].nunique()}")

# ------------------------------------------------------------------
# LEVEL 1: FEATURE-LEVEL DRIFT
# ------------------------------------------------------------------
print()
print("=" * 90)
print("LEVEL 1: FEATURE-LEVEL DRIFT (per-numeric-column KS / PSI / JS-divergence)")
print("=" * 90)

# focus on the meaningful features (skip pure-calendar fields that drift mechanically
# on any short window vs 3-year baseline). Include them at the end for completeness.
meaningful_features = [
    "trip_count",                                   # target
    "lag_15min", "lag_1h", "lag_2h", "lag_1day", "lag_1week",
    "roll_mean_1h", "roll_mean_2h", "roll_mean_1day",
    "zone_slot_baseline",
    "is_holiday", "cbd_pricing_active", "is_weekend", "is_airport_zone",
    "borough_id", "service_zone_id",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
]
calendar_features = ["year", "month", "dayofyear", "weekofyear", "slot_of_day", "hour", "minute", "dayofweek"]

print(f"  {'feature':<25} {'b.mean':>9} {'d.mean':>9} {'b.std':>8} {'d.std':>8} {'KS':>7} {'PSI':>7} {'JS':>7}  interpretation")
print(f"  {'-' * 25:<25} {'-' * 9:>9} {'-' * 9:>9} {'-' * 8:>8} {'-' * 8:>8} {'-' * 7:>7} {'-' * 7:>7} {'-' * 7:>7}")
rows = []
for c in meaningful_features + calendar_features:
    if c not in df.columns:
        continue
    if not pd.api.types.is_numeric_dtype(df[c]):
        continue
    b = pd.to_numeric(baseline[c], errors="coerce").dropna()
    d = pd.to_numeric(drift[c], errors="coerce").dropna()
    if len(b) < 50 or len(d) < 50:
        continue
    ks_stat, ks_p = ks_2samp(b, d)
    psi_v = psi(b, d)
    js_v = js_div(b, d)
    rows.append((c, float(b.mean()), float(d.mean()), float(b.std()), float(d.std()),
                 float(ks_stat), float(psi_v), float(js_v), c in calendar_features))

# sort by PSI desc (most-drifted first), but print meaningful before calendar
meaningful_rows = sorted([r for r in rows if not r[8]], key=lambda r: -(r[6] if not np.isnan(r[6]) else 0))
calendar_rows = sorted([r for r in rows if r[8]], key=lambda r: -(r[6] if not np.isnan(r[6]) else 0))

for r in meaningful_rows + [(None,)] + calendar_rows:
    if r == (None,):
        print(f"  ----- calendar features below (expected to drift on a 27-day window vs 3-year baseline) -----")
        continue
    c, bm, dm, bs, ds, ks_s, psi_v, js_v, _ = r
    if np.isnan(psi_v):
        interp = "skip"
    elif psi_v < 0.1:
        interp = "stable"
    elif psi_v < 0.25:
        interp = "moderate drift"
    else:
        interp = "SIGNIFICANT DRIFT"
    print(f"  {c:<25} {bm:>9.3f} {dm:>9.3f} {bs:>8.3f} {ds:>8.3f} {ks_s:>7.3f} {psi_v:>7.3f} {js_v:>7.3f}  {interp}")

# ------------------------------------------------------------------
# LEVEL 3a: PER-ZONE TRIP_COUNT MEANS (biggest drift first)
# ------------------------------------------------------------------
print()
print("=" * 90)
print("LEVEL 3a: PER-ZONE TRIP_COUNT MEAN DRIFT (top 20 zones by absolute delta)")
print("=" * 90)
b_zone = baseline.groupby("PULocationID")["trip_count"].agg(["mean", "count"]).rename(columns={"mean": "b_mean", "count": "b_n"})
d_zone = drift.groupby("PULocationID")["trip_count"].agg(["mean", "count"]).rename(columns={"mean": "d_mean", "count": "d_n"})
zone_drift = b_zone.join(d_zone, how="inner")
zone_drift["delta"] = zone_drift["d_mean"] - zone_drift["b_mean"]
zone_drift["pct_change"] = (zone_drift["delta"] / zone_drift["b_mean"]) * 100
zone_drift = zone_drift.sort_values("delta", key=abs, ascending=False)
print(f"  {'zone':>6} {'b_mean':>10} {'d_mean':>10} {'delta':>10} {'pct%':>10} {'b_rows':>10} {'d_rows':>10}")
for zone, row in zone_drift.head(20).iterrows():
    print(f"  {zone:>6} {row['b_mean']:>10.3f} {row['d_mean']:>10.3f} {row['delta']:>+10.3f} {row['pct_change']:>+9.1f}% {int(row['b_n']):>10,} {int(row['d_n']):>10,}")

# ------------------------------------------------------------------
# LEVEL 3b: PER-BOROUGH TRIP_COUNT
# ------------------------------------------------------------------
print()
print("=" * 90)
print("LEVEL 3b: PER-BOROUGH TRIP_COUNT MEAN")
print("=" * 90)
borough_names = {0: "Manhattan", 1: "Queens", 2: "Brooklyn", 3: "Bronx", 4: "Staten Island", 5: "EWR/other"}
for bid in sorted(df["borough_id"].dropna().unique()):
    bm = baseline[baseline["borough_id"] == bid]["trip_count"].mean()
    dm = drift[drift["borough_id"] == bid]["trip_count"].mean()
    bs = baseline[baseline["borough_id"] == bid]["trip_count"].std()
    ds = drift[drift["borough_id"] == bid]["trip_count"].std()
    n_b = len(baseline[baseline["borough_id"] == bid])
    n_d = len(drift[drift["borough_id"] == bid])
    delta_pct = (dm - bm) / bm * 100 if bm > 0 else float("nan")
    name = borough_names.get(int(bid), f"borough_{int(bid)}")
    print(f"  borough_id={int(bid)} ({name:<18}) b={bm:>7.2f}±{bs:>5.2f}  d={dm:>7.2f}±{ds:>5.2f}  delta={dm - bm:>+6.2f} ({delta_pct:>+5.1f}%)  n_b={n_b:>9,} n_d={n_d:>8,}")

# ------------------------------------------------------------------
# LEVEL 3c: PER-HOUR-OF-DAY DEMAND (sum across all zones)
# ------------------------------------------------------------------
print()
print("=" * 90)
print("LEVEL 3c: PER-HOUR-OF-DAY TRIP_COUNT MEAN (each hour, both windows)")
print("=" * 90)
b_hour = baseline.groupby("hour")["trip_count"].mean()
d_hour = drift.groupby("hour")["trip_count"].mean()
print(f"  {'hour':>4} {'b_mean':>10} {'d_mean':>10} {'delta':>10} {'pct%':>10}")
for h in range(24):
    bm = b_hour.get(h, float("nan"))
    dm = d_hour.get(h, float("nan"))
    delta = dm - bm
    pct = (delta / bm) * 100 if bm and not np.isnan(bm) and bm > 0 else float("nan")
    marker = "  *" if (not np.isnan(pct) and abs(pct) > 20) else ""
    print(f"  {h:>4} {bm:>10.3f} {dm:>10.3f} {delta:>+10.3f} {pct:>+9.1f}%{marker}")
print("  (* marks hours with >20% absolute change)")

# ------------------------------------------------------------------
# LEVEL 3d: PER-DAY-OF-WEEK PATTERN
# ------------------------------------------------------------------
print()
print("=" * 90)
print("LEVEL 3d: PER-DAY-OF-WEEK TRIP_COUNT MEAN")
print("=" * 90)
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
b_dow = baseline.groupby("dayofweek")["trip_count"].mean()
d_dow = drift.groupby("dayofweek")["trip_count"].mean()
print(f"  {'dow':>4} {'name':<5} {'b_mean':>10} {'d_mean':>10} {'delta':>10} {'pct%':>10}")
for d in range(7):
    bm = b_dow.get(d, float("nan"))
    dm = d_dow.get(d, float("nan"))
    delta = dm - bm
    pct = (delta / bm) * 100 if bm and not np.isnan(bm) and bm > 0 else float("nan")
    marker = "  *" if (not np.isnan(pct) and abs(pct) > 15) else ""
    print(f"  {d:>4} {dow_names[d]:<5} {bm:>10.3f} {dm:>10.3f} {delta:>+10.3f} {pct:>+9.1f}%{marker}")
print("  (* marks days with >15% absolute change)")

# ------------------------------------------------------------------
# LEVEL 3e: WEEKDAY vs WEEKEND interaction per borough
# ------------------------------------------------------------------
print()
print("=" * 90)
print("LEVEL 3e: WEEKDAY vs WEEKEND by BOROUGH (concept-drift candidate)")
print("=" * 90)
print(f"  {'borough':<20} {'is_weekend':>11} {'b_mean':>10} {'d_mean':>10} {'delta':>10} {'pct%':>10}")
for bid in sorted(df["borough_id"].dropna().unique()):
    name = borough_names.get(int(bid), f"borough_{int(bid)}")
    for we in [0, 1]:
        b_sub = baseline[(baseline["borough_id"] == bid) & (baseline["is_weekend"] == we)]
        d_sub = drift[(drift["borough_id"] == bid) & (drift["is_weekend"] == we)]
        if len(b_sub) < 30 or len(d_sub) < 30:
            continue
        bm = b_sub["trip_count"].mean()
        dm = d_sub["trip_count"].mean()
        delta = dm - bm
        pct = (delta / bm) * 100 if bm > 0 else float("nan")
        marker = "  *" if abs(pct) > 20 else ""
        print(f"  {name:<20} {we:>11} {bm:>10.3f} {dm:>10.3f} {delta:>+10.3f} {pct:>+9.1f}%{marker}")
print("  (* marks borough+weekend cells with >20% absolute change)")

print()
print("=" * 90)
print("DONE")
print("=" * 90)
