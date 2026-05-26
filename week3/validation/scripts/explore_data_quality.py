"""Open-ended exploration: compare baseline (Jan 1-15 trusted) to corrupted (Jan 16 - Feb 1).

Dumps:
- Row counts
- Null rates per column (delta)
- Value ranges per numeric column (min, max, mean, std, p95, p99) - delta
- Duplicate counts by (PULocationID, time_bucket)
- Distribution comparison (KS + PSI) per numeric column
- Per-zone row counts and obvious anomalies
- is_holiday flag rate
- Schema diff (columns and dtypes)

Run from repo root: python week3/explore_data_quality.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

BASE_PATH = Path("week3/data/demand_enriched_baseline.parquet")
NEW_PATH  = Path("week3/data/demand_enriched_corrupted.parquet")

# ---------- helpers ----------

def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    e = pd.to_numeric(expected, errors="coerce").dropna().values
    a = pd.to_numeric(actual, errors="coerce").dropna().values
    if len(e) == 0 or len(a) == 0:
        return float("nan")
    edges = np.linspace(min(e.min(), a.min()), max(e.max(), a.max()) + 1e-9, bins + 1)
    e_hist, _ = np.histogram(e, bins=edges)
    a_hist, _ = np.histogram(a, bins=edges)
    e_pct = np.maximum(e_hist / e_hist.sum(), 1e-6)
    a_pct = np.maximum(a_hist / a_hist.sum(), 1e-6)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

def fmt(v):
    if v is None: return "-"
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return "-"
    if isinstance(v, float): return f"{v:.4f}"
    return str(v)

# ---------- load ----------

print("=" * 78)
print("LOADING")
print("=" * 78)
base = pd.read_parquet(BASE_PATH)
new = pd.read_parquet(NEW_PATH)
print(f"baseline:  {len(base):>9,} rows  x  {len(base.columns):>3} cols  ({BASE_PATH})")
print(f"corrupted: {len(new):>9,} rows  x  {len(new.columns):>3} cols  ({NEW_PATH})")

# ---------- schema diff ----------

print()
print("=" * 78)
print("SCHEMA DIFF (columns present / dtypes)")
print("=" * 78)
base_cols = set(base.columns); new_cols = set(new.columns)
only_in_base = sorted(base_cols - new_cols)
only_in_new  = sorted(new_cols - base_cols)
print(f"in baseline not in corrupted: {only_in_base or 'none'}")
print(f"in corrupted not in baseline: {only_in_new or 'none'}")
shared = sorted(base_cols & new_cols)
dtype_diffs = []
for c in shared:
    if str(base[c].dtype) != str(new[c].dtype):
        dtype_diffs.append((c, str(base[c].dtype), str(new[c].dtype)))
if dtype_diffs:
    print("dtype differences:")
    for c, b, n in dtype_diffs:
        print(f"  {c}: baseline={b}  corrupted={n}")
else:
    print("dtype differences: none")

# ---------- null rates ----------

print()
print("=" * 78)
print("NULL RATES PER COLUMN (delta = corrupted - baseline, sorted by abs delta)")
print("=" * 78)
print(f"  {'column':<28} {'baseline':>10} {'corrupted':>10} {'delta':>10}")
rows = []
for c in shared:
    b_null = base[c].isna().mean()
    n_null = new[c].isna().mean()
    rows.append((c, b_null, n_null, n_null - b_null))
rows.sort(key=lambda r: -abs(r[3]))
for c, b_null, n_null, delta in rows[:15]:
    if abs(delta) < 1e-6 and b_null < 1e-6 and n_null < 1e-6:
        continue
    print(f"  {c:<28} {b_null:>10.4%} {n_null:>10.4%} {delta:>+10.4%}")

# ---------- value ranges (numeric columns) ----------

print()
print("=" * 78)
print("NUMERIC VALUE RANGES (baseline -> corrupted)")
print("=" * 78)
numeric_cols = [c for c in shared if pd.api.types.is_numeric_dtype(base[c]) and not pd.api.types.is_bool_dtype(base[c])]
print(f"  {'column':<24} {'src':<10} {'min':>10} {'max':>10} {'mean':>10} {'std':>10} {'p95':>10} {'p99':>10}")
for c in numeric_cols:
    for name, df in [("baseline", base), ("corrupted", new)]:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) == 0:
            print(f"  {c:<24} {name:<10}  (all null)")
            continue
        print(f"  {c:<24} {name:<10} {fmt(float(s.min())):>10} {fmt(float(s.max())):>10} "
              f"{fmt(float(s.mean())):>10} {fmt(float(s.std())):>10} "
              f"{fmt(float(s.quantile(0.95))):>10} {fmt(float(s.quantile(0.99))):>10}")
    print()

# ---------- distribution drift (KS + PSI) ----------

print()
print("=" * 78)
print("DISTRIBUTION DRIFT PER NUMERIC COLUMN (KS test + PSI, sorted by PSI)")
print("=" * 78)
print(f"  {'column':<24} {'KS stat':>10} {'KS pval':>10} {'PSI':>10}   interpretation")
psi_rows = []
for c in numeric_cols:
    b = pd.to_numeric(base[c], errors="coerce").dropna()
    n = pd.to_numeric(new[c], errors="coerce").dropna()
    if len(b) < 50 or len(n) < 50:
        continue
    ks_stat, ks_p = ks_2samp(b, n)
    psi_v = psi(b, n)
    psi_rows.append((c, float(ks_stat), float(ks_p), psi_v))
psi_rows.sort(key=lambda r: -(r[3] if not np.isnan(r[3]) else 0))
for c, ks_stat, ks_p, psi_v in psi_rows:
    if np.isnan(psi_v):
        interp = "skip"
    elif psi_v < 0.1:
        interp = "stable"
    elif psi_v < 0.25:
        interp = "moderate drift"
    else:
        interp = "SIGNIFICANT DRIFT"
    print(f"  {c:<24} {fmt(ks_stat):>10} {fmt(ks_p):>10} {fmt(psi_v):>10}   {interp}")

# ---------- duplicate detection ----------

print()
print("=" * 78)
print("DUPLICATE DETECTION (by PULocationID + time_bucket)")
print("=" * 78)
def dup_summary(df, name):
    if "PULocationID" not in df.columns or "time_bucket" not in df.columns:
        print(f"  {name}: missing PULocationID or time_bucket; skipping")
        return
    counts = df.groupby(["PULocationID", "time_bucket"]).size()
    dup_keys = counts[counts > 1]
    total_dup_rows = int((dup_keys - 1).sum())
    print(f"  {name}: {len(dup_keys):>6,} duplicate (zone, time) keys, "
          f"{total_dup_rows:>6,} extra rows beyond first")
    if len(dup_keys) > 0:
        affected_zones = sorted(dup_keys.index.get_level_values("PULocationID").unique().tolist())
        time_min = dup_keys.index.get_level_values("time_bucket").min()
        time_max = dup_keys.index.get_level_values("time_bucket").max()
        print(f"  {name}: affected zones ({len(affected_zones)}): {affected_zones[:20]}{'...' if len(affected_zones)>20 else ''}")
        print(f"  {name}: time range of duplicates: {time_min} to {time_max}")
dup_summary(base, "baseline ")
dup_summary(new,  "corrupted")

# ---------- per-zone row counts (anomaly check) ----------

print()
print("=" * 78)
print("PER-ZONE ROW COUNT COMPARISON (zones with biggest count delta)")
print("=" * 78)
if "PULocationID" in shared:
    b_counts = base.groupby("PULocationID").size().rename("baseline")
    n_counts = new.groupby("PULocationID").size().rename("corrupted")
    counts = pd.concat([b_counts, n_counts], axis=1).fillna(0).astype(int)
    counts["delta"] = counts["corrupted"] - counts["baseline"]
    counts = counts.sort_values("delta", key=abs, ascending=False)
    print(f"  {'zone':>6} {'baseline':>10} {'corrupted':>10} {'delta':>10}")
    for zone, row in counts.head(15).iterrows():
        print(f"  {zone:>6} {int(row['baseline']):>10,} {int(row['corrupted']):>10,} {int(row['delta']):>+10,}")
    print(f"  ... ({len(counts)} total zones)")

# ---------- is_holiday rate ----------

print()
print("=" * 78)
print("is_holiday FLAG RATE (overall + by date)")
print("=" * 78)
for name, df in [("baseline", base), ("corrupted", new)]:
    if "is_holiday" in df.columns:
        rate = df["is_holiday"].mean()
        print(f"  {name}: is_holiday=1 rate overall: {rate:.4%}  ({int(df['is_holiday'].sum()):,} of {len(df):,} rows)")

if "is_holiday" in new.columns and "time_bucket" in new.columns:
    new2 = new.copy()
    new2["time_bucket"] = pd.to_datetime(new2["time_bucket"])
    by_date = new2.groupby(new2["time_bucket"].dt.date)["is_holiday"].mean()
    print()
    print(f"  corrupted: is_holiday rate by date (showing rates > 0):")
    nonzero = by_date[by_date > 0]
    for d, r in nonzero.items():
        print(f"    {d}: {r:.4%}")

# ---------- per-zone lag_1week sanity (correlation with own trip_count) ----------

print()
print("=" * 78)
print("LAG_1WEEK vs TRIP_COUNT CORRELATION PER ZONE (lower correlation = suspicious)")
print("=" * 78)
def lag_corr(df, name):
    if "lag_1week" not in df.columns or "trip_count" not in df.columns:
        return
    rows = []
    for zone, g in df.groupby("PULocationID"):
        if len(g) < 50: continue
        c = g[["lag_1week", "trip_count"]].corr().iloc[0, 1]
        rows.append((int(zone), float(c) if not np.isnan(c) else None))
    rows.sort(key=lambda r: r[1] if r[1] is not None else 0)
    print(f"  {name}: lowest 10 zone correlations (suspicious if much lower than baseline):")
    for z, c in rows[:10]:
        print(f"    zone {z}: corr = {fmt(c)}")
    print(f"  {name}: highest 5: ")
    for z, c in rows[-5:]:
        print(f"    zone {z}: corr = {fmt(c)}")
lag_corr(base, "baseline ")
lag_corr(new,  "corrupted")

print()
print("=" * 78)
print("DONE")
print("=" * 78)
