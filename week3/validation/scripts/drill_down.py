"""
Three drill-downs run in one pass:
  A: re-run broad comparison with the proper historical-vs-new split
     (historical = corrupted[time_bucket < 2026-01-16],
      new        = corrupted[2026-01-16 <= time_bucket < 2026-02-02])
  B: trip_count invalid values - which sentinels, which zones, which dates
  C: is_holiday flag - which 2026 dates are flagged vs the real US Federal holiday calendar
"""
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

CORR = "week3/data/demand_enriched_corrupted.parquet"
BATCH_START = pd.Timestamp("2026-01-16")
BATCH_END   = pd.Timestamp("2026-02-02")  # exclusive

# 2026 US Federal holidays (the ones reasonable to expect in NYC taxi data)
HOLIDAYS_2026 = {
    pd.Timestamp("2026-01-01").date(): "New Year's Day",
    pd.Timestamp("2026-01-19").date(): "MLK Day",
    pd.Timestamp("2026-02-16").date(): "Presidents Day",
    pd.Timestamp("2026-05-25").date(): "Memorial Day",
    pd.Timestamp("2026-06-19").date(): "Juneteenth",
    pd.Timestamp("2026-07-03").date(): "Independence Day (observed)",
    pd.Timestamp("2026-07-04").date(): "Independence Day",
    pd.Timestamp("2026-09-07").date(): "Labor Day",
    pd.Timestamp("2026-10-12").date(): "Columbus Day",
    pd.Timestamp("2026-11-11").date(): "Veterans Day",
    pd.Timestamp("2026-11-26").date(): "Thanksgiving",
    pd.Timestamp("2026-12-25").date(): "Christmas Day",
}

def psi(expected, actual, bins=10):
    e = pd.to_numeric(expected, errors="coerce").dropna().values
    a = pd.to_numeric(actual,   errors="coerce").dropna().values
    if len(e) == 0 or len(a) == 0: return float("nan")
    edges = np.linspace(min(e.min(), a.min()), max(e.max(), a.max()) + 1e-9, bins + 1)
    eh, _ = np.histogram(e, bins=edges); ah, _ = np.histogram(a, bins=edges)
    e_pct = np.maximum(eh / eh.sum(), 1e-6); a_pct = np.maximum(ah / ah.sum(), 1e-6)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

print("=" * 78)
print("LOAD + SPLIT")
print("=" * 78)
df = pd.read_parquet(CORR)
df["time_bucket"] = pd.to_datetime(df["time_bucket"])
hist = df[df["time_bucket"] < BATCH_START]
new  = df[(df["time_bucket"] >= BATCH_START) & (df["time_bucket"] < BATCH_END)]
print(f"historical: {len(hist):>9,} rows  ({hist['time_bucket'].min()}  to  {hist['time_bucket'].max()})")
print(f"new batch:  {len(new):>9,} rows  ({new['time_bucket'].min()}  to  {new['time_bucket'].max()})")

# ------------------------------------------------------------------
# A: broad comparison on the proper split
# ------------------------------------------------------------------
print()
print("=" * 78)
print("A: NULL RATES (historical vs new, delta sorted)")
print("=" * 78)
print(f"  {'column':<28} {'hist':>10} {'new':>10} {'delta':>10}")
rows = []
for c in hist.columns:
    h = hist[c].isna().mean()
    n = new[c].isna().mean()
    rows.append((c, h, n, n - h))
rows.sort(key=lambda r: -abs(r[3]))
for c, h, n, d in rows:
    if abs(d) < 1e-6 and h < 1e-6 and n < 1e-6: continue
    print(f"  {c:<28} {h:>10.4%} {n:>10.4%} {d:>+10.4%}")

print()
print("=" * 78)
print("A: DISTRIBUTION DRIFT PER NUMERIC COLUMN (KS + PSI, sorted by PSI)")
print("=" * 78)
print(f"  {'column':<24} {'h.mean':>10} {'n.mean':>10} {'h.max':>10} {'n.max':>10} {'KS':>8} {'PSI':>8}  interp")
numeric_cols = [c for c in hist.columns if pd.api.types.is_numeric_dtype(hist[c]) and not pd.api.types.is_bool_dtype(hist[c])]
psi_rows = []
for c in numeric_cols:
    h = pd.to_numeric(hist[c], errors="coerce").dropna()
    n = pd.to_numeric(new[c],  errors="coerce").dropna()
    if len(h) < 50 or len(n) < 50: continue
    ks_stat, ks_p = ks_2samp(h, n)
    psi_v = psi(h, n)
    psi_rows.append((c, float(h.mean()), float(n.mean()), float(h.max()), float(n.max()), float(ks_stat), float(psi_v)))
psi_rows.sort(key=lambda r: -(r[6] if not np.isnan(r[6]) else 0))
for c, hm, nm, hmx, nmx, ks_s, psi_v in psi_rows:
    if np.isnan(psi_v): interp = "skip"
    elif psi_v < 0.1:   interp = "stable"
    elif psi_v < 0.25:  interp = "moderate drift"
    else:               interp = "SIGNIFICANT DRIFT"
    print(f"  {c:<24} {hm:>10.3f} {nm:>10.3f} {hmx:>10.3f} {nmx:>10.3f} {ks_s:>8.3f} {psi_v:>8.3f}  {interp}")

print()
print("=" * 78)
print("A: DUPLICATES (by PULocationID + time_bucket) - within new batch only")
print("=" * 78)
counts = new.groupby(["PULocationID", "time_bucket"]).size()
dups   = counts[counts > 1]
total_extra = int((dups - 1).sum())
print(f"  duplicate (zone, time) keys: {len(dups):,}")
print(f"  extra rows beyond first occurrence: {total_extra:,}")
if len(dups) > 0:
    aff_zones = sorted(dups.index.get_level_values("PULocationID").unique().tolist())
    print(f"  affected zones ({len(aff_zones)}): {aff_zones[:25]}{'...' if len(aff_zones)>25 else ''}")
    print(f"  time range of duplicates: {dups.index.get_level_values('time_bucket').min()}  to  {dups.index.get_level_values('time_bucket').max()}")
    # how many dups per affected zone
    dups_by_zone = dups.reset_index().groupby("PULocationID").size().sort_values(ascending=False)
    print(f"  duplicates by affected zone (top 10):")
    for z, ct in dups_by_zone.head(10).items():
        print(f"    zone {int(z)}: {int(ct):,} duplicate keys")

# ------------------------------------------------------------------
# B: trip_count drill-down (new batch only)
# ------------------------------------------------------------------
print()
print("=" * 78)
print("B: trip_count INVALID VALUES (new batch only)")
print("=" * 78)
print(f"  total rows in new batch: {len(new):,}")
print(f"  min: {new['trip_count'].min()}")
print(f"  max: {new['trip_count'].max()}")
print(f"  count of exactly -5:    {(new['trip_count'] == -5).sum():,}")
print(f"  count of exactly -1:    {(new['trip_count'] == -1).sum():,}")
print(f"  count of exactly 0:     {(new['trip_count'] == 0).sum():,}")
print(f"  count of exactly 9999:  {(new['trip_count'] == 9999).sum():,}")
print(f"  count of exactly 99999: {(new['trip_count'] == 99999).sum():,}")
print(f"  count negative (<0):    {(new['trip_count'] < 0).sum():,}")
hist_p99 = hist['trip_count'].quantile(0.99)
hist_max = hist['trip_count'].max()
print(f"  historical baseline p99: {hist_p99:.1f}, max: {hist_max:.0f}")
print(f"  count > historical max ({int(hist_max)}): {(new['trip_count'] > hist_max).sum():,}")

invalid_mask = (new['trip_count'] < 0) | (new['trip_count'] > hist_max)
invalid = new[invalid_mask]
print(f"  total 'invalid' rows (negative OR > historical max): {len(invalid):,}")
if len(invalid) > 0:
    z_counts = invalid['PULocationID'].value_counts()
    print(f"  invalid rows by zone (top 10): {z_counts.head(10).to_dict()}")
    print(f"  number of distinct zones affected: {z_counts.shape[0]}")
    d_counts = invalid['time_bucket'].dt.date.value_counts().sort_index()
    print(f"  invalid rows by date (top 10 dates):")
    for d, ct in d_counts.head(10).items():
        print(f"    {d}: {ct:,} invalid rows")
    print(f"  date range of invalid rows: {d_counts.index.min()}  to  {d_counts.index.max()}")
    # value distribution within invalid
    val_counts = invalid['trip_count'].value_counts().sort_index()
    print(f"  invalid value distribution (top 10 by frequency):")
    for v, ct in val_counts.sort_values(ascending=False).head(10).items():
        print(f"    value {int(v):>6}: {int(ct):,} rows")

# ------------------------------------------------------------------
# C: is_holiday drill-down vs real 2026 calendar
# ------------------------------------------------------------------
print()
print("=" * 78)
print("C: is_holiday FLAG vs REAL 2026 US FEDERAL HOLIDAYS")
print("=" * 78)
data_2026 = df[df["time_bucket"].dt.year == 2026].copy()
data_2026["date"] = data_2026["time_bucket"].dt.date
rate_by_date = data_2026.groupby("date")["is_holiday"].mean().sort_index()

print(f"  Real 2026 US Federal holidays (expected is_holiday=1):")
for d, name in sorted(HOLIDAYS_2026.items()):
    rate = rate_by_date.get(d, None)
    if rate is None:
        status = "no data"
    elif rate >= 0.99:
        status = "FLAGGED (correct)"
    elif rate > 0:
        status = f"partial ({rate:.2%})"
    else:
        status = "MISSING (should be flagged!)"
    print(f"    {d}: {name:<30} -> {status}")

print()
print(f"  Dates with is_holiday > 0 in 2026 corrupted data:")
for d, r in rate_by_date[rate_by_date > 0].items():
    real = HOLIDAYS_2026.get(d, None)
    mark = "OK" if real else "MISLABELED"
    print(f"    {d} ({r:.2%}): {real or 'NOT A REAL HOLIDAY'} [{mark}]")

# Count mislabeled rows
real_holiday_dates = set(HOLIDAYS_2026.keys())
mislabeled_mask = (data_2026["is_holiday"] == 1) & (~data_2026["date"].isin(real_holiday_dates))
missing_mask    = (data_2026["is_holiday"] == 0) & (data_2026["date"].isin(real_holiday_dates))
print()
print(f"  rows flagged is_holiday=1 on dates that are NOT real 2026 holidays: {int(mislabeled_mask.sum()):,}")
print(f"  rows flagged is_holiday=0 on dates that ARE real 2026 holidays:      {int(missing_mask.sum()):,}")
print(f"  total 2026 rows in corrupted: {len(data_2026):,}")
print(f"  mislabel rate: {mislabeled_mask.mean():.4%}")

# What's the date range of the mislabel pattern?
if mislabeled_mask.sum() > 0:
    mis = data_2026[mislabeled_mask]
    mis_dates = sorted(mis["date"].unique())
    print(f"  distinct dates with the false-holiday flag: {len(mis_dates)}")
    print(f"  first 20 such dates: {mis_dates[:20]}")
    print(f"  date range: {min(mis_dates)} to {max(mis_dates)}")

print()
print("=" * 78)
print("DONE")
print("=" * 78)
