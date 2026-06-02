"""Run the 8 monitoring metrics and produce an alerts report.

Loads the upstream parquet, slices it into a baseline window and a new
window, scores both with the LightGBM model from week2/, runs all 8
metrics from metric_template.MetricComputer, applies alert thresholds,
and writes a JSON report.

CI usage (see .github/workflows/monitor-drift.yml):
  python -m scripts.compute_metrics \
    --parquet week4/data/demand_enriched_week4.parquet \
    --model   week2/model/lgbm_demand_model.txt \
    --baseline-end 2026-01-16 \
    --new-start    2026-02-02 \
    --new-end      2026-03-01 \
    --out week4/data/metrics_report.json

Exit code is 0 when the metrics ran cleanly (whether or not they fired
alerts) - the "block deployment / open issue" decision is recorded in the
JSON output, not the workflow exit code. Same cosmetic-green pattern as
Week 3's validate-data workflow.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE))
    from metric_template import MetricComputer  # type: ignore
else:
    from .metric_template import MetricComputer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compute_metrics")

# ----------------------------------------------------------------------
# Alert thresholds (locked in design session; see Week 4 writeup).
# Two tiers: WARN (informational) and CRITICAL (gate / page).
# ----------------------------------------------------------------------
THRESHOLDS = {
    "metric_1_accuracy": {
        "deviance_rel_delta_warn": 0.10,
        "deviance_rel_delta_critical": 0.20,
    },
    "metric_2_accuracy_by_zone": {
        "deviance_rel_delta_critical": 0.20,
        "mape_abs_delta_critical": 0.10,
    },
    "metric_3_null_rates": {"warn": 0.005, "critical": 0.01},
    "metric_4_ks_test": {"pvalue_warn": 0.05, "pvalue_critical": 0.01},
    "metric_5_psi": {"warn": 0.10, "critical": 0.25},
    "metric_6_prediction_distribution": {"ks_pvalue_critical": 0.01},
    "metric_7_data_freshness": {"age_hours_warn": 1.0, "age_hours_critical": 2.0},
    "metric_8_duplicate_rate": {"warn": 0.001, "critical": 0.005},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", default="week4/data/demand_enriched_week4.parquet")
    p.add_argument("--model", default="week2/model/lgbm_demand_model.txt")
    p.add_argument("--baseline-end", default="2026-01-16",
                   help="exclusive end of baseline window")
    p.add_argument("--new-start", default="2026-02-02")
    p.add_argument("--new-end", default="2026-03-01",
                   help="exclusive end of new window")
    p.add_argument("--out", default="week4/data/metrics_report.json")
    p.add_argument("--reference-time", default=None,
                   help="reference 'now' for data-freshness; default=new-end")
    return p.parse_args()


def evaluate_alerts(metrics: dict) -> tuple[list, list]:
    """Apply THRESHOLDS, return (criticals, warnings) lists of alert dicts."""
    crit: list = []
    warn: list = []

    m = metrics.get("metric_1_accuracy", {})
    if "deviance_rel_delta" in m:
        d = m["deviance_rel_delta"]
        t = THRESHOLDS["metric_1_accuracy"]
        if d >= t["deviance_rel_delta_critical"]:
            crit.append({"metric": "metric_1_accuracy", "field": "deviance_rel_delta",
                         "value": d, "threshold": t["deviance_rel_delta_critical"]})
        elif d >= t["deviance_rel_delta_warn"]:
            warn.append({"metric": "metric_1_accuracy", "field": "deviance_rel_delta",
                         "value": d, "threshold": t["deviance_rel_delta_warn"]})

    m2 = metrics.get("metric_2_accuracy_by_zone", {})
    t2 = THRESHOLDS["metric_2_accuracy_by_zone"]
    for grouping, segments in m2.items():
        for seg_id, stats in segments.items():
            dev_d = stats.get("deviance_rel_delta")
            map_d = stats.get("mape_abs_delta")
            if dev_d is not None and dev_d >= t2["deviance_rel_delta_critical"]:
                crit.append({"metric": "metric_2_accuracy_by_zone",
                             "segment": f"{grouping}/{seg_id}",
                             "field": "deviance_rel_delta",
                             "value": dev_d,
                             "threshold": t2["deviance_rel_delta_critical"]})
            if map_d is not None and map_d > t2["mape_abs_delta_critical"]:
                crit.append({"metric": "metric_2_accuracy_by_zone",
                             "segment": f"{grouping}/{seg_id}",
                             "field": "mape_abs_delta",
                             "value": map_d,
                             "threshold": t2["mape_abs_delta_critical"]})

    t3 = THRESHOLDS["metric_3_null_rates"]
    for col, rate in metrics.get("metric_3_null_rates", {}).items():
        if rate >= t3["critical"]:
            crit.append({"metric": "metric_3_null_rates", "field": col,
                         "value": rate, "threshold": t3["critical"]})
        elif rate >= t3["warn"]:
            warn.append({"metric": "metric_3_null_rates", "field": col,
                         "value": rate, "threshold": t3["warn"]})

    t4 = THRESHOLDS["metric_4_ks_test"]
    ks = metrics.get("metric_4_ks_test", {})
    g = ks.get("global", {})
    if "pvalue" in g:
        if g["pvalue"] < t4["pvalue_critical"]:
            crit.append({"metric": "metric_4_ks_test", "field": "global.pvalue",
                         "value": g["pvalue"], "threshold": t4["pvalue_critical"]})
        elif g["pvalue"] < t4["pvalue_warn"]:
            warn.append({"metric": "metric_4_ks_test", "field": "global.pvalue",
                         "value": g["pvalue"], "threshold": t4["pvalue_warn"]})
    for h, stats in ks.get("per_hour", {}).items():
        if stats["pvalue"] < t4["pvalue_critical"]:
            crit.append({"metric": "metric_4_ks_test", "field": f"hour_{h}.pvalue",
                         "value": stats["pvalue"], "threshold": t4["pvalue_critical"]})

    t5 = THRESHOLDS["metric_5_psi"]
    psi = metrics.get("metric_5_psi", {})
    for col, val in psi.get("global", {}).items():
        if val >= t5["critical"]:
            crit.append({"metric": "metric_5_psi", "field": f"global.{col}",
                         "value": val, "threshold": t5["critical"]})
        elif val >= t5["warn"]:
            warn.append({"metric": "metric_5_psi", "field": f"global.{col}",
                         "value": val, "threshold": t5["warn"]})
    for borough, cols in psi.get("per_borough", {}).items():
        for col, val in cols.items():
            if val >= t5["critical"]:
                crit.append({"metric": "metric_5_psi",
                             "field": f"borough_{borough}.{col}",
                             "value": val, "threshold": t5["critical"]})

    m6 = metrics.get("metric_6_prediction_distribution", {})
    if m6.get("collapsed"):
        crit.append({"metric": "metric_6_prediction_distribution",
                     "field": "collapsed", "value": True})
    t6 = THRESHOLDS["metric_6_prediction_distribution"]
    if "ks_pvalue" in m6 and m6["ks_pvalue"] < t6["ks_pvalue_critical"]:
        crit.append({"metric": "metric_6_prediction_distribution",
                     "field": "ks_pvalue", "value": m6["ks_pvalue"],
                     "threshold": t6["ks_pvalue_critical"]})

    m7 = metrics.get("metric_7_data_freshness", {})
    age = m7.get("age_hours")
    if age is not None:
        t7 = THRESHOLDS["metric_7_data_freshness"]
        if age > t7["age_hours_critical"]:
            crit.append({"metric": "metric_7_data_freshness", "field": "age_hours",
                         "value": age, "threshold": t7["age_hours_critical"]})
        elif age > t7["age_hours_warn"]:
            warn.append({"metric": "metric_7_data_freshness", "field": "age_hours",
                         "value": age, "threshold": t7["age_hours_warn"]})

    m8 = metrics.get("metric_8_duplicate_rate", {})
    t8 = THRESHOLDS["metric_8_duplicate_rate"]
    rate = m8.get("rate", 0)
    if rate >= t8["critical"]:
        crit.append({"metric": "metric_8_duplicate_rate", "field": "rate",
                     "value": rate, "threshold": t8["critical"]})
    elif rate >= t8["warn"]:
        warn.append({"metric": "metric_8_duplicate_rate", "field": "rate",
                     "value": rate, "threshold": t8["warn"]})

    return crit, warn


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    parquet = (repo / args.parquet) if not Path(args.parquet).is_absolute() else Path(args.parquet)
    model_path = (repo / args.model) if not Path(args.model).is_absolute() else Path(args.model)
    out_path = (repo / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    log.info("loading model: %s", model_path)
    booster = lgb.Booster(model_file=str(model_path))
    features = booster.feature_name()

    log.info("loading parquet: %s", parquet)
    df = pd.read_parquet(parquet)

    baseline_end = pd.Timestamp(args.baseline_end)
    new_start = pd.Timestamp(args.new_start)
    new_end = pd.Timestamp(args.new_end)
    ref_time = pd.Timestamp(args.reference_time) if args.reference_time else new_end

    baseline_df = df[df["time_bucket"] < baseline_end].copy()
    new_df = df[(df["time_bucket"] >= new_start) & (df["time_bucket"] < new_end)].copy()
    log.info("baseline=%d rows  new=%d rows", len(baseline_df), len(new_df))

    log.info("scoring baseline + new with model")
    baseline_df["yhat"] = booster.predict(baseline_df[features])
    new_predictions = booster.predict(new_df[features])
    new_actuals = new_df["trip_count"].astype(float).values

    log.info("computing 8 metrics")
    mc = MetricComputer(baseline_df)
    metrics = mc.compute_all_metrics(new_df, predictions=new_predictions, actuals=new_actuals)
    metrics["metric_7_data_freshness"] = mc.metric_7_data_freshness(
        new_df, reference_time=ref_time,
    )

    criticals, warnings_ = evaluate_alerts(metrics)

    report = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_window": {"end": args.baseline_end, "rows": int(len(baseline_df))},
        "new_window": {
            "start": args.new_start, "end": args.new_end, "rows": int(len(new_df)),
        },
        "model_file": str(model_path.relative_to(repo)),
        "metrics": metrics,
        "alerts": {
            "critical_count": len(criticals),
            "warning_count": len(warnings_),
            "critical": criticals,
            "warning": warnings_,
        },
        "decision": (
            "BLOCK_DEPLOY_AND_PAGE" if criticals
            else "MONITOR" if warnings_
            else "OK"
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log.info("wrote %s  (%d critical, %d warning)",
             out_path, len(criticals), len(warnings_))

    print("=" * 88)
    print(f"COMPUTE_METRICS  decision={report['decision']}  "
          f"critical={len(criticals)}  warn={len(warnings_)}")
    print("=" * 88)
    if criticals:
        print("\nCRITICAL ALERTS:")
        for a in criticals[:20]:
            print(f"  - {a}")
        if len(criticals) > 20:
            print(f"  ... and {len(criticals)-20} more (see report JSON)")
    if warnings_:
        print(f"\n{len(warnings_)} warnings (see report JSON for full list)")


if __name__ == "__main__":
    main()
