# Week 4 Architecture Diagram

Mermaid source for the drift monitoring + retraining workflow.

Render at https://mermaid.live and export as PNG or SVG for embedding in
the Week 4 report.

```mermaid
flowchart TD
    A[Upstream parquet<br/>course repo]
    B[monitor-drift.yml<br/>Triggers: hourly cron, push to main, manual workflow_dispatch]
    C[compute_metrics.py<br/>8 metrics, 2-tier thresholds]
    D[detect_drift.py<br/>4-pattern detector]
    E{decision per cycle}
    F[No action<br/>wait next cycle]
    G[Auto-file GitHub Issue<br/>label: drift-alert]

    H{Retraining trigger fires<br/>if any input is true}
    H1[2 consecutive BLOCK cycles]
    H2[Monthly cron<br/>1st of month, 02:00 UTC]
    H3[Manual workflow_dispatch]

    I[Train LightGBM<br/>Poisson objective<br/>full history + 90d reweighting<br/>holdout: last 7 days]
    J{Offline gate<br/>beats current on holdout<br/>RMSE AND deviance?}
    K[Abandon<br/>file Issue: retrain failed]
    L[Canary deploy<br/>5% traffic for 24h]
    M{Canary gate<br/>per-segment Tier-B metrics hold?}
    N[Auto-rollback to parent_version<br/>file Issue: rollback]
    O[Promote canary to 100%<br/>watchdog monitors Tier-B for 24h]
    P[(GCS bucket<br/>models/lgbm/<br/>v_YYYY-MM-DD_sha.txt<br/>+ JSON sidecar<br/>keep last 6 versions)]

    A --> B
    B --> C
    B --> D
    C --> E
    D --> E
    E -->|OK or MONITOR| F
    E -->|BLOCK_DEPLOY_AND_PAGE| G
    G --> H1
    H1 --> H
    H2 --> H
    H3 --> H
    H -->|trigger fires| I
    I --> J
    J -->|fail| K
    J -->|pass| L
    L --> M
    M -->|fail| N
    M -->|pass| O
    I -.->|register new version| P
    O -.->|set deployed_at| P
    N -.->|read parent_version| P
```

## How to render

1. Open https://mermaid.live in a browser.
2. Paste everything between the triple backticks above (the `flowchart TD ...`
   block) into the left editor pane.
3. The diagram appears in the right pane.
4. Click `Actions` (top right corner) and choose `PNG` or `SVG`.
5. Insert the exported image into the Week 4 report.

## What is in the diagram

- DATA SOURCE: upstream parquet from the course repo
- MONITOR: monitor-drift.yml with all three triggers labeled (hourly cron,
  push to main, manual workflow_dispatch)
- DRIFT EVAL: compute_metrics.py and detect_drift.py running in parallel
- DECISION: per-cycle gate that either takes no action or files a
  drift-alert GitHub Issue
- TRIGGER: retraining trigger evaluator with three possible inputs
  (2 consecutive BLOCK cycles, monthly cron, manual workflow_dispatch)
- TRAIN: LightGBM Poisson training on full history with 90-day
  reweighting, holding out the last 7 days
- VALIDATE: offline gate (must beat current model on holdout RMSE and
  Poisson deviance) followed by 5% canary for 24 hours
- DEPLOY: promote canary to 100% with a 24-hour watchdog, or auto-rollback
  to parent_version on failure
- STORAGE: GCS bucket with versioned model files and JSON metadata
  sidecars; last 6 versions kept

The dashed arrows are reads and writes against the GCS bucket: train
writes a new version, promote sets the deployed_at field, rollback reads
the parent_version key.
