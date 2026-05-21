# Week 2 — Architecture Diagram

## Required by README

> Show: GitHub → Artifact Registry → GKE → external IP

## Rendering instructions

The Mermaid source below renders cleanly via:

1. **https://mermaid.live** (recommended): paste the source between the ```` ```mermaid ```` fences, then **Actions → Export to PNG** (or SVG / PDF). Save as `week2/architecture.png` or `week2/architecture.pdf`.
2. **VS Code**: install the *Markdown Preview Mermaid Support* extension, then open this file and use the markdown preview.
3. **GitHub**: it renders inline in markdown previews automatically when you push the repo.

---

## Architecture (Mermaid source)

```mermaid
flowchart LR
    DEV["👤 Developer<br/>git push origin main"]

    subgraph GH ["GitHub — DominicTanzillo/Ops-AI-Student"]
        REPO["📦 Repository"]
        SECRET["🔐 Secret: GCP_SA_KEY"]
        CI["CI: ci.yml<br/>pytest + docker build<br/>(every push & PR)"]
        CD["CD: cd.yml<br/>build → push → kubectl set image<br/>(main branch only)"]
    end

    subgraph GCP ["GCP Project: ops-ai-dpt7  /  us-central1-a"]
        SA["🔑 Service Account<br/>github-actions@<br/>container.developer<br/>artifactregistry.writer/reader<br/>storage.objectViewer"]

        subgraph AR ["📦 Artifact Registry — docker-repo"]
            IMG["demand-api image<br/>tags: latest + ${github.sha}"]
        end

        subgraph GCS ["💾 Cloud Storage — gs://ops-ai-dpt7-data"]
            BUCKET["demand_enriched.parquet (74 MB)<br/>demand_api_model.joblib<br/>zone_hour_avg_fare.parquet<br/>taxi_zones.geojson"]
        end

        subgraph CLUSTER ["☸️ GKE Cluster — operationalizing-ai (2× n1-standard-2)"]
            subgraph DEP ["Deployment: demand-api (replicas: 2, RollingUpdate maxSurge=1 maxUnavailable=1)"]
                POD1["Pod #1<br/>init: gsutil cp from GCS<br/>main: FastAPI :8000<br/>readinessProbe initialDelay=30s<br/>livenessProbe initialDelay=60s"]
                POD2["Pod #2<br/>init: gsutil cp from GCS<br/>main: FastAPI :8000"]
            end
            SVC["Service: LoadBalancer<br/>port 80 → targetPort 8000"]
        end

        EXTIP["🌐 External IP<br/>(e.g., 35.254.153.3)"]
    end

    CLIENT["🧑 Client / curl<br/>/health<br/>/api/heatmap<br/>/api/forecast<br/>/api/recommendations"]

    DEV -->|push| REPO
    REPO -->|trigger| CI
    REPO -->|trigger main| CD
    CD -.->|auth via| SECRET
    SECRET -.->|impersonates| SA
    CD -->|docker push| IMG
    CD -->|kubectl set image| DEP
    IMG -->|imagePullSecret pull| POD1
    IMG -->|imagePullSecret pull| POD2
    BUCKET -->|init container| POD1
    BUCKET -->|init container| POD2
    POD1 --> SVC
    POD2 --> SVC
    SVC --> EXTIP
    CLIENT <-->|HTTP :80| EXTIP

    classDef gcp fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#000
    classDef gh fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#000
    classDef client fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px,color:#000
    class AR,GCS,CLUSTER,SA,EXTIP gcp
    class REPO,SECRET,CI,CD gh
    class DEV,CLIENT client
```

---

## ASCII fallback (in case Mermaid rendering is unavailable)

```
                    ┌─────────────────────────────────┐
                    │  GitHub  DominicTanzillo/...    │
                    │                                 │
   developer ──push─►  Repository                     │
                    │       │                         │
                    │       ├── CI (ci.yml):          │
                    │       │   pytest + docker build │
                    │       │                         │
                    │       └── CD (cd.yml, main only)│
                    │              │                  │
                    │              ▼ uses Secret      │
                    │           GCP_SA_KEY            │
                    └──────────────│──────────────────┘
                                   │ auth as
                                   ▼
                    ┌─────────────────────────────────────────────┐
                    │  GCP Project: ops-ai-dpt7 / us-central1-a   │
                    │                                             │
                    │  Service Account: github-actions@           │
                    │    container.developer                      │
                    │    artifactregistry.writer + reader         │
                    │    storage.objectViewer                     │
                    │                                             │
                    │   ┌─────────────────────┐                   │
   docker push ────►│   │ Artifact Registry   │                   │
                    │   │ docker-repo:        │                   │
                    │   │  demand-api:latest  │                   │
                    │   │  demand-api:$sha    │                   │
                    │   └──────────┬──────────┘                   │
                    │              │ image pull                   │
                    │              ▼                              │
                    │   ┌─────────────────────────────────────┐   │
                    │   │ GKE Cluster: operationalizing-ai    │   │
                    │   │ 2× n1-standard-2                    │   │
                    │   │                                     │   │
                    │   │ Deployment: demand-api (2 replicas) │   │
                    │   │   ┌─────────┐  ┌─────────┐          │   │
                    │   │   │ Pod #1  │  │ Pod #2  │          │   │
                    │   │   │ init:   │  │ init:   │◄──┐      │   │
                    │   │   │  gsutil │  │  gsutil │   │      │   │
                    │   │   │ main:   │  │ main:   │   │      │   │
                    │   │   │ FastAPI │  │ FastAPI │   │      │   │
                    │   │   │ :8000   │  │ :8000   │   │      │   │
                    │   │   └────┬────┘  └────┬────┘   │      │   │
                    │   │        └──────┬─────┘        │      │   │
                    │   │               ▼              │      │   │
                    │   │  Service: LoadBalancer       │      │   │
                    │   │    port 80 → 8000            │      │   │
                    │   └───────────────┬──────────────┘      │   │
                    │                   │                     │   │
                    │  ┌────────────────▼──────────────────┐  │   │
   client / curl ───┤  │  External IP   35.254.153.3       │  │   │
   /health, /api/* ─►  └───────────────────────────────────┘  │   │
                    │                                          │  │
                    │  ┌────────────────────────────────────┐  │  │
                    │  │ Cloud Storage                      │  │  │
                    │  │ gs://ops-ai-dpt7-data              │  │  │
                    │  │   demand_enriched.parquet (74 MB)  │──┘  │
                    │  │   demand_api_model.joblib          │     │
                    │  │   zone_hour_avg_fare.parquet       │     │
                    │  │   taxi_zones.geojson               │     │
                    │  └────────────────────────────────────┘     │
                    └─────────────────────────────────────────────┘
```

---

## What each arrow proves (for the design report)

| Edge | Operational property it demonstrates |
|---|---|
| Developer → Repository (push) | Source-of-truth is git; no out-of-band production changes |
| Repository → CD | Trigger is push-to-main; production is automated from green CI |
| CD → Secret → SA | Credentials never live on developer laptops; rotated by deleting key.json + recreating |
| CD → Artifact Registry | Immutable image artifacts tagged by git SHA enable rollback (`kubectl rollout undo`) |
| Artifact Registry → Pod | Private AR + IAM-gated read enforces supply-chain integrity (READING.md lines 86-98) |
| GCS → init container | Decouples model artifacts from image: rebuild image without rebuilding model; tradeoff: ~30s startup |
| Service: LoadBalancer | GCP-managed L4 LB gives stable external IP; survives pod restarts |
| readinessProbe ≠ livenessProbe (30s vs 60s initialDelay) | Per READING.md line 40: avoid restart cascades on slow model load |
