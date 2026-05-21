# Week 2 — AI Usage Disclosure

**Student:** Dominic Tanzillo (NetID: dpt7)
**Course:** AIPI 561 — Operationalizing AI, Summer 2026
**Submission:** Week 2 — Deployment & CI/CD
**Date:** May 21, 2026

## Summary

This submission combines three categories of work: (1) course-provided scaffolding (READMEs, Dockerfile, Kubernetes skeletons with TODO placeholders, FastAPI backend code, pre-trained model artifacts); (2) AI-assisted implementation of code and configuration changes; and (3) student-authored design decisions, verification, communication with the TA, and the written design report. AI was used as an implementation assistant for code and configuration edits. All design decisions, all troubleshooting calls made under uncertainty, all GCP authentication and billing actions, and the final prose of the design report are my own.

## Tool used

Anthropic Claude (model `claude-opus-4-7`) accessed via the Claude Code CLI on Windows 11. Two multi-turn sessions on 2026-05-20 and 2026-05-21.

## High-level prompts I gave to drive the session

| # | Prompt (paraphrased) | Effect |
|---|---|---|
| 1 | "Help me set up Week 2: deploy the pre-trained LightGBM API to GKE with CI/CD per the course README" | Initial planning |
| 2 | (Pasted Week 2 README + slides) | "Please make sure that the work I've done so far is not missing anything from the README and slides. | Spot-checking work |
| 3 | "NetID is dpt7, GCP budget is $50 for the whole course, be smart about cost" | Project naming + cost discipline rules |
| 4 | "Walk me through gcloud setup" | Step-by-step authentication instructions which I completed |
| 5 | "Why is the forecast endpoint returning [] / hanging?" (multiple variants during debug) | Diagnostic guidance during debugging |
| 6 | "TA pushed updates overnight, merge them and reconcile with our local changes" | Upstream merge of TA's authoritative LightGBM fix |
| 7 | "CI/CD failed at rollout, what's wrong?" | Diagnostic guidance for the LFS-pointer-in-image bug |


## Where AI assisted

### Files where AI generated or substantively contributed

| File | AI contribution |
|---|---|
| `week2/starter/k8s/deployment.yaml` | Filled TODO values: replicas=2, RollingUpdate maxSurge=1 / maxUnavailable=1, imagePullPolicy=Always, resources requests cpu=512m mem=1Gi, limits cpu=1000m mem=3Gi, readinessProbe initialDelay=30s period=10s failureThreshold=3, livenessProbe initialDelay=60s period=30s failureThreshold=3. Substituted project ID `ops-ai-dpt7` into image path. I reviewed and approved each value. |
| `week2/starter/k8s/service.yaml` | Filled TODO values: type=LoadBalancer, selector app=demand-api, port=80, targetPort=8000 |
| `week2/starter/k8s/configmap.yaml` | Substituted GCS bucket name to `ops-ai-dpt7-data` |
| `week2/starter/.github/workflows/cd.yml` | Substituted `GCP_PROJECT_ID` to `ops-ai-dpt7` |
| `.github/workflows/ci.yml`, `.github/workflows/cd.yml` | Identified the course-shipped workflows at `week2/starter/.github/workflows/` would never trigger from there; copied them to `.github/workflows/` at repo root. Added `lfs: true` to checkout steps. |
| `week2/backend/requirements.txt` | Added `scikit-learn==1.5.0` (later removed when the TA's update switched the model loader to native LightGBM). Bumped `lightgbm==4.1.0` to `lightgbm==4.5.0` after empirically observing 4.1.0 cannot parse the v4-format model file (raised `LightGBMError: unordered_map::at`). |
| `week2/backend/data.py` | Performance optimization to `forecast_demand()`: hoisted the per-zone metadata lookup above the 672-iteration synthetic-history loop, eliminating ~2,688 redundant pandas dataframe filters per request. Forecast latency improved from 73 s to 3.6 s (~20× speedup). The rest of `data.py` is course-provided. |
| `week2/metadata/Lookups/taxi_zone_lookup.csv` | Diagnosed that the file was stored as a Git LFS pointer, which broke the CD-built Docker image because GitHub forks do not inherit LFS storage from upstream. Re-staged the file via `git rm --cached + git add` so it is now a regular git blob. |

### CLI commands AI generated and executed (with my approval for billable / credential actions)

- All `gcloud` commands: project creation, billing link, API enables, GCS bucket create, artifact uploads, Artifact Registry repo create, service account create + IAM bindings (4 roles), `key.json` generation, GKE cluster create / delete, and credential fetches.
- All `kubectl` commands: secret creation (`artifact-registry-secret` and `gcs-sa-key`), manifest application, rollout status / restart, log inspection, and rollback (`kubectl rollout undo` exercised twice during CD debugging).
- All `docker` commands: build, tag, push.
- All `git` commands: upstream remote configuration, fetch, merge (including the upstream merge of TA's LightGBM fix on 2026-05-21), file re-staging, commit, push.

## Where AI did NOT assist (my original work)

- The Week 2 design report (`week2/writeup.docx` and the exported `week2/week2writeup.pdf`) was written by me. 
- All GCP authentication and billing actions, including the interactive OAuth flow, the selection of "Billing Account for Education" for the $50 course credit, and the approval of each billable step (GKE cluster create, `key.json` generation).
- The screenshots PDF was assembled by me from terminal screenshots I captured directly in PowerShell.
- The `System Architecture.pdf` was rendered and laid out by me.
- All design decisions: which billing account to use, whether to detach the fork from the GitHub fork network, whether to make the repo private vs. public, whether to switch to the LightGBM `.txt` model after the TA's update, whether to invest the additional effort in the forecast latency fix, whether to add `lfs: true` to workflows vs. re-staging the CSV as a regular blob, and whether to do a final CI/CD verification run.
- Communications with the TA (Ananya Jogalekar) about the model artifact mismatch were authored by me.
- The reading of the syllabus, slides, README, READING.md, REQUIREMENTS.md, and SETUP_CHECKLIST.md was done by me.

## Course-provided baseline (NOT AI, NOT me)

For transparency, the following files are course-provided and are either unmodified or modified only per explicit README TODO instructions:

- Course documentation: `week2/README.md`, `READING.md`, `REQUIREMENTS.md`, `SETUP_CHECKLIST.md`, `TROUBLESHOOTING.md`.
- Backend Python: `week2/backend/main.py`, `week2/backend/data.py` (TA-authored; I added the perf optimization noted above), `week2/backend/requirements.txt` (TA-authored; I bumped one version pin).
- `week2/starter/Dockerfile` (unmodified).
- Original templates for `week2/starter/k8s/configmap.yaml`, `deployment.yaml`, `service.yaml` with TODOs filled per README guidance.
- Original templates for `week2/starter/.github/workflows/ci.yml`, `cd.yml`.
- Pre-trained model and data artifacts in `week2/data/`, `week2/model/`, `week2/metadata/`.

---

**Signed:** Dominic Tanzillo (NetID: dpt7)
**Date:** 2026-05-21

This disclosure is provided in compliance with the AIPI 561 AI Usage Policy (course syllabus, "Use of AI Assistants" section).
