# Week 3 — AI Usage Disclosure

**Student:** Dominic Tanzillo (NetID: dpt7)
**Course:** AIPI 561 — Operationalizing AI, Summer 2026
**Submission:** Week 3 — Data Quality Validation & Graceful Degradation
**Date:** May 26, 2026

## Summary

This submission combines three categories of work: (1) course-provided scaffolding (READMEs, READING, the corrupted parquet, template files that I significantly extended); (2) AI-assisted implementation of validation code, tests, the graceful-degradation layer, and the CI workflow; and (3) student-authored design decisions, the investigation methodology, all severity and threshold choices, communication with the TA, and the prose of the writeup. AI was used as an implementation assistant for code and configuration edits and as a fact-checking / grammar-editing pass on the writeup. All design decisions, all troubleshooting calls made under uncertainty, all engineering trade-offs, and the final prose of the writeup are my own.

## Tool used

Anthropic Claude (model `claude-opus-4-7`) accessed via the Claude Code CLI on Windows 11. Multi-turn sessions spanning 2026-05-21 through 2026-05-26.

## High-level prompts I gave to drive the session

| # | Prompt (paraphrased) | Effect |
|---|---|---|
| 1 | "Let's get started on Weeks 3 and 4. I've done my homework, here's the slide content" | Initial planning and ensuring system has same access to information I do |
| 2 | "When you come to a question ask me so that I've done all the thinking work and you've just helped me to realize those goals" | Mode-switch: AI now asks; I decide to avoid AI making decisions |
| 3 | Investigation approach choices (open-ended exploration first; manifest only as self-check; discuss design framework now) | Set the investigation order |
| 4 | Design-principle answers (top fear = silent failures; tier by severity in gray zones; structured-log per occurrence on runtime degradation) | The three-principle spine of my design |
| 5 | Severity choice: all three checks CRITICAL; lag baseline: pre-compute JSON | First-pass validator design |
| 6 | "Do we want the one week lag or to add a sin/cosine feature for days/weeks. Can you reproduce the statistics with that change" | Lag-check methodology question; led to clarification of correlation-vs-semantic approaches |
| 7 | "Drop lag check entirely + document gap" | Final lag decision after seeing 28% false-positive rate |
| 8 | Test scope: comprehensive per-check positive + negative + edge; Degradation strategy: detect + auto-fix + fail-loud at 5% | Test and runtime decisions |
| 9 | Cron: hourly; Failure response: fail workflow + auto-create GitHub Issue | Workflow design |
| 10 | "I agree and have taken 3 days to confirm that any design decisions" | Affirmed the design decisions after 3 days of independent deliberation |
| 11 | Cleanup choices: delete duplicate templates + ge_config; move scripts to validation/scripts/; sparse-checkout course repo for data | Final structural choices |
| 12 | "Change [outlier threshold] to 1.5x historical max" | Threshold tuning decision (310 to 465) |

## Where AI assisted

### Files where AI generated or substantively contributed

| File | AI contribution |
|---|---|
| `week3/validation/check_data_quality.py` | `DataQualityValidator` class with three check methods (`check_value_ranges`, `check_holiday_labels`, `check_duplicates`), `Issue` dataclass for structured findings, `validate()` aggregator that returns `{passed, checks_run, issues, summary}`, and a CLI entry point with `--fail-on-critical` for CI gating. Severity choices, check selection, and the threshold tuning (310 → 465 = 1.5× historical max) were mine; AI translated them to code. |
| `week3/validation/test_data_quality.py` | 20 pytest tests: per-check positive / negative / edge cases plus 2 integration smoke tests against the real parquet. AI scaffolded the test layout per my "comprehensive per-check positive + negative + edge" choice. All 20 pass. |
| `week3/validation/holiday_calendar.py` | US Federal holiday lookup for 2023–2026 (`US_FEDERAL_HOLIDAYS` dict + `is_real_holiday()` / `holiday_name()` helpers). Used by the holiday-labels check to detect mislabeled `is_holiday=1` rows. |
| `week3/validation/clean_data.py` | Graceful-degradation layer applied at data-load time. Three fixes (drop invalid trip_count, deduplicate by `(PULocationID, time_bucket)`, correct mislabeled is_holiday flags). Raises `DataLoadTooBadError` if drop_rate > 5%. Emits a structured WARNING log line per fix. Implements my "detect + auto-fix critical, fail-loud on uncertainty" choice. |
| `week3/backend/data.py` | Added ~12 lines: a sys.path-import shim for the validation package and a `clean_dataframe(df)` call after each `pd.read_parquet` site (in `_load` and `_load_full_demand`). When fixes are applied, prints a one-line summary including the drop rate. The rest of `data.py` is course-provided. |
| `.github/workflows/validate-data.yml` (root) and `week3/.github/workflows/validate-data.yml` | Hourly cron + push triggers (path-filtered). Sparse-checks-out the upstream course repo for the parquet at run time (per TA guidance: do not upload large datasets to the solution repo). Stages the parquet into the expected path. Runs the pytest suite as a meta-test of the validator. Runs the validator with `--fail-on-critical` against the current batch. Opens / updates a GitHub Issue with a structured summary on critical findings. Uploads the JSON report as a workflow artifact. |
| `week3/validation/scripts/explore_data_quality.py`, `drill_down.py` | Investigation scripts used during the open-ended-exploration phase (null rates, value ranges, KS / PSI drift, duplicates by zone, per-zone aggregates, is_holiday-vs-federal-calendar cross-reference). Not run in CI; kept in `scripts/` for transparency about how the findings were derived. |

### CLI commands AI generated and executed

- All `git` operations: fetching upstream, comparing diffs, selective `git checkout upstream/main -- <path>` to sync just docs and Week 4 requirements without merging the 75 MB raw parquet into our history, and the threshold-change commit.
- `python -m pytest week3/validation/test_data_quality.py` for verifying the 20-test suite after each change.
- The selective sync of `week3/README.md`, `week3/SETUP_CHECKLIST.md`, `week3/backend/requirements.txt`, `week4/README.md`, `week4/requirements.txt` from upstream after the TA's mid-week force-push.

## Where AI did NOT assist (my original work)

- The Week 3 writeup (`week3/Week3Writeup.pdf`) was written by me. AI provided a fact-checking pass against the investigation numbers and flagged several typos in my draft (parquet vs paraquet, Kolmogorov vs Kolmogorow, etc.), but the prose, structure, and analytical conclusions are mine.
- All design decisions during the session: investigation methodology (open-ended exploration first vs hypothesis-driven), manifest stance, the three design principles, severity assignment (all three checks CRITICAL), the lag-check drop decision after seeing the 28% false-positive rate on a clean historical slice, test scope, degradation strategy, cron frequency, failure-response approach, repo strategy after the TA force-push, data strategy (workflow sparse-checkout vs commit-to-solution-repo), cleanup choices, and the threshold tuning from 310 to 465 (1.5× historical max).
- The independent observation during investigation that the shipped `demand_enriched_baseline.parquet` was a misleading 1,440-row single-zone sample, not a usable baseline. This was later confirmed when the TA removed that file as "redundant."
- All reading of the course slides, READING.md, the TA's mid-week announcements, and the updated upstream README.

## Course-provided baseline (NOT AI, NOT me)

For full transparency, the following files are course-provided and are either unmodified or modified only per explicit README TODO instructions:

- Course documentation: `week3/README.md`, `READING.md`, `SETUP_CHECKLIST.md`.
- Backend Python: `week3/backend/main.py`, `week3/backend/data.py` (TA-authored; I added the validation hook noted above), `week3/backend/requirements.txt` (TA-authored; I bumped `lightgbm` to 4.5.0 in Week 2 after diagnosing a load failure; TA later matched this pin in the upstream).
- Pre-trained model and data artifacts: `week3/data/demand_enriched_corrupted.parquet` (fetched at workflow run time from the upstream course repo rather than committed to the solution repo).
- Course-shipped templates that my files extend: `validate-data.yml`, `check_data_quality.py`, `test_data_quality.py` — all originally TODO-scaffolds in the upstream; my implementations replace the TODO sections with working code.

## Verification

A complete chronological activity log is maintained at the solution repo root in `AI_USAGE_NOTES.md` (gitignored to keep the public solution repo tidy). It contains the high-level prompt history, the per-decision rationale captured during the working session, and the source material from which the writeup was drafted. I can produce that file on request if grading verification requires it.

---

**Signed:** Dominic Tanzillo (NetID: dpt7)
**Date:** 2026-05-26

This disclosure is provided in compliance with the AIPI 561 AI Usage Policy (course syllabus, "Use of AI Assistants" section).
