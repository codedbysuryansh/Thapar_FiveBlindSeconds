# Phase 5 optimisation workspace (research only)

This folder holds the optimisation and experimentation programme that follows the Phase 4 build.
Nothing here is shipped, pushed or published. The submission package lives elsewhere:

- `champion/v1.0.0/` holds the immutable fallback (read-only, user-immutable flag, SHA256SUMS).
- `rc/` holds the separate final release candidate.
- `release/` is the local git repository at tag v1.0. It is never rewritten. A future release gets a new tag.

## Rules of the programme

1. **The champion is never edited.** Every experiment is a new candidate folder `candidates/<ID>/` with its own complete
   `main.py` snapshot, `config.json` and `meta.json` (SHA-256 of both, parent, hypothesis, creation time). A candidate is
   never overwritten. The harness refuses to run one whose main.py changed after registration.
2. **Same protocol for every candidate.** The inputs are the organisers' `degrade_imu.py` (SHA-256 cc7398db…2f31,
   unmodified) applied to the 17 public sequences. They are cached once in `cache/missions/` as the zip files main.py reads.
   Scoring uses the Phase 4 scorer `tools/evaluate.py` (= the research scorer p4.score; evo agreement within 5e-7 m),
   imported from the read-only champion copy. Ground truth is used only for scoring.
3. **Honest drag coefficient.** EuRoC missions use the leave-one-sequence-out k_drag (the mean of the other ten flights'
   fits). A candidate that changes the drag model must bring its own leave-one-out table.
4. **Mission sets** (see `protocol.json`):
   - DEV (seeds 1001–1004) is for exploration and tuning.
   - DEV2 (2001–2004) replicates DEV results before confirmation.
   - CONF (3001–3010) is used only for promotion decisions.
   - STRESS (cropped failure times, severity ×0.5 and ×1.5, ramp shapes) is used for robustness.
   - S42 (the organisers' fixed seed) is reported and never selected on.
   All sets cover both column orders and all 17 sequences.
5. **Tuned parameters are validated out of sample.** Use leave-one-environment-out (MH, V1, V2, TUM) for anything fitted
   to EuRoC. Hand-held parameters can only be left out at sequence level (TUM-VI has a single room), and this is stated
   wherever it applies.
6. **Paired comparison.** Every candidate is compared mission by mission against the current champion on the same set.
   - Composite = rubric-weighted (30 / 20 / 10) relative change of the mean waypoint, blind and after-dropout errors.
     Negative is better.
   - Points proxy = 60 × the weighted max(0, 1 − error / motionless error) of the same components.
   - Intervals are 95 % percentile intervals from a sequence-level bootstrap stratified by dataset (4,000 resamples,
     fixed seed).
   - Reports also list per-metric changes (ATE, RPE, horizontal and vertical ATE, worst waypoint, orientation), wins and
     losses by sequence, new failures, decision changes, byte identity and runtime.
7. **Promotion** (all must hold, on CONF and STRESS, after a DEV/DEV2 screen):
   - The pooled composite improves, with the 95 % interval below 0. Alternatively, the candidate is accuracy-neutral (interval within ±0.5 %) and measurably faster or leaner (a Pareto gain).
   - Neither dataset is significantly worse (composite interval entirely above 0), and no rubric component is significantly
     worse by more than 1 % in either dataset.
   - STRESS shows no significant composite loss and no new failures (contract errors, fallbacks, non-finite output).
   - Runtime (tools/measure.py, sequential, quiet machine) stays within +10 % of the champion unless the accuracy gain
     justifies it. The 2.1 GHz Xeon worst case stays at or below 12 ms per IMU-second.
   - Memory: own heap at 3 minutes stays within +0.5 MB, and the normal path stays below 8 MB at 10 minutes.
   - It passes the Phase 4 regression equivalents: T1 output contract, T4 robustness suite, T6 environments, T7 edge cases,
     and renderer identity.
7a. **Practical significance** (stated in round 2, applied since round 1): a change whose DEV gain is below 0.5 % composite on
   the dataset it affects is not pursued to DEV2/CONF even when its interval excludes 0 (for example P09h, X117e), because
   every promoted change costs a full regression and a gain that small is below the run-to-run spread of the hidden test.
7b. **Robustness test T4 for a new champion** (stated in round 2): no errors and no fallbacks; every decision change listed and
   explained (a change the candidate makes on purpose, such as the bias mode after the new rest detector, is reported, not
   hidden); the mean ATE of each family (E46, E55, E60) not worse; the cases > 25 % worse listed. A held-out robustness set
   (E46 stress seed 1, F55 = E55 windows of seed-7 files) is used for anything designed after looking at T4 cases.
8. **Negative results are kept.** Every candidate stays in `registry.json` / `REGISTRY.md` with status `accepted`,
   `rejected`, `pending`, `inconclusive`, `promoted` or `validation`, and the precise reason.
9. **No external action.** Nothing is pushed, uploaded, published or submitted without the user's explicit approval.

## Layout

- `harness/fbsopt.py`: cache building, candidate registration, runner (in-process, spawn pool), scorer and paired comparison, registry.
- `cache/`: degraded inputs and ground truth (derived from EuRoC/TUM-VI; local only; EuRoC is non-commercial).
- `candidates/<ID>/`: main.py, config.json, meta.json, run logs, results_<SET>.json, compare_<SET>_vs_<REF>.{txt,json}.
- `src/`: development source (`main_dev.py`, flags off = champion behaviour, verified byte-identical).
- `notes/`: experiment notes and analyses.
- `logs/`: workspace-level logs.
