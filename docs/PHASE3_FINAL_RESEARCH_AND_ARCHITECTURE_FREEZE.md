# PHASE 3 — FINAL RESEARCH, COMPETITIVE RED-TEAMING AND ARCHITECTURE FREEZE

**Five Blind Seconds · Samsung PRISM XR Hackathon 2026 · Track 03 "Path Finder: Trajectory Estimation Under Sensor Degradation"**
21 Nov 2026. Supersedes `PHASE2_DECISION_GATE_REPORT.md` wherever the two differ. Phase 4 builds exactly what §12 specifies.

Research reference implementation (not the submission): `research/harness/fbs6.py` (`front_final`, `back_final`, `estimate_final`, `V5`) and `research/harness/blind5.py` (`final_blind`), on the Mac in `~/SamsungPRISM_FiveBlindSeconds`. Every experiment (E40–E65) has its script and its results (JSON or log) on the Mac; Appendix A gives the paths. Unless stated otherwise every number below comes from **the organisers' own `degrade_imu.py` applied to the 17 public sequences** (11 EuRoC MAV flights, 6 TUM-VI rooms), scored the way the FAQ says the evaluator scores: SE(3) Umeyama alignment over all output rows, then errors against the public ground truth.

How to read the metrics (one example, MH_03 with the IMU dying at the end): **ATE** = RMS position error over every row after alignment; **wp** = mean error at 8 evenly spaced pre-failure "waypoints"; **b_abs** = mean absolute error over the five blind seconds; **b_anc** = error of the blind-window *displacement* measured from the last valid pose (what the predictor alone controls); **b_end** = error at the last row; **drop** = mean error in the 1 s after every gap of ≥ 5 missing samples; **ori** = RMS orientation error after the same alignment. "Motionless" = all positions zero; it is the honest floor every IMU-only method must beat.

---

## 0. Bottom line

**Frozen architecture (v5).** A batch, platform-conditioned smoother. Ingest the CSV onto a uniform slot grid (duplicates share a slot; zero, NaN and exact-repeat rows are invalid; missing rows are re-inserted as invalid slots); treat the trailing invalid run as the blind window; estimate attitude with an integrated-random-walk gravity smoother whose rotation uncertainty grows with the square of each gap's length; classify the platform from how rigidly the specific force stays on one body axis (rotor if the lock statistic < 0.035); fit the gyro bias as a 20 s-knot spline for rotors (constant or rest-anchored for hand-held unless a drift test fires); then smooth position with a per-platform stochastic prior: rotor = 17-state spring-damper + rotor-drag + lever-arm model at 50 Hz (25 Hz in the submission configuration), hand-held = per-axis Matérn-5/2 stationary prior at 100 Hz, both with zero-velocity rows at genuine rest and accelerometer-bias variance injected at each gap in proportion to its length. A radius guard falls back to the hand-held solution if the rotor solution diverges. Blind window: rotor = smoothed velocity at failure decaying with a 0.5 s time constant; hand-held = the smoother's own prediction; orientation held. Output: one pose per input row, duplicates and blind rows included.

**Results on the official degradation, fresh seeds** (E57: full sequences, seeds 5–13 — never used for any choice — plus seed 42, both column orders, 20 missions per sequence, 340 missions):

| Group | Metric | Motionless | Phase 2 v4 (as is) | v4 + fixed gate and blind tail | **v5 (frozen)** | v5 vs motionless, per-sequence skill [95 % CI], wins |
|---|---|---|---|---|---|---|
| EuRoC (11) | ATE | 3.448 | 3.551 | 2.946 | **2.656** | 25.5 % [16.8, 34.0], 11/11 |
| EuRoC | waypoints | 3.248 | 3.360 | 2.759 | **2.480** | 27.9 % [17.8, 38.1], 11/11 |
| EuRoC | blind b_abs | 3.083 | 3.086 | 2.970 | **2.761** | 6.3 % [−0.0, 12.2], 8/11 |
| EuRoC | after-dropout | 3.045 | 3.110 | 2.504 | **2.282** | 25.0 % [15.7, 34.3], 11/11 |
| TUM-VI (6) | ATE | 1.080 | 1.117 | 1.117 | **0.451** | 57.6 % [43.7, 68.1], 6/6 |
| TUM-VI | waypoints | 1.110 | 1.116 | 1.116 | **0.388** | 63.1 % [41.0, 79.0], 6/6 |
| TUM-VI | blind b_abs | 0.614 | 0.615 | 0.615 | **0.599** | 2.4 % [−1.6, 6.4], 4/6 |
| TUM-VI | after-dropout | 1.025 | 1.001 | 1.001 | **0.408** | 58.7 % [43.5, 71.2], 6/6 |
| All 17 | ATE | 2.612 | 2.692 | 2.300 | **1.878** | 36.8 % [26.9, 47.1], 17/17 (aggregate −28.1 %) |

Metres. v5 beats the Phase 2 pipeline on 17/17 sequences (per-sequence 29.7 % [17.9, 42.2]). The Phase 2 pipeline, run as it was, is *worse than motionless* on the official data (§2.3): the organisers' script broke two of its assumptions.

**Submission configuration.** The runtime cost model (§12.6) moves two internal rates: gravity smoother 200 → 12.5 Hz and rotor smoother 50 → 25 Hz (`fbs6.V5P4`). Measured on the same 578 missions (E65): EuRoC ATE 2.670 m (+0.5 %, 95 % CI of the paired difference [−0.009, +0.043] m, n.s.), waypoints +0.3 %, blind +0.1 %, after-dropout +0.1 %; TUM-VI identical; every gate decision identical. Phase 4 reproduces **V5P4**.

**What is strong, what is weak.** Waypoints and post-dropout error: large, significant gains everywhere. **The blind five seconds is the weakest component and is close to its information limit:** even a perfect predictor of the blind-window motion (an oracle used only as a diagnostic) would lower EuRoC b_abs by only 6–8 % and TUM-VI b_abs not at all, because the blind error is dominated by the position error already present at the moment of failure — where the official script's degradation is at its maximum (§8). Efficiency (40 %): a cost model built from kernels measured in their final form gives ≈ 8 ms per IMU-second on the M3 Pro and ≈ 15.6 ms on a slow 2.1 GHz cloud vCPU, worst case, against the 20 ms budget; own heap planned ≤ 4 MB (§12.5–12.6). The finished program is measured in Phase 4.

**Acceptance gate (§13):** all nine items are established with evidence at the research and architecture level. Item 8 (resource feasibility) rests on measured kernel costs and a validated design, not yet on a measurement of the final `main.py`; two engineering unknowns cannot be settled by any Phase 3 experiment: whether the organisers' tracemalloc window includes `import numpy` (7.5–8.1 MB on its own) and how fast the evaluation machine is (§14).

**Changes since the Phase 2 decision (v4 → v5):** gap-filled platform statistic; integrated-random-walk gravity smoother with the T² gap law; gap-length-scaled accelerometer-bias variance; per-platform gyro-bias rule (rotor spline); per-platform smoother rates; radius guard; gate threshold 0.04 → 0.035; rotor blind rule (damped velocity) — each backed by §6/§8. Rejected after testing: learned velocity prior (E48), extra despiker, adaptive noise, accelerometer-bias body states, hand-held blind extrapolation, more spline iterations.

---

## 1. Governing sources and compliance matrix

**Sources and precedence** (later and more specific wins): S1 = organiser deck v1 PDF (5 Nov; SHA-256 `04c90d6d…c65a32`), slides 13–17 (Track 03) and 23–26 (rules); S2 = deck V3-2 pptx and the 9-slide submission template, attached to the organisers' mail of 13 Nov (Track 03 slides identical to S1); S3 = that mail (deadline, GitHub, form); S4 = `degrade_imu.py` (SHA-256 `cc7398db…2f31`); S5 = the official Track 3 FAQ. Precedence: **S5 > S3 > S2 > S1**. Slide numbers are the printed footers.

| # | Requirement (exact wording) | Source | Implication | Required behaviour (Phase 4) | Status | Evidence |
|---|---|---|---|---|---|---|
| C1 | "Estimate the 6-DoF trajectory (position + orientation) through all mission checkpoints using only degraded IMU streams" | S1 s13 | IMU only; no camera, VIO, GPS or other sensor anywhere in the scored pipeline | `main.py` reads only the IMU CSV; no network, no other files at inference | Design meets | v5 input = `imu0/data.csv` only |
| C2 | "Your pipeline must estimate the trajectory using ONLY the degraded IMU file. Waypoints are used internally by our evaluation pipeline … completely hidden from your algorithm." | S5 | No waypoint or timestamp side information; Phase 2 Scenario A (supplied checkpoints) is irrelevant | No checkpoint input path in `main.py` | Design meets | §3 |
| C3 | "Datasets: EuRoC MAV & TUM-VI; organizer degradation script provided"; "You can freely use any sequence from the publicly available EuRoC MAV or TUM-VI datasets as your sample data" | S1 s13, S5 | Hidden data likely degraded EuRoC/TUM-VI; possibly the public sequences themselves (unknown) | Generalise across platforms; no sequence recognition or ground-truth lookup | Design meets | §5, §7 |
| C4 | "The evaluation dataset is a zipped folder of raw IMU telemetry … File path: imu0/data.csv … comma-separated … Header line: #timestamp [ns], w_RS_S_* [rad s^-1], a_RS_S_* [m s^-2] … timestamp [ns] — 64-bit integer, nanoseconds since Unix epoch" | S1 s15 | Column order gyro, then accelerometer (EuRoC); int64 ns; header present | Accept a zip, a folder containing `imu0/data.csv`, or a CSV path; parse header-optional, whitespace-tolerant, int64 timestamps (never float) | To build (P4) | §12.3; E51 audit |
| C5 | "accel includes gravity"; "unknown gravity alignment" | S1 s14–15 | No assumption about IMU mounting | Gravity direction and thrust axis estimated from data | Verified | E60: rotated sensor frames change TUM-VI by ≤ 4 mm and EuRoC order A by ≤ 5.4 cm (median 9 mm), never a gate or mode decision; order-B rotor sensitivity disclosed (§7.6) |
| C6 | "The evaluation dataset will contain timestamps during the blind 5-second window … but the sensor values will be unusable/invalid." "DO NOT append extra poses beyond the input timestamps." | S5 | Blind rows exist in the input (the script writes exact zeros) | Trailing invalid run = blind window; predict a pose for each of those rows; append nothing | Design meets | `split_tail`; E52/E57 |
| C7 | "dropouts are represented by physically MISSING rows … sudden jumps in your timestamp delta" | S5 | Gaps are discovered from timestamps | Slot grid at the median Δt; missing slots invalid; the output still has exactly the input rows | Design meets | `ingest2`; E45, E58 |
| C8 | "File name: estimated_trajectory.txt (space-separated) · Header optional — commented lines must start with # · One estimated pose per row, exactly 8 columns · timestamp — floating-point seconds (Unix epoch) · tx ty tz (meters) · qx qy qz qw — normalized unit quaternion (world frame)" | S1 s16 | TUM format | Write `%d.%09d` from integer ns (no float rounding), positions in metres, unit quaternion (x, y, z, w) | Verified on research output | E51: 8 columns, timestamps identical to input, ‖q‖−1 < 1e-7, finite |
| C9 | "exactly one pose per IMU frame — dropped or misaligned rows incur heavy penalties or failure" | S1 s16, S5 | Duplicated input rows (the script makes them, same timestamp) need their own output rows | One output row per input row, in input order, duplicates kept | Verified on research output | E51: 35,629 in / 35,629 out with 49 duplicate stamps; evo accepts them |
| C10 | "Trajectory CSV: timestamp, x, y, z, qx, qy, qz, qw" vs. "estimated_trajectory.txt (space-separated)" | S1 s14 vs s16 | Wording conflict | Follow s16 (the format slide and evo's TUM reader) | Resolved by design | E51 |
| C11 | "Standalone Python script" (s14) vs "main.py / main.cpp # entrypoint / build system" (s17) → "we are requiring all teams to stick exclusively to Python for Track 3 … Please submit a standalone Python script (main.py)." | S1 s14, s17; **S5 resolves** | C/C++ entry point not accepted | `main.py`, Python + NumPy only; the Phase 2 C prototype is retired to evidence | Resolved | S5 |
| C12 | "≤ 8 MB peak RAM" → "measured at the ALGORITHM / HEAP level … the memory footprint utilized by your data structures, buffers, and trajectory estimation logic, not the base OS-level RSS of the runtime" | S1 s13–14; S5 | Heap of our own data, tracemalloc-like. **Unknown:** whether `import numpy` (7.5 MB traced on macOS, 8.1 MB on Linux, on its own) falls inside their window | Own heap ≤ 4 MB at the longest public sequence; report both numbers (with and without the import) in README and `metadata.json` | Plan; not measured on the final build | §12.5; E50 |
| C13 | "≤ 20 ms compute per 1 s of IMU, single CPU thread, offline" | S1 s13–14 | Runtime budget ~3.7 s for the 184 s MH_01 input | Pin BLAS/OpenMP threads to 1 before importing NumPy; measured-kernel cost model ≈ 8 ms per IMU-second on the M3 Pro and ≈ 15.6 ms on a 2.1 GHz Xeon cloud vCPU (worst case); Phase 4 T5 targets ≤ 9 / ≤ 17 ms | Plan; kernels measured | §12.6; E50, E62, E62b, E64 |
| C14 | "You may read the entire CSV into memory (while strictly staying under the 8 MB limit) and perform non-causal / batch smoothing" | S5 | Offline smoothing allowed | Forward filter + backward RTS pass | Design meets | §3 |
| C15 | "the estimated trajectory will be aligned to the hidden ground truth using SE(3) alignment (Umeyama) prior to computing the Absolute Trajectory Error" | S5 | No scale alignment; frame offset irrelevant | Our scorer = `evo_ape -a` | Verified | E51: ours vs evo ≤ 0.4 mm on six missions |
| C16 | "Ranked with the open-source evo trajectory evaluation suite … ATE (RMSE) … RPE — relative pose error: local drift and rotational error per unit of time" | S1 s17 | RPE (translation and rotation) may also be used; its settings are unpublished | Keep orientation physically sensible everywhere, incl. the blind rows | Design meets; RPE settings unknown | E51 (our RPE ≈ evo's within 2 %, rotation up to 26 % apart on one room), §7.8, §8 |
| C17 | "EVALUATION (100 PTS) · Execution & Efficiency — 40% · Peak memory 20% · Runtime 20% · Trajectory Closeness — 60% · Waypoints 30% · Blind 5 s 20% · Dropout 10%" | S1 s14 | Exact weights; no metre-to-points map published | Optimise every component; never report an "official score" | Recorded | §10 |
| C18 | "Tiers: T1 checkpoints · T2 blind 5 s (+20%) · T3 dropout resilience (+20%)" | S1 s14 | Conflicts with "Dropout 10%" in the same slide | Preserved as an ambiguity; no total invented | Unresolved (cannot ask) | — |
| C19 | "After the final checkpoint the IMU dies; the aircraft must dead-reckon its 3D path for 5 seconds" | S1 s13 | Blind window always at the end (the script zeros the last 5 s) | §8 rule | Design meets | E53, E57 |
| C20 | "team_name.zip ├── estimated_trajectory.txt # final generated trajectory file ├── metadata.json # optional: runtime perf, compute specs ├── src/ … main.py … README.md # how to compile & run your code" | S1 s17 | We may have to ship a generated trajectory and the source | Phase 5: produce the bundle from the public sample protocol; `metadata.json` with measured memory/runtime | To do (P5) | §12.9 |
| C21 | "All the code, resources, document, demo, ppt, package, readme etc must be put in 'Public GitHub' link. & share the link through the form" | S3 | Repository must be public at submission | Phase 5 only (the user's standing rule forbids publishing before) | To do (P5) | §12.9 |
| C22 | "README with reproducible setup instructions, Docker files & other requirements" | S1 s24 | Dockerfile + requirements expected | `Dockerfile` (python:3.12-slim + numpy), `requirements.txt`, exact run commands | To do (P5) | §12.9 |
| C23 | "Demo video, max 3 minutes (YouTube or Drive link)"; FAQ: "aim for 2 to 3 minutes, with a strict maximum of 3 minutes" (s23 still says "5-min demo video") | S1 s23–24, S5 | 2–3 min | Phase 5 video 2:30–2:55 | To do (P5) | — |
| C24 | "Presentation file (PPT or PDF) — CollegeName_TeamName, follow the nomenclature"; template: Track No / XR Hackathon / Team Name / College Name / Members; THE PROBLEM; SOLUTION; ARCHITECTURE; Track Final Output; DIFFERENTIATION; DEMO & LOCAL SETUP; Brownie Slides; Thank You | S1 s24, S2 | Fixed 9-slide structure; "Teams NOT following the submission guideline would lead to direct disqualification" | `Thapar_FiveBlindSeconds` deck on the template | To do (P5) | — |
| C25 | "Final Submission Date: 22nd Nov"; form `https://forms.gle/86xENrtqhx988hBt8`; one submission per team | S3, S1 s24 | Supersedes 18 Nov | Phase 5 | To do (P5) | — |
| C26 | Dataset terms: EuRoC deposit "In Copyright – Non-Commercial Use Permitted" (ETH Research Collection, DOI 10.3929/ethz-b-000690084); TUM-VI attribution | P1B-01 | Do not redistribute EuRoC data in a public repository | Repo ships code, the unmodified organiser script reference and download instructions, not dataset files; cite both datasets | Plan | §12.9 |
| C27 | "single CPU thread" and "offline" | S1 s13 | No threads, no GPU, no network | Environment pins `OMP_NUM_THREADS=1` etc.; no multiprocessing | Plan | §12.6 |

**Ambiguities that remain (we may not ask the organisers):** (A1) which column order *they* fed to their script (§2.2; both validated); (A2) how `main.py` is invoked (arguments, input and output paths) — handled by a defensive CLI (§12.2); (A3) whether the tracemalloc window includes interpreter start-up and imports; (A4) the T2/T3 "+20 %" tiers; (A5) the exact waypoint, blind and dropout metric definitions (alignment over all rows or a subset; absolute vs. anchored blind error) — §8 picks rules that are good under every definition we could think of; (A6) the evaluation machine; (A7) whether the hidden sequences are the public ones (decides how much a memorising learned model would gain — §5).

---

## 2. The official degradation, audited (E40–E45)

### 2.1 What `degrade_imu.py` does to a file

Severity ramps from 0.02 to `max_level` (1.0) as (t/T)^1.7, so **the last seconds before the blind window are the worst-degraded part of every file**.

| Fault | Parameters (× severity) | How it appears in the CSV | v5 handling |
|---|---|---|---|
| White noise | accel 0.20 m/s², gyro 0.015 rad/s per sample | larger sample scatter | data-driven noise level (high-frequency std) |
| Bias random walk | accel 0.010 m/s²/√s, gyro 0.0006 rad/s/√s | slow offsets | spline gyro bias (rotor) / drift test (hand-held); OU accel-bias states |
| Scale random walk | 0.0015 /√s | slowly varying gain | absorbed by the bias and noise states |
| Spikes | p = 0.003, same row on both sensors, accel 6, gyro 1.0 (× U(0.5, 1.5)) | outlier rows | gravity-magnitude plausibility + legacy despike (no measurable effect, kept for safety) |
| Stuck values | p = 0.008, 2…(2 + 8·sev) samples | exact repeats | every exact repeat is invalid (first sample of the run kept) |
| Single-row drops | p = 0.06 | missing rows (Δt jumps) | slot grid; invalid slots |
| Burst drops | p = 0.0015, 5–50 samples | missing runs | same; gap-length-aware uncertainty |
| Duplicates | p = 0.003 | repeated row with the **same** timestamp | share one slot; one output row each |
| Protection | first 2 s free of spikes/stuck/drops | — | — |
| Blind window | last 5 s set to exact zeros, rows kept | zero rows at the end | blind window detection |
| Rounding | 6 decimals | — | — |

### 2.2 The column-order ambiguity

The script's docstring orders columns (t, accel, gyro); EuRoC files are (t, gyro, accel). "Order A" = the docstring is right (the script is given a re-ordered frame); "order B" = the script is run on an EuRoC file as-is, so the **gyro columns receive the accelerometer parameters**. Under B, seven EuRoC flights (MH_01–05, V1_01, V2_01: median |ω| < 0.5, which the script reads as "g units") get gyro noise of ~0.18 rad/s and gyro bias up to 0.05–0.13 rad/s by the end; the others get ~0.018 rad/s and ≤ 0.01 rad/s. We cannot know which the organisers used, so **every result is reported for both**; v5 is tuned on neither in particular (E52/E57 tables give both).

### 2.3 Why the Phase 2 pipeline failed on official data, and the root-cause fixes

| Finding | Cause | Fix in v5 | Effect |
|---|---|---|---|
| v4 EuRoC ATE 3.55 m (motionless 3.45): 95/153 platform-gate errors (E41) | per-segment zero-phase low-pass of v4 breaks when single-sample drops are frequent; the lock statistic rose from ≤ 0.027 to 0.045–0.063 | `thrust_lock_fill`: fill gaps first, one low-pass over the whole stream | 0 gate errors in the 578 full / crop / level / clean missions of E52 (same in E57); short prefixes in §7.4 |
| TUM-VI health fell 0.995 → 0.83, noisy tilt (E44) | IRW gravity smoother injected rotation uncertainty c·‖ω‖·T at *every* gap, so a 5 ms drop reset it | rotation uncertainty θ = min(k₂T²/2, 1 rad) with k₂ = 0.25 rad/s² (rotation from an unknown angular acceleration over a gap) | E58: linear law costs +0.52 m on TUM-VI (6/6) and +0.15 m on EuRoC (8/11) |
| Back end re-opened the full accelerometer-bias variance (0.25) at every gap (E44) | gap events were treated as resets irrespective of length | variance × min(1, T / 2 s): a random walk's variance grows with elapsed time | E58: without it +0.27 m TUM-VI (6/6), +0.32 m EuRoC (10/11) |
| Spline gyro bias hurt TUM-VI heading (ori 9° → 37°) (E41) | drift gate calibrated on our own simulator | per-platform rule: rotor always spline; hand-held spline only if D > 2 mrad/s or health < 0.9 | E58: rotor const +0.047 m (10/11); hand-held always-spline +0.015 m |

---

## 3. Frozen architecture (v5)

### 3.1 Pipeline

```
imu0/data.csv
  │
  ├─[1 Ingest]        uniform slot grid at the median Δt; rows with the same timestamp share a slot (one output row each);
  │                   zero, NaN and exact-repeat rows are invalid (first of a stuck run kept); missing rows → invalid slots
  ├─[2 Blind split]   trailing run of invalid slots = the five blind seconds (predicted, never fitted)
  ├─[3 Repair]        ‖a‖ outside 5.0–15.5 m/s² → invalid; legacy despiker (no measurable effect on official data, kept for safety)
  ├─[4 Platform gate] lock = median over 10 s windows of RMS(f⊥)/g, f = specific force after gap filling and ONE 1 Hz low-pass,
  │                   f⊥ = component perpendicular to the window's mean direction.   rotor ⇔ lock < 0.035
  ├─[5 Attitude]      OU gyro bridge across gaps → strapdown → IRW gravity smoother (V5P4: on 12.5 Hz block means; gap rotation uncertainty θ = min(k₂T²/2, 1 rad),
  │                   k₂ = 0.25 rad/s²) → tilt correction.  Gyro bias: rotor → penalised linear spline (20 s knots, λ₂ = 3, ≤ 6 passes);
  │                   hand-held → constant or rest-anchored, spline only if drift D > 2 mrad/s or gravity health < 0.9
  ├─[6 Position]      world-frame specific force (mean removed) is a MEASUREMENT of a stochastic motion prior; Kalman filter + RTS smoother
  │                     rotor     : spring-damper prior + OU accel bias per axis, 2 drag-bias + 3 lever-arm states (17 states),
  │                                 rotor-drag rows in flight, zero-velocity rows at genuine rest; measurements averaged in 4-sample blocks (50 Hz;
  │                                 V5P4: 8-sample blocks, 25 Hz)
  │                     hand-held : Matérn-5/2 stationary prior + OU accel bias per axis (3 independent axes with one shared 4 × 4
  │                                 covariance), zero-velocity rows; 2-sample blocks (100 Hz)
  │                     every gap : accelerometer-bias variance += 0.25 · min(1, T / 2 s)
  ├─[7 Guard]         if RMS radius(rotor solution) > 12 × RMS radius(hand-held solution) → use the hand-held solution
  ├─[8 Blind]         rotor: p(τ) = p_L + 0.5 s·(1 − e^(−τ/0.5 s))·v_L ; hand-held: the smoother's own prediction ; orientation held at q_L
  └─[9 Output]        one pose per INPUT row, input order: "<sec>.<9-digit ns> tx ty tz qx qy qz qw", unit quaternion
```

Analogy for the core idea: a strapdown integrator is a blindfolded walker counting steps — every miscount is carried forever. v5 instead asks "which smooth, physically plausible path of a drone (or of a hand-held rig) best explains every accelerometer sample I saw, before *and after* this moment?", the way a detective reconstructs a route from all the clues at once rather than one at a time. The prior ("drones hover around a room; people do not walk through walls") is what keeps the unobservable low-frequency part of the motion from drifting away.

### 3.2 Models and equations (per world axis j ∈ {x, y, z} unless stated)

- **Hand-held prior (Matérn-5/2, ℓ = 1 s, σₚ = 2 m):** state (p, ṗ, p̈); F = [[0,1,0],[0,0,1],[−λ³, −3λ², −3λ]], λ = √5/ℓ, white-noise spectral density 16/3·σₚ²λ⁵. Stationary: position has finite variance, so the smoother pulls unobservable drift back towards the trajectory's centre instead of letting it grow as t^1.5.
- **Rotor prior (spring-damper):** ṗ = v; v̇ = −ω₀²p − v/τᵥ + d; ḋ = −d/τₐ + w, with T_p = 2π/ω₀ = 20 s, τᵥ = 2 s, τₐ = 0.3 s, σₐ = 1.0 m/s².
- **Accelerometer bias:** OU, τ_b = 20 s, σ_b = 0.1 m/s², one state per world axis; at the end of every gap of length T its variance is increased by 0.5²·min(1, T/2 s).
- **Measurement:** a_w,j = p̈_j + b_j + n, n ~ N(0, σₙ²) with σₙ the high-frequency standard deviation of the stream itself (data-driven: adapts to the degradation level). a_w = R(q)·f − mean.
- **Rotor-drag rows (in flight only, i = 1, 2 = the two body axes perpendicular to the thrust axis):** e_i·f = −k·(R e_i)·v + c_i + e_i·(ω̇ × r + ω × (ω × r)) + n, k = 0.187 s⁻¹ (mean of 11 per-flight fits; leave-one-sequence-out values in every evaluation), σ = 1.4 m/s²; c_i = OU drag biases (τ 100 s, σ 0.15); r = IMU lever arm (3 states, prior 0.15 m).
- **Zero-velocity rows:** v = 0 ± 0.02 m/s inside windows accepted by the genuine-rest test (quietest 5 % of 0.5 s windows with consistent gyro means).
- **Gravity smoother:** integrated random walk on the gravity vector expressed in the gyro-integrated frame, measured by the specific force (σ 1.5 m/s²); time constant 3 s; across a gap of length T the gravity direction may have turned by θ = min(k₂T²/2, 1 rad), injected as process noise (gθ/σ)². (Physically: an unknown angular acceleration integrated twice over the gap — a 5 ms drop is ~nothing, a 2 s gap can be a full turn.)
- **Gyro bias spline:** bias increments as a linear spline in time (20 s knots), estimated by penalised least squares from the drift of the tilt correction, with the regressors passed through the same gravity filter and 1 s difference as the response (otherwise least squares under-estimates the bias when the body rotates; Phase 3 `attitude6` finding, WORKLOG 14 Nov: room1 ramp tilt error 3.0° → 0.6°).
- **Smoother:** sequential scalar Kalman updates, Van Loan discretisation per distinct Δt, RTS backward pass. Measurements of a block of d samples are averaged (information-preserving for white noise: R/count).
- **Blind (rotor):** τ = time since the last valid sample; v_L = smoothed velocity there. Displacement saturates at 0.5 s·v_L: a shrinkage of a velocity that IMU-only data barely observe (mean velocity error at failure 0.41 m/s vs a mean true speed of 0.32 m/s on full EuRoC sequences, E53).

### 3.3 Parameters and their provenance

| Stage | Parameter | Value | Chosen by | Ablation (E58, ΔATE when removed / reverted; EuRoC, TUM-VI) |
|---|---|---|---|---|
| Ingest | duplicates share a slot; exact repeats invalid | on | official-script audit (E40) | +0.001, +0.000 (n.s.) — kept for correctness of the row map |
| Repair | ‖a‖ window | 5.0–15.5 m/s² | E23 (touches 0 samples on clean data) | +0.001, +0.000 (n.s.) — safety only |
| Gate | lock threshold | **0.035** (was 0.04) | E55: empty gap between rotor flight windows ≤ 0.0317 and hand-held ≥ 0.0408 over 884 missions/windows (E52 + E55); cost asymmetry (rotor-on-hand-held up to +220 % ATE when the guard misses vs +25 % the other way) | no gate (all hand-held): +0.646 (11/11), 0; all rotor without guard: 0, +31.6 m |
| Gravity smoother | time constant, σ | 3 s, 1.5 m/s² | Phase 3 E25 | **removing it: +1.79 (11/11), +1.49 (6/6)** — the single most important component |
| | gap law, k₂ | T², 0.25 rad/s² | E44 root cause; E46b (0.25 s gaps prefer small k₂; ≥ 2 s gaps insensitive); E47 grouped CV (0–0.25 within 0.5 %) | linear law: +0.150 (8/11), +0.519 (6/6); k₂ = 1: +0.010, +0.028 (6/6) |
| Gyro bridge | OU τ, mean window | 0.4 s, ±5 s | E26 | ~0 on official data (gaps are short); kept for 2 s gaps (E26: 39° → 25° rotation error) |
| Gyro bias | rotor: spline; knots, λ₂ | always; 20 s, 3 | E46 gate sweep, E47 | constant bias on rotor: +0.047 (10/11) |
| | hand-held: drift threshold, health | 2 mrad/s, 0.9 | Phase 2 E16c, E46 (official TUM D 99th pct 1.68) | always-spline on hand-held: +0.015 (n.s.) |
| Rotor prior | T_p, τᵥ, τₐ, σₐ | 20 s, 2 s, 0.3 s, 1.0 | Phase 2 E13 (10/11 LOSO); E49 grouped CV (+1.0 % for the CV choice) | — |
| Hand-held prior | ℓ, σₚ | 1 s, 2 m | Phase 1B E02 (LOSO majority); E49 (CV choice −1.2 % but worse on 4/6 rooms → kept) | — |
| Drag rows | k, σ | 0.187 s⁻¹, 1.4 m/s² | Phase 2 E05/E16 | removing drag: +0.635 (11/11) |
| Lever arm | 3 states, prior | 0.15 m | Phase 2 E16 | removing: +0.121 (9/11) |
| ZUPT | σ | 0.02 m/s | Phase 2 | removing: +0.052 (8/11), 0 |
| Gap bias injection | 0.5², T/2 s | — | E44 root cause; E45 (gap_ref 1/2/4 equal) | removing the T/2 s scaling: +0.316 (10/11), +0.266 (6/6) |
| Governor | block size rotor / hand-held | 4 / 2 (**V5P4: 8 / 2**) | E47 (rotor 2/4/8 = +0.0/+0.15/+0.7 %; hand-held 2/4/8 = +0.4/+2.5/+9 %); V5P4 from the runtime model (§12.6) and E65 (+0.5 % EuRoC ATE, n.s.) | full rate: −0.000, −0.003 (cost ×4 / ×2 runtime) |
| Gravity-smoother rate | block means of d slots | full rate (**V5P4: d = 16, 12.5 Hz**) | E63/E63b (≤ 4 mm per mission down to 3.1 Hz), E64c cost | — |
| Guard | radius ratio | 12 | E47 (EuRoC max 8.50, TUM-VI min 16.40 on full sequences); E55b | 0 on full sequences; rescues all-rotor on TUM-VI (+31.6 → +0.015) |
| Blind | rotor decay τ; hand-held | 0.5 s; model | E53 (selection), E57 (fresh-seed confirmation) | §8 |

Every component that survived costs ATE when removed, except dedup, repair, bridge and guard, which cost nothing on the official data and are kept for robustness and for the output contract; none of them is expensive (§12.6).

---

## 4. Results under the official degradation (E52, E57)

Protocol: `p4.official` reproduces the organisers' script bit-for-bit (a diagnostic copy that also returns the fault masks gives identical output on 34/34 files, E40). Drag coefficient leave-one-sequence-out in every evaluation (the submission will use the all-data mean). Seeds 1–4 were used for tuning (E41–E47), seeds 42, 1–4 and the crops for the blind-rule selection (E53); **the confirmation set is seed 42 plus seeds 5–13**, which no decision saw except seed 42.

### 4.1 Full sequences (the official situation: IMU dies at the end of the recording)

See the table in §0. By column order (E57, metres):

| | EuRoC A | EuRoC B | TUM-VI A | TUM-VI B |
|---|---|---|---|---|
| Motionless ATE | 3.448 | 3.448 | 1.080 | 1.080 |
| **v5 ATE** | **2.576** | **2.737** | **0.448** | **0.454** |
| v5 waypoints | 2.414 | 2.547 | 0.383 | 0.393 |
| v5 blind b_abs | 2.717 | 2.806 | 0.603 | 0.596 |
| v5 after-dropout | 2.171 | 2.394 | 0.400 | 0.416 |
| v5 RPE (1 s, translation) | 0.317 | 0.356 | 0.172 | 0.176 |
| v5 orientation RMS (after the position-based alignment) | 22.1° | 53.0° | 6.1° | 7.1° |

Order B is harder on EuRoC because seven flights receive ~0.18 rad/s gyro noise and up to 0.13 rad/s gyro bias (§2.2); TUM-VI is unaffected by the order. The orientation row depends on the SE(3) rotation fitted to *positions*; when that rotation is ill-determined it can flip — V2_01 order A changes by up to 144° between two near-identical position solutions (E65) while the alignment-free rotational RPE stays at 1.77° per second. Read it as a diagnostic, not as attitude accuracy.

Per sequence (20 missions each, ATE; motionless | v4 with fixed gate | **v5**; v5 waypoints, blind, after-dropout):

| Sequence | Motionless | v4 (fixed) | **v5** | wp | b_abs | drop |
|---|---|---|---|---|---|---|
| MH_01_easy | 4.326 | 4.352 | **4.138** | 3.739 | 4.146 | 3.207 |
| MH_02_easy | 4.662 | 4.579 | **4.293** | 3.997 | 4.558 | 3.409 |
| MH_03_medium | 3.583 | 2.903 | **2.590** | 2.287 | 3.022 | 2.156 |
| MH_04_difficult | 6.811 | 5.246 | **4.922** | 4.992 | 3.155 | 4.108 |
| MH_05_difficult | 6.868 | 5.407 | **5.151** | 5.236 | 3.342 | 4.716 |
| V1_01_easy | 1.854 | 1.607 | **1.589** | 1.344 | 1.928 | 1.324 |
| V1_02_medium | 1.779 | 1.446 | **0.942** | 0.891 | 1.065 | 0.768 |
| V1_03_difficult | 1.582 | 1.605 | **1.001** | 0.795 | 1.364 | 1.028 |
| V2_01_easy | 2.299 | 2.208 | **2.146** | 2.020 | 2.884 | 2.114 |
| V2_02_medium | 2.113 | 1.693 | **1.330** | 1.104 | 2.459 | 1.320 |
| V2_03_difficult | 2.058 | 1.362 | **1.118** | 0.880 | 2.449 | 0.956 |
| room1 | 1.004 | 1.074 | **0.340** | 0.237 | 0.560 | 0.334 |
| room2 | 1.218 | 1.161 | **0.304** | 0.205 | 0.638 | 0.250 |
| room3 | 1.252 | 1.290 | **0.413** | 0.219 | 0.654 | 0.337 |
| room4 | 1.080 | 1.108 | **0.812** | 0.885 | 0.770 | 0.760 |
| room5 | 0.935 | 1.083 | **0.412** | 0.342 | 0.632 | 0.362 |
| room6 | 0.989 | 0.983 | **0.425** | 0.437 | 0.341 | 0.404 |

The Machine Hall flights (20–40 m paths in a large hall) remain hard for every IMU-only method: v5 is 4–28 % better than motionless there, 7–47 % on the Vicon-room flights and 25–75 % in the TUM-VI rooms.

### 4.2 Other conditions (E57, ATE; motionless | v4 fixed | v5)

| Condition | EuRoC | TUM-VI |
|---|---|---|
| IMU dies at 60 % of the recording (2 seeds) | 3.455 \| 3.088 \| **2.869** | 1.113 \| 1.157 \| **0.548** |
| IMU dies at 80 % (2 seeds) | 3.537 \| 3.067 \| **2.790** | 1.125 \| 1.175 \| **0.495** |
| Half the official severity (max-level 0.5) | 3.448 \| 2.900 \| **2.537** | 1.079 \| 1.133 \| **0.427** |
| 1.5 × the official severity | 3.449 \| 3.089 \| **2.828** | 1.080 \| 1.108 \| **0.492** |
| Clean control (official structure, no degradation) | 3.446 \| 2.568 \| **2.551** | 1.079 \| 0.414 \| 0.427 |

The clean control is the cost check: v5 equals v4 on clean EuRoC and is 3 % worse on clean TUM-VI (0.427 vs 0.414) — the price of the gap-aware, degradation-hardened front end on data that has no gaps. Gate errors: 0/578; guard fired: 0/578.

---

## 5. Competitor threat model and strongest baselines (identical protocol)

The strongest plausible opponents have premium coding agents, read the same papers and will try the obvious and the less obvious. Each row below was **reproduced under our identical protocol** (official script, same seeds, SE(3) alignment, same metrics) wherever it can be run without information the rules forbid.

| Approach a strong team would build | Scientific basis | Reproduced as | EuRoC ATE | TUM-VI ATE | Verdict |
|---|---|---|---|---|---|
| Ship "zero motion" | aligned constant beats drift | motionless | 3.448 | 1.080 | the floor; any method above it loses points |
| Naive strapdown (gravity removed, double integration) | textbook INS | E59, **with our attitude** | 25.9 | 29.9 | diverges quadratically — explodes, as slide 13 warns |
| Strapdown + zero-phase high-pass after each integration ("heave filter" for bounded motion), cut-off chosen per held-out environment | marine heave estimation (Reis 2023, SBG delayed heave, Tokta 2026) | E59, with our attitude; fc ∈ {0.005…0.2} Hz, LOEO choice | 3.263 | 0.550 | the strongest simple classical baseline; v5 is 20 % / 18 % better. In hindsight-tuned form (fc 0.05 Hz) TUM-VI 0.497 vs v5 0.453 |
| ESKF/EKF + RTS with ZUPT and a constant-velocity or white-noise prior | standard INS + smoothing | Phase 2 E10 (our own mission simulator, not the official script): estimators without a stationary prior | 12–108 m (waypoints) | — | without a stationary prior the low-frequency position is unobservable and drifts like strapdown |
| Our own Phase 2 v4 (Matérn / spring + drag + gate) | Phase 2 | E52 v4raw | 3.551 | 1.117 | breaks on the official dropout pattern (gate 95/153 errors); ≈ what a team gets with a per-segment filter design |
| v4 with the gate fixed and the blind tail handled | Phase 3 | E52 v4T | 2.946 | 1.117 | still loses 0.3 m / 0.67 m to v5: the dropout physics of §2.3 is the difference |
| Learned velocity regressor (TLIO/RoNIN-style 1-D CNN, 4 conv layers), trained on the organisers' degradation of all 17 public sequences (8 realisations each, seeds 100–102, max-level 0.5/1/1.5), fused into v5 as velocity pseudo-measurements | learned inertial odometry | E48 **in-pool** (the test sequence's motion was in the training set) | 2.870 (σ 1 m/s) | **0.388** (σ 1 m/s) | memorises the six rooms (−18 % on TUM-VI) but still **hurts** EuRoC (+7 %) |
| same, honest generalisation | — | E48 **leave-one-environment-out** | 3.100 | 0.790 | **strongly harmful**: +16 % / +66 % vs v5 |
| Learned model alone (no physics) | AirIO / TLIO family (L09–L11) | not re-implemented; Phase 1 found the published learned IMU-only results on the EuRoC Vicon rooms (3.1–4.2 m, clean data) above the motionless reference there (1.6–2.7 m), confirmed under AirIO's own ATE convention on three sequences in Phase 1B | — | — | needs PyTorch (import far above 8 MB) or a hand-written port; no published robustness to this kind of degradation; E48 shows the learned signal is weak here |
| Recognise the sequence and emit the public ground truth | — | not reproduced | — | — | ground truth in inference — forbidden by our own rules; would be disqualifying if detected |

E48 reference rows (same 88 / 48 missions): v5 2.672 / 0.476; network velocity RMSE in-pool 0.665 m/s (truth RMS 0.815) on EuRoC and 0.647 (truth 1.220) on TUM-VI; held-out 0.710 / 1.014. On the organisers' degradation, learned velocity barely beats "velocity = 0" on drones even when the network has seen the very trajectory. Caveat: TUM-VI velocity labels are a Savitzky–Golay derivative of the motion-capture positions (0.25 s window, applied across mocap gaps as well); they carry real information (the in-pool network gains 18 %) but may be noisy near gaps, so the TUM-VI in-pool threat is, if anything, under-estimated. The network (561 k parameters) was trained for 30 epochs on 158 k windows (in-pool) / 101–108 k windows (held-out folds); a larger model or longer training could memorise more.

**How an excellent opposing team could still beat us, and our position:**
1. **If the hidden test files are the public TUM-VI rooms**, a team that memorises them (in-pool network) can beat our TUM-VI numbers by ~18 %. Countering that requires memorising too, which loses 16–66 % if the test data are new; we do not do it and we say so (§11, §14). On EuRoC, memorisation did not help even in-pool.
2. **A better accelerometer-bias / tilt model for hand-held motion** — the error budget (§9) shows perfect tilt would gain 16 % on TUM-VI. No experiment of ours found a way to get it from the data (E41/E45/E49 variants: abias, sn_roll, priors).
3. **Engineering:** a faster, leaner implementation could out-score us on the 40 % efficiency share if ours is sloppy — §12 treats efficiency as a first-class requirement.
4. **Velocity information for drones** is the lever on EuRoC (§9: a 1 Hz velocity oracle would cut ATE by 64 %). The only IMU-only sources are rotor drag (used), bounded position (used) and learned regression (harmful here).

---

## 6. Component ablations (E58; full sequences, seeds 42 and 1, both orders: 44 EuRoC + 24 TUM-VI missions)

| Variant (one change) | EuRoC ATE | ΔATE vs v5 [95 % CI over sequences], sequences worse | TUM-VI ATE | ΔATE vs v5, sequences worse |
|---|---|---|---|---|
| **v5** | 2.607 | — | 0.453 | — |
| no gravity smoother (Phase 2 low-pass attitude) | 4.394 | **+1.787** [+1.338, +2.295], 11/11 | 1.944 | **+1.491** [+0.947, +1.993], 6/6 |
| no platform gate (all hand-held) | 3.253 | **+0.646** [+0.371, +0.997], 11/11 | 0.453 | 0 |
| all rotor, no guard | 2.607 | 0 | 32.04 | **+31.59**, 6/6 |
| all rotor, guard on | 2.607 | 0 | 0.468 | +0.015 [−0.002, +0.042], 4/6 |
| no rotor drag rows | 3.242 | **+0.635** [+0.369, +0.970], 11/11 | 0.453 | 0 |
| no T/2 s scaling of gap bias variance | 2.924 | **+0.316** [+0.087, +0.678], 10/11 | 0.719 | **+0.266** [+0.204, +0.342], 6/6 |
| linear (Phase 2) gap rotation law | 2.757 | **+0.150** [+0.041, +0.279], 8/11 | 0.972 | **+0.519** [+0.356, +0.666], 6/6 |
| no lever-arm states | 2.728 | **+0.121** [+0.004, +0.266], 9/11 | 0.453 | 0 |
| no zero-velocity rows | 2.659 | **+0.052** [+0.003, +0.121], 8/11 | 0.453 | 0 |
| constant gyro bias on rotors (no spline) | 2.654 | **+0.047** [+0.021, +0.074], 10/11 | 0.453 | 0 |
| k₂ = 1 (instead of 0.25) | 2.617 | +0.010 [−0.002, +0.023], 6/11 | 0.481 | **+0.028** [+0.014, +0.042], 6/6 |
| spline gyro bias always on hand-held | 2.607 | 0 | 0.468 | +0.015 [−0.002, +0.042], 4/6 |
| no repair · no dedup · no gyro bridge · no guard | 2.607–2.608 | ≤ +0.001 (n.s.) | 0.452–0.454 | ≤ +0.001 (n.s.) |
| full-rate smoother (no governor) | 2.607 | +0.000 [−0.034, +0.039] | 0.451 | −0.003 [−0.005, −0.000], 1/6 |

Spline iterations (E61): 12 or 20 passes instead of 6 change EuRoC ATE by −0.005 / −0.006 (order B, n.s.) and nothing elsewhere → **6 kept** (cheaper).

---

## 7. Robustness and generalisation

### 7.1 Selection discipline

| Decision | How it was made | Held-out evidence |
|---|---|---|
| k₂ (gap rotation law) | grouped CV over environment groups (E47): 0.1 (rotor groups) / 0 (TUM-VI); everything in 0–0.25 within 0.5 % | E46b gap-length sweep; fresh seeds E57 |
| Governor (block sizes) | E47 cost curve, all groups | E58 full-rate control |
| Guard ratio | E47 class ranges (EuRoC max 8.50, TUM-VI min 16.40) | E55b windows (§7.4) |
| Motion priors | E49 grouped CV (rotor CV choice +1.0 % worse than default; hand-held CV choice −1.2 % but worse on 4/6 rooms) → defaults kept | E57 |
| Blind rule | E53 leave-one-environment-out selection (rotor: cv_0.5 chosen in all three folds) | E57 fresh seeds (§8) |
| Gate threshold | moved into the empty class gap (E55) | no decision changes on any of the 884 missions/windows whose lock was recorded (E52, E55; E57 uses the E52 inputs) |
| Drift threshold (hand-held) | Phase 2 E16c on our simulator | official data: D ≤ 1.97 mrad/s on all 204 hand-held E52 missions (0 triggers; margin thin, but a false trigger costs only +0.015 m, E58) |
| Drag coefficient | per-flight regression (Phase 2) | every evaluation uses the leave-one-sequence-out value |
| Learned prior | rejected | E48 LOEO (§5) |

### 7.2 Seeds and column orders

Seed-to-seed coefficient of variation of a sequence's ATE (10 seeds, E57): median 5.1 % (max 15.1 %) on EuRoC, 4.4 % (max 8.1 %) on TUM-VI; group mean ATE over seeds 2.585–2.747 m (EuRoC) and 0.437–0.459 m (TUM-VI). Every headline comparison is made per sequence and bootstrapped over sequences, so seed noise cannot carry it. Both column orders are reported everywhere.

### 7.3 Combined and beyond-official corruption (E46, our stress suite: 17 sequences × 2 seeds, failure at 60 %, official zero-row blind format)

E46 ran the pre-freeze v5c (same front end and gap physics as v5; EuRoC column = its "spline always" variant, which is the frozen rotor rule; TUM-VI column = the gated hand-held rule as frozen; blind = smoother prediction; full-rate smoother).

| Condition | EuRoC v4 → v5 | TUM-VI v4 → v5 |
|---|---|---|
| clean | 2.61 → 2.66 | 0.42 → 0.44 |
| 3 × 2 s gaps (removed / zero / held encodings identical) | 3.03 → 2.88 | 0.85 → 0.85 |
| 6 × 2 s gaps | 3.30 → 3.01 | 0.99 → 1.00 |
| level L3 (heavy noise + bias drift + gaps) | 3.24 → 2.91 | 1.10 → 0.99 |
| L2 + gyro ramp 0.002 rad/s² (combined) | 3.44 → 3.11 | 1.59 → 1.09 |
| gyro ramp 0.002 rad/s² | 3.63 → 2.82 | 1.72 → 0.95 |
| gyro ramp 0.01 rad/s² | 11.67 → 3.39 | 7.28 → 1.56 |
| gyro ramp 0.05 rad/s² (extreme) | 16.78 → 6.85 | 13.35 → 8.01 |
| gyro bias random walk | 3.41 → 3.10 | 1.34 → 1.13 |
| spikes | 3.42 → 2.65 | 1.01 → 0.44 |
| gyro gain fault × 5 | 47.2 → 3.02 | 35.3 → 0.85 |
| accelerometer ramp | 2.80 → 2.81 | 0.79 → 0.79 |
| half sample rate (100 Hz) | 2.95 → 3.00 | 0.43 → 0.44 |
| timestamp jitter (σ 0.3 ms) | 2.62 → 2.68 | 0.42 → 0.44 |

0 gate errors in every condition. The rotor's "spline always" rule is what survives gyro ramps (EuRoC 0.002 rad/s²: constant bias 4.52, spline 2.82). The cost of robustness is visible on clean data (+2–5 %), consistent with the clean control of §4.2.

### 7.4 Platform gate, guard and the drift detector (E55, E55b)

Windows of every sequence (prefixes 15/25/40/60 %, suffixes from 30/60 %, middles 20–50 % and 50–80 %, full), both orders:

| Window | Duration | EuRoC lock min / median / max | EuRoC run as rotor | TUM-VI lock min / median / max | TUM-VI run as rotor |
|---|---|---|---|---|---|
| full | 125 / 135 s | 0.009 / 0.017 / 0.027 | 22/22 | 0.053 / 0.188 / 0.395 | 0/12 |
| middles, suffixes, 60 % prefix | 37–95 s | 0.007 / 0.014–0.018 / **0.032** | 110/110 | **0.041** / 0.11–0.21 / 0.43 | 0/60 |
| 40 % prefix | 50 / 54 s | 0.009 / 0.019 / 0.065 | 17/22 (3 by the guard) | 0.081 / 0.111 / 0.274 | 0/12 |
| 25 % / 15 % prefixes | 19–34 s | 0.009 / 0.024–0.054 / 0.156–1.0 | 12/22, 10/22 | 0.068–0.091 / … | 0/24 |

- **On every window that contains flight, the classes are separated by an empty gap (0.032, 0.041)**; the threshold now sits inside it (0.035) with 10 % margin to the rotor side and 17 % to the hand-held side. Gate errors on full, crop and level missions: 0/578 (E52) and 0/578 (E57).
- **The "errors" on short prefixes are the right answer.** The first 15–40 % of the Machine Hall recordings contain the pre-take-off phase (the vehicle on the ground / being handled). There the hand-held solution is far better than the rotor one (e.g. MH_01 15 % prefix: hand-held 0.23 m vs forced rotor 17.2 m); v5's choice is within 1.6 % of the per-window best on every prefix set.
- **The guard is an independent second lock.** Forcing the rotor branch on every TUM-VI window, the guard fires on 95/108 (all full, suffix and ≥ 34 s prefix windows: radius ratio ≥ 18 vs threshold 12); it misses some 20–40 s windows (ratio ≥ 2.0). On EuRoC flight windows it never fires (ratio ≤ 7.9). Together: a hand-held recording is mis-run as a rotor only if the lock is < 0.035 (never observed, minimum 0.041) **and** the guard misses (it caught every window of ≥ 54 s and every prefix of ≥ 34 s; it missed 13 of the 36 windows of 20–41 s).
- **Very short inputs** (< ~15 s incl. the blind window) have no complete 10 s lock window → lock = 1 → hand-held, the safe branch.
- **Drift detector:** official data D ≤ 1.97 mrad/s on hand-held (threshold 2.0): 0 false triggers in 204 missions, thin margin, cheap failure (+0.015 m, n.s.).

### 7.5 Timing irregularities, duplicates and gaps
- Duplicated rows (same timestamp, made by the script) share one slot and keep their own output row (E51: 49 duplicates in MH_01, rows in = rows out, evo accepts them).
- Timestamp jitter (σ 0.3 ms): ≤ +1 % against clean (E46); half the sample rate (100 Hz input) runs with every decision right and ATE within 2 % of v4 (E46). Four dropout encodings (removed / zero / held / NaN) give identical results (Phase 2 E12, E46).
- Recovery after gaps (the "dropout" component): v5 after-dropout error 2.282 m (EuRoC) and 0.408 m (TUM-VI) vs motionless 3.045 / 1.025 (E57); the two gap-physics fixes are worth +0.32/+0.27 m and +0.15/+0.52 m (E58).

### 7.6 Sensor-frame rotation (E60)
The same official file re-expressed in four rotated sensor frames: TUM-VI changes by ≤ 4 mm (invariant); EuRoC order A by ≤ 5.4 cm (median 9 mm). **Under the order-B gyro degradation (0.18 rad/s noise, up to 0.13 rad/s bias) the rotor estimate is sensitive:** the seven affected flights move by sd 0.1–0.3 m between frames (e.g. MH_05 B: 5.0–6.0 m), without a systematic penalty (mean over rotations 2.595 vs 2.614 unrotated). More spline iterations narrow the spread a little but do not change the mean (E61). This is variance of a hard, nonlinear problem, recorded as a known limitation; no gate or mode decision ever flips.

### 7.7 Long sequences
Not validated with ground truth beyond the public sequences (the longest is MH_01, 184 s; TUM-VI rooms up to 144 s). The algorithm is linear in the number of samples; the risk is memory, handled by design in §12.5 (budget sized for 10 minutes at 200 Hz).

### 7.8 evo cross-check (E51)
Six official missions written in the exact submission format, scored by the evo 1.38.1 CLI: `evo_ape tum gt est -a` equals our scorer to ≤ 0.4 mm (e.g. 4.04701 vs 4.04708 m; 0.87256 vs 0.87260 m); evo accepts the duplicated timestamps (de-duplicated copy differs by < 1 mm); format audit passes (rows = input rows, 8 columns, timestamps identical to the input, ‖q‖ error < 1e-7, all finite). Translation RPE over 200 frames agrees within 0.1 % on EuRoC and 1–2 % on TUM-VI; rotational RPE within 1 % except room4 (ours 6.10°, evo 4.84°): with duplicated timestamps and motion-capture gaps, frame-based pairing differs between implementations — the FAQ fixes only the ATE protocol, so our RPE figures are diagnostics. Without alignment evo gives 0.8–2.6 m larger errors (1.4–5.5 m absolute), which is why the FAQ's statement that SE(3) alignment is used matters.

---

## 8. The five blind seconds (priority investigation; E53, E57)

### 8.1 What the blind score is made of

Write the error of a blind row as **e_b = e_L + ε_b**: e_L is the (aligned) position error at the last valid sample, inherited from the trajectory before failure; ε_b is the error of the predicted displacement since then. A predictor controls only ε_b. Analogy: a sprinter's photo-finish position depends far more on where she was at the 95 m mark than on how well she guesses her last stride.

E53 measured both parts with two **diagnostic oracles** (never candidates): `ORACLE_disp` = the true displacement inserted after the estimated last pose (ε_b = 0), and `ORACLE_v` = the true velocity at failure, damped.

| Full sequences (E53, seeds 42, 1–4, both orders) | EuRoC b_abs | EuRoC b_anc | TUM-VI b_abs | TUM-VI b_anc |
|---|---|---|---|---|
| true 5 s displacement | — | 0.31 m | — | 0.40 m |
| hold (zero displacement) | 2.671 | 0.310 | 0.650 | 0.397 |
| smoother's own prediction (v5 default) | 2.696 | 0.595 | **0.591** | **0.373** |
| velocity at failure, 0.5 s decay (cv_0.5) | 2.671 | 0.332 | 0.631 | 0.416 |
| best realistic candidate | 2.649 (AR(5) on the velocity history) | 0.310 (hold) | 0.591 (model) | 0.372 (0.75 × model) |
| ORACLE true velocity, 1 s decay | 2.547 | 0.175 | 0.750 | 0.618 |
| **ORACLE perfect displacement** | **2.490** | 0.011 | **0.635** | 0.004 |

**Finding:** even a perfect predictor would improve EuRoC b_abs by only 6–8 % (2.490 vs 2.649–2.696) and TUM-VI b_abs not at all. The blind score is set by e_L, and e_L is large because the official severity ramp peaks exactly at the end of the file. The velocity at failure is barely observable: its mean error (0.41 m/s on EuRoC full sequences) exceeds the mean true speed (0.32 m/s). (The TUM-VI velocity oracle looks bad because our TUM-VI ground-truth velocity is a differentiated motion-capture signal that is unreliable as an oracle input; see §9.)

### 8.2 Candidates and selection

29 position candidates (hold; the smoother's prediction and its 0.25/0.5/0.75 shrinkage; damped constant velocity with τ = 0.25–2 s and ∞; ridge AR(5/10/20) on the 10 Hz smoothed velocity of the last 60 s, plain and damped; pull towards the trajectory centroid; equal-weight combinations) and 13 orientation candidates (hold; gyro or yaw-rate extrapolation over 1 s / 5 s windows with 0.5 s, 2 s or no decay) were scored on 340 missions: full sequences (5 seeds) and failure at 80 %, 60 % and 40 % of the recording (official zero-row format), both orders. **How the choice was made (before the confirmation seeds were run), per platform:** a candidate had to be within ~1 % of the best full-sequence b_abs (the official situation; b_abs is the most likely official definition — SE(3) alignment over all rows, then the blind rows) and close to the best full-sequence anchored error (b_anc guards against definitions that measure the prediction itself, anchored or relative); among those, lower regret when the IMU dies mid-recording, and the simplest rule when differences were within noise. This is a judgement over four criteria, not a single formula: on the rotor side "damped AR(5)/AR(10), 0.5 s" is statistically indistinguishable from the chosen rule and more complex. Independently, choosing by held-out anchored error under leave-one-environment-out picked the same rotor rule in all three EuRoC folds (E53). For hand-held data the provisional E53 choice weighted the mid-recording failures more and picked ½(model + AR5); the fresh seeds showed it significantly worse on full recordings (§8.3), so the final hand-held rule is the smoother's own prediction — the one the full-sequence criterion selects.

| Platform | Candidate | b_abs regret: full / mean of 4 / worst | b_anc regret: full / mean / worst | Qualified? |
|---|---|---|---|---|
| rotor | AR(5) | 0.0 / 1.3 / 2.5 % | 76 / 22 / 76 % | no (anchored significantly worse) |
| rotor | smoother prediction (v5 default) | 1.8 / 2.0 / 3.8 % | 92 / 26 / 92 % | no |
| rotor | **cv_0.5** | 0.8 / 3.1 / 5.6 % | **7.1 / 7.2 / 12.0 %** | **yes — chosen; also chosen by leave-one-environment-out selection in all three EuRoC folds** |
| rotor | damped AR(5), 0.5 s | 0.6 / 3.0 / 5.5 % | 8.5 / 7.5 / 12.0 % | yes (equivalent, more complex) |
| hand-held | **smoother prediction** | **0.0** / 7.9 / 22.1 % | **0.4** / 17.8 / 36.1 % | yes |
| hand-held | ½ (model + AR(5)) | 2.1 / 4.6 / 11.3 % | 6.5 / 14.5 / 24.0 % | yes (selected in E53, **rejected in E57**) |

### 8.3 Fresh-seed confirmation (E57: seed 42 + 5–13, the E52 protocol; paired over sequences)

| Change vs v5 default | Full sequences | Crops / severity levels |
|---|---|---|
| rotor cv_0.5 | b_abs −0.015 [−0.065, +0.044] (8/11 better); **b_anc −0.210 [−0.303, −0.111] (10/11)**; ATE ±0.000 | b_abs +0.037 [−0.078, +0.134]; b_anc −0.057 [−0.166, +0.062] |
| hand-held ½(model + AR5) + yaw extrapolation | **b_abs +0.019 [+0.002, +0.035] (1/6 better) — worse**; ori_b +2.3° | b_abs −0.045 [−0.098, −0.003] (5/6); ori_b −11.2° |

The hand-held combination wins only when the IMU dies mid-motion; on full recordings — the official situation — it is significantly worse, so **the hand-held blind rule stays the smoother's own Matérn prediction**. The rotor rule is adopted: neutral on b_abs and much better on the displacement itself.

### 8.4 Orientation in the blind window
Held orientation is best or within 1–2° on full sequences (EuRoC 41.4°, TUM-VI 26.3° RMS in the window); yaw-rate extrapolation helps only on mid-motion crops of the hand-held data. ATE is translation-only; rotation enters only RPE, where a held attitude means "no rotation" in the window. **Frozen: hold.** The large blind-window orientation errors are inherited yaw drift (yaw is unobservable for IMU-only; §9).

### 8.5 Frozen blind rule
Rotor: p(τ) = p_L + 0.5 s · (1 − e^(−τ/0.5 s)) · v_L. Hand-held: the smoother's prediction (its Matérn-5/2 prior conditioned on all data, which decays within ~1 s). Orientation: q_L. Result (E57, full): EuRoC b_abs 2.761 (motionless 3.083; −10.4 %), anchored error 0.316 (order A) / 0.339 (order B) vs 0.463 / 0.611 with the Phase 2 rule (the true displacement is 0.31 m); TUM-VI b_abs 0.599 (motionless 0.614).

---

## 9. Error budget: where the remaining error lives (E56, diagnostic oracles)

Ground truth injected on purpose into one part of the frozen pipeline at a time (full sequences, seeds 42 and 1, both orders):

| Substitution | EuRoC ATE | TUM-VI ATE |
|---|---|---|
| v5 | 2.605 | 0.453 |
| true attitude (roll, pitch, yaw) | 2.543 (−2 %) | 0.377 (−17 %) |
| true yaw only | 2.577 (−1 %) | 0.449 (−1 %) |
| true roll/pitch only | 2.581 (−1 %) | 0.382 (−16 %) |
| true velocity at 0.1 Hz (σ 0.1 m/s) | 1.884 (−28 %) | 0.426 (−6 %) |
| true velocity at 1 Hz (σ 0.05 m/s) | **0.932 (−64 %)** | 0.927 (worse: not a usable oracle, see below) |

**Drones: attitude is not the bottleneck — velocity is.** Even perfect orientation buys 2 %; a velocity fix every 10 s buys 28 %. This kills two tempting research directions (yaw observability through drag, better gravity alignment) and explains why rotor drag (the only IMU-only velocity signal) is the second most valuable component (+0.64 m when removed). **Hand-held: tilt is the lever (−16 %)**: residual gravity leakage from the accelerometer bias as the device rotates. These two lines are the honest "what would it take to do better" for the presentation. (EuRoC's ground-truth velocity comes from the dataset's state estimate; TUM-VI provides only motion-capture poses, and our loader's Savitzky–Golay derivative, also applied across mocap gaps, made the dense TUM-VI velocity oracle *harmful* — so the TUM-VI velocity rows say nothing about the true ceiling. The cause was not isolated.)

---

## 10. Score analysis per rubric component (no official score is claimed)

There is no published map from metres or milliseconds to points, so **no total is computed**. For each component: exact definition known to us, measured performance, remaining gap and risk.

| Component (weight) | Definition we can rely on | v5 measured (full official sequences, E57) | Strongest feasible improvement found | Remaining gap / risk |
|---|---|---|---|---|
| Waypoints (30) | positions at hidden waypoints after SE(3) alignment (FAQ); our proxy: 8 pre-failure checkpoints | EuRoC 2.480 m (−23.6 % vs motionless), TUM-VI 0.388 m (−65 %) | velocity information (§9) — no IMU-only source left untested | Machine Hall flights stay > 3.7 m for IMU-only; waypoint count/placement unknown |
| Blind 5 s (20) | rows of the last 5 s; absolute after alignment most likely | EuRoC 2.761 m (−10 %), TUM-VI 0.599 m (−2.5 %) | information-limited (§8.1): at most −6 to −8 % on EuRoC even with a perfect predictor | definition unknown (absolute vs anchored vs endpoint) — rule chosen to be good under all three |
| Dropout (10) | accuracy around the gaps; our proxy: 1 s after each gap ≥ 5 samples | EuRoC 2.282 m (−25 %), TUM-VI 0.408 m (−60 %) | done (two gap-physics fixes) | exact definition unknown; "+20 %" tier ambiguity |
| Peak memory (20) | tracemalloc-like heap of our data structures (FAQ) | research code not representative; Phase 4 target ≤ 4 MB at 184 s, ≤ 8 MB at 10 min; largest kernels measured ≤ 3.5 MB | §12.5 plan | **`import numpy` alone 7.5–8.1 MB traced** if their window includes imports (A3) |
| Runtime (20) | ≤ 20 ms per IMU-second, one thread | measured-kernel cost model (E64): ≈ 7.9 / 6.9 ms per IMU-second (rotor / hand-held path) on the M3 Pro, ≈ 15.6 / 15.5 ms on a 2.1 GHz Xeon cloud vCPU | §12.6 plan | grader CPU unknown (A6); fixed start-up cost (`import numpy` 0.09–0.31 s) weighs on short files |

Expected-value view: the closeness share is already near what IMU-only physics allows under this degradation (the alternatives in §5 are worse); the largest remaining *controllable* risk to the total is efficiency (40 %), which Phase 4 must engineer and measure first-class.

---

## 11. Innovation and prior art (targeted; no established algorithm is called novel)

Register IDs refer to `EVIDENCE_REGISTER.md`. "Closest prior art" is the nearest thing we found, not a claim that nothing closer exists; no patent professional has assessed anything here and no patentability is claimed.

| v5 element | Closest prior art | What is established | Our distinction (evidence) | Honest classification |
|---|---|---|---|---|
| Accelerometer as a *measurement* of a continuous-time stochastic motion state, batch-smoothed | Burnett, Schoellig, Barfoot 2024 (L28: IMU-as-measurement with a white-noise-on-jerk GP prior); Barfoot et al. GP regression (P1B-09); Talbot et al. survey (P1B-06); Särkkä state-space GPs | the formulation | IMU-only, stationary priors (Matérn-5/2, spring-damper) so that unobservable drift is bounded; validated on the organisers' degradation (§4) | application of an established formulation |
| Stationary / bounded-motion prior for IMU-only position | marine heave filters: Godhavn 1998, Küchler 2011 (P1B-13), Reis et al. 2023 (P2-01), Auestad et al. (P2-02), SBG "delayed heave" (P2-03), Tokta 2026 (P1B-07); patents US 6,382,022, US 6,201,763 (P1B-15, P2-04) | bounded vertical motion from a biased accelerometer, incl. non-causal ("delayed") smoothing | 3-D, room-scale, two platform-specific priors; E59: a tuned heave-style high-pass integrator with our attitude is 20 % / 18 % worse than v5 under held-out tuning | known principle, new setting |
| Rotor-drag velocity rows | Martin & Salaün 2010 (P2-05), Leishman et al. 2014 (L24), Abeywardena et al. (L25), Faessler et al. 2017 (L27); patent US 11,525,683 (P2-06); "aerodynamics-inspired IO" needs rotor speed (P2-08) | drag ∝ body-frame velocity for multirotors | used IMU-only inside a batch smoother with a spring prior and **online lever-arm states** (EuRoC lever arm (0.020, 0.016, 0.099) m, sd 3–9 mm over 11 flights, Phase 2 E15); E58: drag is the second most valuable component | established model, careful integration |
| Gap-length-aware uncertainty (rotation θ = min(k₂T²/2, 1); bias variance ∝ min(1, T/2 s)) | discretised Wiener-acceleration and random-walk models (any filtering text) | the variance laws themselves | **diagnosis** that the official dropout pattern (thousands of 5 ms drops) breaks a ‖ω‖·T reset and a per-gap full-variance reset, with measured cost (+0.52 m, +0.27 m on TUM-VI, E58) | standard theory applied correctly where others are likely to get it wrong |
| IRW gravity smoother in the gyro-integrated frame | complementary filters (Mahony et al.), VQF (L03), offline tilt smoothing (Phase 1 hypothesis list) | gravity-direction filtering | zero-phase, gap-aware, with the regressor-matched spline bias fit; E58: the most valuable component (+1.8 m / +1.5 m when removed) | engineering of known pieces |
| Spline gyro bias from the tilt-correction drift | continuous-time bias splines (B-spline batch estimation, e.g. Furgale et al.; Kalibr) — from memory, not re-read in Phase 3 | spline bias models | the matched-regressor detail (otherwise LS under-estimates bias when the body rotates; room1 ramp tilt 3.0° → 0.6°) | known model, implementation insight |
| Platform gate (thrust-axis lock) + divergence guard | UAV-type classification from flight data with learned models (arXiv 2403.00565); thrust-axis identification (arXiv 2409.01080) | classifying platforms from IMU data | a two-line physical statistic and a radius guard, validated on 1,000+ missions and windows (§7.4); we found no IMU-odometry method that switches motion priors on such a statistic (one search pass — absence of evidence only) | simple, defensible engineering; not claimed as an algorithmic contribution |
| Blind window: velocity shrinkage | Singer model (1970), damped-trend forecasting, constant-velocity baselines (L30) | damped extrapolation | the decomposition result (§8.1: the inherited error dominates; even a perfect predictor gains ≤ 8 %) and a fresh-seed-confirmed rule | empirical finding |
| Evaluation method | Zhang & Scaramuzza 2018 (L32) alignment conventions; Tartan-IMU zero-submission normalisation (L23) | ATE/RPE, trivial baselines | organiser-script-faithful protocol, both column orders, motionless reference, grouped CV, fresh-seed confirmation, oracle error budget, learned-model red team | rigour, not novelty |
| XR context | IMU-only fallback and pose prediction are claimed in XR patents (P2-10: US 12,380,658, US 10,810,747, US 11,144,117, US 12,717,150; assignees unverified) | inertial fallback when tracking is lost | — | relevant to Samsung's interest; nothing claimed |

**What we can defend in front of Samsung's XR Perception Group:** not a new algorithm, but (1) the evidence that the organisers' own degradation breaks the obvious designs (Phase 2 v4 *and* naive filters), with the two root causes and their principled fixes; (2) a platform-conditioned smoother whose every component is justified by a leave-one-out ablation; (3) an honest map of the limits — velocity, not attitude, bounds drone accuracy; the blind score is set before the IMU dies; learned priors memorise rather than generalise under this degradation. These are reproducible and checkable, which is the point.

**IP note.** The organisers require a public GitHub repository (C21). Publishing it (Phase 5) discloses everything in this document. Nothing here looks patentable to us, but that is not a professional opinion; the publication decision is the user's, at Phase 5.

---

## 12. Phase 4 implementation blueprint

Phase 4 builds **one standalone `main.py` (Python 3.9+, NumPy only)** that reproduces `fbs6.estimate_final(..., cfg=fbs6.V5P4, blind=blind5.final_blind)` within the tolerances of §12.8 and meets the efficiency targets of §12.5–12.6. Nothing in the research harness (SciPy, Numba, pandas) may be imported at inference.

### 12.1 Repository layout (Phase 5 publishes it)

```
main.py                    the entry point: everything needed at inference, NumPy only
requirements.txt           numpy>=1.21
Dockerfile                 python:3.12-slim + numpy; ENTRYPOINT ["python", "/app/main.py"]
README.md                  problem, method, results (tables of §0/§4), exact run commands, Docker, evaluation, limitations, licences
metadata.json              measured peak heap (with and without the numpy import), runtime per IMU-second, machine, versions
tools/make_samples.py      downloads nothing; applies the organisers' degrade_imu.py (unmodified, referenced by SHA-256) to local
                           EuRoC / TUM-VI copies, both column orders
tools/evaluate.py          SE(3) Umeyama ATE + waypoint / blind / dropout proxies (= research p4.score), evo cross-check
tools/measure.py           tracemalloc peak (post-import and from start) and wall time per IMU-second, single thread
tools/check_output.py      the output contract (§12.8 T1)
tests/                     contract, regression-vs-reference, edge-case and robustness tests
research/                  Phase 1–3 harness and experiments (reproducibility of this report); NO dataset files (EuRoC licence)
```

### 12.2 Command-line contract (A2: the grader's invocation is unknown, so accept every reasonable one)

- `python main.py INPUT [OUTPUT]`, also `-i/--input`, `-o/--output`. INPUT may be a `.zip` (search members for `*/imu0/data.csv`, else any `data.csv`, else the only `.csv`), a directory (same search), or a CSV file. No INPUT: search the working directory the same way.
- OUTPUT: a file path, or a directory (→ `DIR/estimated_trajectory.txt`); default `./estimated_trajectory.txt`. Print the path written; exit code 0.
- **Never exit without a valid file**: any exception in estimation falls back (v5 → hand-held branch only → motionless with gravity-levelled orientation), still writing exactly one pose per input row. Fallbacks are logged to stderr.
- Before `import numpy`: set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS` to 1 (C27). No threads, processes, network or randomness.

### 12.3 Input and output specification

- **Parse** in chunks of ~2^18 bytes: skip blank lines and lines whose first non-space character is `#` or a letter; split on commas (tolerate spaces, CRLF); timestamp field → Python `int` → int64 (never via float: 19-digit nanoseconds lose precision as float64); six values → float (scientific notation, `nan`, empty → invalid row). Store values as float32 (6 × 4 B) and timestamps as int64.
- Column order EuRoC (t, ω_xyz, a_xyz) per s15. Defensive unit guard: if the median |a| of valid rows is < 2 (g units) multiply by 9.80665; if the median |ω| is > 20 (deg/s) multiply by π/180 (neither occurs in the official data; logged if triggered).
- **Output** in input order, one line per input row, duplicates included: `"%d.%09d %.6f %.6f %.6f %.7f %.7f %.7f %.7f\n"` from (ns // 1e9, ns % 1e9), position (m), unit quaternion (qx, qy, qz, qw), Hamilton convention, body-to-world, world frame gravity-aligned (z up). Optional header `# timestamp tx ty tz qx qy qz qw`. Single spaces only (evo's TUM reader splits on single spaces).
- **Renderer:** the vectorised fixed-point ASCII renderer of E62b (digits computed with integer arithmetic into a byte matrix, pad bytes removed) — **byte-identical to `%`-formatting on all 35,629 rows tested**, 0.15 ms per IMU-second on the M3 Pro vs 2.85 ms for `%`-formatting; run it in chunks of ≤ 4,096 rows to keep its buffer ≤ 0.5 MB.

### 12.4 Algorithm, module by module (reference = research function; port faithfully)

| Module | Reference | Specification for `main.py` |
|---|---|---|
| M1 ingest | `fbs6.ingest2(held_run=2, dedup=True)` | slot Δ = median of positive timestamp differences; slot index = round((t − t₀)/Δ); rows with equal timestamps share one slot (first valid row's values); a slot is invalid if missing, all-zero, NaN, or an exact repeat of the previous slot's 6 values (first sample of a stuck run kept); keep `rows` (input row → slot) |
| M2 blind split | `fbs6.split_tail` | L = last valid slot; n_tail = N − 1 − L; estimation uses slots 0…L |
| M3 repair | `fbs6.repair(g_lo=5.0, g_hi=15.5)` | ‖a‖ plausibility windows → invalid; legacy despike floors (2e-3 rad/s, 2e-2 m/s²) — port as is |
| M4 platform statistics | `fbs6.thrust_lock_fill`, `fbs3.rest_info` | fill gaps (linear interpolation), ONE zero-phase 2nd-order Butterworth low-pass at 1 Hz over the whole stream — implemented as \|H(f)\|² in the frequency domain after odd-extension padding, one column at a time (E64c: identical to a forward-backward biquad to 2e-12 in the interior, 0.02 / 0.04 ms per IMU-second on M3 / Xeon; the research code's SciPy `sosfiltfilt` differs only within ~2 s of the ends, which a median over 10 s windows absorbs — T2 checks the decisions); 10 s windows with 5 s hop, median of RMS(f⊥)/g; rest windows (0.5 s, quietest 5 %, gyro-std gate 0.035 rad/s, ≥ 3 windows, consistent gyro means) → ZUPT mask and gyro-bias anchors |
| M5 gate | `V5["gate"]` | rotor ⇔ lock < 0.035; fewer than one complete window → lock = 1 (hand-held) |
| M6 gyro bridge | `fbs5.bridge_gyro` | OU interpolation across gaps, τ = 0.4 s, mean over ±5 s |
| M7 strapdown | `fbs.integrate_gyro` | log-depth quaternion prefix scan in chunks of 4,096 (E64c: 0.063 / 0.136 ms per IMU-second per pass on M3 / Xeon) |
| M8 gravity smoother | `fbs6.gravity_smooth_dec(d=16)` (= `fbs5.gravity_smooth` on block means; quad law, k₂ = 0.25, τ = 3 s, σ 1.5) | **at 12.5 Hz** (blocks of 16 slots): block means of valid samples (measurement variance 1/count), continuous intensity dt₀/τ⁴ in the original Δt, gap jumps (9.81·min(k₂T²/2, 1)/σ)² summed into the block that ends the gap, smoothed series linearly interpolated back to the slots (E63/E63b: per-mission ATE change ≤ 4 mm, orientation ≤ 0.05°, 0 gate changes on 68 missions; E65: part of V5P4). The gain and RTS-coefficient recursion depends only on the grid, the mask and the jumps: compute it **once per input and per noise setting** (two settings: σ 1.5 for the gravity estimate, σ 1.5/9.81 for the matched regressors) in pure-Python floats, cache it, and run every pass's mean recursions as vectorised affine scans over all columns (E64c: exact to 3e-14; per call 0.056 / 0.082 ms per IMU-second on M3 / Xeon). Merge the two regressor calls of a spline iteration into one 6-column call |
| M9 attitude passes | `fbs6.fe_const` (`attitude5b`, 2 bias iterations), `fbs6.fe_drift`, `fbs6.fe_spline` (`fbs5.attitude6`, 20 s knots, λ₂ = 3, λ₀ = 0.05, ≤ 6 passes, tol 3e-4, matched regressors) | health = 1st percentile of ‖g_I‖/g; coarse PI start (`fbs5.coarse_bias`, Mahony forward-backward) only if health < 0.9 (1 of 374 rotor missions in E52) |
| M10 bias rule | `front_final` | rotor → spline; hand-held → constant / rest-anchored unless D > 2 mrad/s or health < 0.9 → spline |
| M11 back-end inputs | `fbs6.prep_from` | a_f (gap-filled), ω_c (bridged − bias), ω_l (15 Hz zero-phase Butterworth of ω_c by FFT, as in M4; used only in the lever-arm terms of the drag rows), a_w = R(q)a_f − mean(valid) |
| M12 models | `fbs.model_matern52(1, 2, 20, 0.1)`, `fbs.model_spring(20, 2, 0.3, 1, 20, 0.1)`, `fbs.discretise` | Van Loan discretisation with a NumPy `expm` (scaling and squaring); stationary P₀ from the Lyapunov equation via the Kronecker vec trick (16 × 16 solve) |
| M13 measurements | `fbs4.build_joint`, `fbs4.block_average` | world-acceleration rows (σₙ = high-frequency std of a_w), drag rows in flight (`fbs.flight_mask`, `fbs.thrust_frame`, k = 0.1873 s⁻¹, σ 1.4, lever terms from ω_l), ZUPT rows (0.02 m/s), bias variance 0.25·min(1, T/2 s) at gap ends; averaged in blocks of 8 (rotor, V5P4) / 2 (hand-held) samples |
| M14 smoother | `fbs3.kf_rts_seq2` | Kalman filter + RTS with √K checkpoints (memory O(√K)), the **tight form of E64**: fixed-shape joint update (rotor: 5 rows — 3 accelerometer + 2 drag — or 8 with the ZUPT rows; inactive rows zeroed with R = 1, z = 0, which is exact); hand-held: one 4 × 4 covariance shared by the three axes, scalar update when only the accelerometer row is active; RTS mean via a vector solve D = P_pred(k+1)⁻¹ (x_smooth(k+1) − x_pred(k+1)), x_smooth(k) = x_filt(k) + P_filt(k) Aᵀ D (no gain matrix); **symmetrise P after every joint update** (without it the skew part grows ×1.4 per step and the filter diverges within ~200 steps, E64 debug). Rates (V5P4): rotor 25 Hz (blocks of 8), hand-held 100 Hz (blocks of 2). Per-step measurement vectors built on the fly (precomputing them costs ~1 MB per 10,000 steps). E64: identical to the reference kernels to 3e-14 m |
| M15 full-rate positions | `fbs4.hermite` | cubic Hermite from smoother-rate (p, v) to slots |
| M16 guard | `fbs6.radius`, ratio 12 | on rotor inputs the hand-held solution is needed only for its radius: run it at 25 Hz (blocks of 8); if the guard fires (never on the 1,156 official missions of E52/E57), run the full 100 Hz hand-held solution and use it. T2/T4 verify that the 25 Hz radius gives identical guard decisions |
| M17 blind | `blind5.final_blind` | rotor: p_L + 0.5 s·(1 − e^(−τ/0.5 s))·v_L (v_L from the smoothed velocity at the last valid slot); hand-held: the smoother's prediction; orientation q_L |
| M18 output | §12.3 | rows → slots map; renderer |

### 12.5 Memory plan (heap as tracemalloc measures it)

Measured facts (re-measured on 21 Nov, repeated runs): `import numpy` alone, with tracing started just before it, leaves **7.48 MB** traced on macOS (NumPy 2.5.4, Python 3.12.7; transient peak 7.55 MB) and **8.06 MB** on Linux (NumPy 2.5.3, Python 3.13; peak 8.13 MB); inside the E50 probe, which imported `zipfile` and other standard-library modules before tracing began, it was 7.32 MB. A bare interpreter under `-X tracemalloc` holds 1.88 MB (macOS) / 2.24 MB (Linux), and 9.24 / 10.07 MB once NumPy is imported. If the organisers' window includes imports, every NumPy solution exceeds 8 MB on its own; their FAQ answer (data structures, buffers, estimation logic, not the runtime) suggests it does not. We therefore budget **our own heap** and report both numbers.

| Structure (MH_01: 35,629 rows, 36,820 slots; rotor 4,600 steps at 25 Hz) | Bytes per element | Size |
|---|---|---|
| **persistent for the whole run:** timestamps (int64) + row → slot map (int32) | 12 per row | 0.43 MB |
| raw ω, a (float32; a overwritten by its gap-filled copy) | 24 per slot | 0.88 MB |
| valid / ZUPT / flight masks (bool) | 3 per slot | 0.11 MB |
| attitude q (float32, full rate) | 16 per slot | 0.59 MB |
| *persistent subtotal* | ~55 per slot | **2.0 MB** |
| front end (transient): gravity block series (9 columns) + cached gains (2 settings) at 12.5 Hz, chunked rotations and block sums | — | ≤ 0.9 MB |
| back end (transient): measurement vectors at the smoother rate, √K checkpoints + one segment of states and covariances, smoother-rate (p, v) | — | ≤ 1.9 MB (E64 measured 1.9 MB for the rotor kernel at 25 Hz *with* all per-step inputs precomputed; built on the fly ≈ 1.2 MB) |
| output (transient): full-rate positions (float32) + one renderer chunk of 4,096 rows | — | ≤ 0.9 MB |
| **target peak = persistent + the largest transient** | | **≤ 4 MB** (≈ 3.9 MB worst case) |

Rules: no full-rate float64 3-column temporaries alive at the same time (compute rotations and block sums chunk-wise); `del` stage outputs as soon as consumed; affine scans at 12.5 Hz peak at 0.8–1.4 MB (E64c) — chunk them (1,024 steps with carry) if a longer input needs it; FFT low-passes one column at a time (all three at once peak at 7 MB, E64c). **Long inputs:** the smoothers need only O(√K) memory, so the persistent ~55 B per slot dominates: ≤ 6 MB up to ~6 min at 200 Hz, ≤ 8 MB up to ~9 min; beyond that drop the full-rate q (recompute it chunk-wise from the bias track for the output; −16 B per slot, ~12 min). Inputs much longer than that cannot be held in 8 MB by any read-everything design (raw float32 plus timestamps alone is 32 B per row); the FAQ's "read the entire CSV into memory (while strictly staying under the 8 MB limit)" implies the evaluation files are short enough, and the public candidates (EuRoC, TUM-VI rooms) are ≤ 3.1 min.

### 12.6 Runtime plan and measured cost model (≤ 20 ms per IMU-second on one thread)

Every kernel of the V5P4 design was timed in the form §12.4 prescribes, single-threaded, on two machines: an Apple M3 Pro (macOS, Python 3.12.7, NumPy 2.5.4) and a shared 2.1 GHz Xeon cloud vCPU (Linux, Python 3.13, NumPy 2.5.3) — the latter as a pessimistic stand-in for an unknown grader. Pass counts come from instrumenting the research pipeline on official missions (E64b: MH_01 order B, the worst rotor case, needs 22 gravity-smoother calls / 66 columns, 10 strapdown passes, 3 gyro bridges, 3 low-passes; room2, a typical hand-held case, needs 6 calls / 18 columns and 6 passes). Units: ms per IMU-second (the official budget is 20).

| Stage | Rotor path (MH_01 B), M3 / Xeon | Hand-held path (room2 A), M3 / Xeon | Source |
|---|---|---|---|
| parse (chunked, int64 timestamps) | 0.53 / 1.18 | 0.53 / 1.18 | E50 on the official-format zip |
| timeline, masks, repair, rest windows, gap fill | ~0.1 / ~0.2 | ~0.1 / ~0.2 | E50 (timeline 0.01 / 0.05) + estimate |
| gyro bridges (3) | ~0.1 / ~0.2 | ~0.1 / ~0.2 | estimate |
| strapdown passes (10 / 6) | 0.63 / 1.36 | 0.38 / 0.82 | E64c (0.063 / 0.136 per pass) |
| rotations + block means per gravity call (22 / 6) | 0.44 / 1.10 | 0.12 / 0.30 | E50 |
| gravity smoother, 12.5 Hz: gains once (2 / 1 settings) + calls (22 / 6) | 1.74 / 2.70 | 0.56 / 0.93 | E64c |
| bias least squares, tilt-drift regressors | ~0.2 / ~0.4 | ~0.1 / ~0.2 | estimate |
| low-passes by FFT (3 / 2) | 0.06 / 0.12 | 0.04 / 0.08 | E64c |
| position smoother (rotor 25 Hz / hand-held 100 Hz), tight form | 2.63 / 4.99 | 4.70 / 11.14 | E64 |
| guard: hand-held solution at 25 Hz | ~1.2 / ~2.9 | — | E64 (half of the 50 Hz cost) |
| Hermite, blind rows, quaternions | ~0.1 / ~0.2 | ~0.1 / ~0.2 | estimate |
| output (vectorised renderer, byte-identical to %-formatting) | 0.15 / 0.26 | 0.15 / 0.26 | E62b |
| **total** | **≈ 7.9 / ≈ 15.6** | **≈ 6.9 / ≈ 15.5** | |

Why these rates: the research code runs the gravity smoother at the full 200 Hz (its 22 calls would cost ~20 ms per IMU-second on the M3 alone in NumPy) and the rotor smoother at 50 Hz (with the gravity smoother already at 12.5 Hz the rotor path would still cost ≈ 10.3 ms on the M3 and ≈ 19.8 ms on the Xeon — no margin on a slow grader). Moving the gravity smoother to 12.5 Hz costs nothing measurable (E63b) and the rotor smoother to 25 Hz costs +0.5 % EuRoC ATE, n.s. (E65). The hand-held smoother stays at 100 Hz (50 Hz would cost +2.5 % TUM-VI ATE, E47). Measured kernel speed-ups that made this possible: gains-once + affine scans for the gravity smoother (×4 vs a per-step NumPy loop), the tight smoother form (rotor ×1.5–1.6, hand-held ×1.1), the vectorised renderer (×19). Rows marked "estimate" are vectorised NumPy steps that E50 suggests cost well under 0.5 ms each; Phase 4 T5 measures the whole program.

Not included: interpreter start-up and `import numpy` (0.09–0.17 s on the M3, 0.24–0.31 s on the Xeon, over repeated runs). For a 184 s input that adds 0.5–1.7 ms per IMU-second; for a 30 s input 3–10 ms. Whether the organisers' timer includes it is unknown (U2).

**Accuracy-for-speed ladder** (apply in order only if T5 misses a target; each step pre-validated on official data): (1) spline passes 6 → 4 (E61b: +0.007 m EuRoC order B, n.s.; 0 elsewhere); (2) gravity smoother 12.5 → 6.25 Hz (E63b: per-mission |ΔATE| ≤ 4 mm); (3) guard radius at 12.5 Hz; (4) hand-held blocks 2 → 4 (E47: +2.5 % TUM-VI ATE) — last resort.

### 12.7 Failure handling

| Situation | Behaviour |
|---|---|
| fewer than 10 valid slots, or every slot invalid | motionless positions; orientation from the mean specific force (levelled) or identity |
| no trailing invalid run | n_tail = 0, nothing predicted |
| non-increasing timestamps | estimate on a stably time-sorted copy; output in input order with the original timestamps |
| a gap longer than 60 s | allowed: the bias variance cap (T/2 s ≥ 1 → full reset) and the stationary prior bound the drift; the gravity jump is capped at 1 rad |
| exception or non-finite value in a stage | fall back along v5 → hand-held only → motionless, per §12.2; NaN check before writing |
| very short input (< 15 s) | lock = 1 → hand-held branch (safe), spline needs ≥ 50 regression points else constant bias |

### 12.8 Tests and acceptance criteria (all must pass before Phase 5)

- **T1 contract** (34 official files: 17 sequences × orders A/B, seed 42; plus T7 edge cases): rows out = rows in; 8 single-space-separated columns; timestamp strings equal `"%d.%09d"` of the input nanoseconds; |‖q‖ − 1| < 1e-6; all finite; `evo_ape tum gt est -a` runs.
- **T2 regression vs the research reference** (`fbs6.estimate_final(..., cfg=fbs6.V5P4, blind=blind5.final_blind)`, same 34 files): identical gate, guard and bias-mode decisions (34/34); per-file ATE within max(1 %, 1 cm); mean ATE within 0.5 %; blind-window rows within 1 cm of the reference.
- **T3 official protocol** (seed 42 + 5–13, both orders, 340 full missions or a stratified 102-mission subset): group means within 1 % of E65's V5P4 values (EuRoC 2.670 m, TUM-VI 0.451 m).
- **T4 robustness**: E46 stress subset, E55 windows and E60 rotations run without error, decisions identical to the reference.
- **T5 efficiency**: peak traced heap after `import numpy` ≤ 4 MB on MH_01 (both orders) and ≤ 8 MB on a 10-minute synthetic input; runtime (excluding the import) ≤ 9 ms per IMU-second on the M3 Pro and ≤ 17 ms on the Linux Xeon 2.1 GHz cloud vCPU for MH_01 order B (rotor path, worst case) and room2 (hand-held path), i.e. within ~10 % of the §12.6 model; if missed, profile first, then apply the §12.6 ladder and re-run T2/T3.
- **T6 cold start**: fresh `python:3.12-slim` container + `pip install numpy`, the README command on a zipped official-format input produces a file that passes T1.
- **T7 edge cases**: header variants, CRLF, trailing blank lines, spaces after commas, scientific notation, zip with a nested folder, no blind window, 1-row, 2-row and all-zero inputs, duplicated timestamps, NaN fields.

### 12.9 Release items (Phase 5, for completeness)
Public GitHub repository (mandated by S3), README with the tables of §0/§4 and the limitations of §14, `metadata.json` from `tools/measure.py`, `team_name.zip` bundle per s17 (estimated trajectory for the official-format sample produced by our tools, `metadata.json`, `src/`), presentation `Thapar_FiveBlindSeconds` on the 9-slide template, demo video 2:30–2:55, Google Form. Dataset files are not redistributed (EuRoC non-commercial licence); both datasets and the organisers' script are cited.

### 12.10 Build order
M1–M2 + M18 first (a motionless-but-contract-perfect `main.py` passes T1 on day one) → M3–M11 front end, checked stage by stage against the reference (tilt and bias traces) → M12–M15 hand-held, then rotor → M16–M17 → T2/T3 → efficiency engineering (T5) → T4/T7 → Docker (T6) → tag v1.0.

---

## 13. Phase 3 acceptance gate (checked honestly)

| # | Gate item (brief §9) | Status | Evidence | Not yet verified / consequence |
|---|---|---|---|---|
| 1 | Complete understanding and mapping of the published requirements | **Met** | §1: 27 requirements with exact wording, source and status; precedence S5 > S3 > S2 > S1; conflicts C10, C11 resolved, C18 preserved | ambiguities A1–A7 cannot be resolved without contacting the organisers (forbidden); each has a designed response |
| 2 | Serious competitor threat model and strongest feasible baselines | **Met** | §5: nine approaches, seven reproduced under the identical protocol, incl. a learned velocity network trained on the organisers' degradation, in-pool and leave-one-environment-out | a larger or longer-trained network could memorise better than ours in-pool (E48 is a lower bound for memorisation) |
| 3 | Component-level evidence for retaining or changing v4 | **Met** | §2.3 root causes; §6 leave-one-component-out on 68 official missions; §3.3 provenance of every parameter | — |
| 4 | Meaningful investigation of the five-second component | **Met** | §8: 29 position + 13 orientation candidates, 340 missions, oracle decomposition, LOEO selection, fresh-seed confirmation; rotor rule changed, hand-held rule kept on evidence | official blind metric definition unknown (A5) |
| 5 | Stronger generalisation and combined-degradation validation | **Met** | §7: grouped CV for every tuned choice; 10 fresh seeds; both column orders; failure at 40–100 %; severity ×0.5–1.5; 29-condition stress suite incl. combined corruption; 306 gate windows; sensor rotations; evo cross-check | only 4 environments and one hand-held environment; no ground-truth validation beyond 184 s |
| 6 | Defensible innovation and prior-art assessment | **Met** | §11: closest prior art for every element; no established algorithm called novel | Phase 3 prior-art checks were at abstract/snippet level; no professional patent search |
| 7 | Score-focused architecture addressing every official component | **Met** | §10 per component; §8 blind; §7.5 dropout; §12.5–12.6 efficiency, including the rate changes of V5P4 chosen for runtime and validated for accuracy (E65) | efficiency not measured on the final build (Phase 4 T5) |
| 8 | Precise, resource-feasible Phase 4 blueprint | **Met at kernel level; end-to-end measurement is Phase 4 T5** | §12: module-level specification with reference functions (`fbs6.V5P4`), I/O contract, memory budget, a cost model built from kernels measured in their prescribed form on two machines (E50, E62, E62b, E64, E64c) and pass counts of the real pipeline (E64b): ≈ 8 ms per IMU-second (M3 Pro) / ≈ 15.6 ms (2.1 GHz Xeon vCPU) worst case, against a 20 ms budget; rate changes validated for accuracy (E63b, E65); pre-validated speed ladder; tests T1–T7 | several small stages are estimates (marked in §12.6); glue code and Python overheads of the real program are unmeasured; the organisers' heap window (A3) and machine (A6) are unknown |
| 9 | No unresolved fundamental research or architecture decision | **Met** | every scientific and architectural choice is frozen (§3.3, §8.5) and the submission configuration is fixed (`fbs6.V5P4`, §0, E65); what remains for Phase 4 is implementation, bounded by T2/T3 tolerances and by the pre-validated ladder if T5 misses | — |

---

## 14. Unverified items, risks and their consequences

| ID | Item | Consequence if it goes against us | What we do |
|---|---|---|---|
| U1 | Whether the organisers' tracemalloc window includes `import numpy` (7.5–8.1 MB on its own) | every NumPy entry exceeds 8 MB; the memory points are lost for everyone using NumPy | report both numbers; keep our own heap ≤ 4 MB; a pure-Python fallback cannot meet the runtime budget (17-state smoother ~324 ms per IMU-second in pure Python, Phase 3 probe) |
| U2 | Grader CPU, and whether start-up and `import numpy` are timed | runtime points; on short files the fixed 0.09–0.31 s import weighs heavily (30 s file: +3–10 ms per IMU-second) | cost model ≈ 15.6 ms per IMU-second on a 2.1 GHz Xeon vCPU worst case; ladder in §12.6; the import cannot be avoided by any NumPy solution |
| U3 | Column order the organisers used (A/B) | order B costs ~6 % EuRoC ATE (2.74 vs 2.58 m) | both validated; nothing order-specific in the design |
| U4 | Exact waypoint, blind and dropout metric definitions | our proxies may rank differently | blind rule chosen to be good under absolute, anchored and endpoint definitions |
| U5 | Hidden files are the public sequences | a memorising competitor could beat our TUM-VI numbers by ~18 % (E48 in-pool) | we do not memorise (held-out it costs 16–66 %); stated openly |
| U6 | Hidden data from other platforms (e.g. a head-mounted XR device, very slow or very still motion) | the gate could see a lock near 0.035 | gate + independent guard (caught every hand-held window of ≥ 54 s and every prefix of ≥ 34 s forced into the rotor branch; missed 13 of 36 windows of 20–41 s, E55b); margins measured on our data only |
| U7 | Sequences much longer than 3 minutes | unvalidated accuracy; memory (≈ 55 B per slot persistent) | linear-time algorithm; O(√K) smoother memory; long-input policy in §12.5 (≤ 8 MB up to ~9 min, ~12 min with q recomputed); longer files would exceed 8 MB for any read-everything design |
| U8 | Order-B rotor sensitivity to the sensor frame (E60) | ±0.1–0.3 m on seven flights | recorded; no systematic penalty |
| U9 | Hand-held drift detector margin (max 1.97 vs 2.0 mrad/s) | an occasional spline on hand-held data | measured cost +0.015 m (n.s.) |
| U10 | Fidelity of the NumPy port to the SciPy/Numba reference | silent accuracy loss | T2/T3 tolerances; stage-by-stage checks |
| U11 | TUM-VI licence (CC BY 4.0, from memory) | attribution wording | verify on the dataset page in Phase 5 |
| U12 | Our TUM-VI ground-truth velocity (Savitzky–Golay derivative of mocap positions) is unreliable as an oracle | the TUM-VI velocity oracles of E53/E56 are not interpretable; E48's TUM-VI labels may be noisy (threat under-estimated) | stated where used; no decision rests on them |
| U13 | Orientation error after the position-based SE(3) alignment is unstable when that rotation is ill-determined (V2_01 flips by up to 144°, E65) | an orientation metric computed this way is noisy for every team | rotational RPE (alignment-free) is our attitude check (1.77° per second on EuRoC, unchanged by V5P4) |

---

## Appendix A — Experiment index (Phase 3, official-data part)

| ID | Script (`research/experiments/`) | Result JSON / log | What it established |
|---|---|---|---|
| E40 | `e40_official_audit.py` | `e40_official_audit.json` | the official script, its masks, the column-order consequences; exact reproduction (34/34) |
| E41/E41b | `e41_official_bench.py` | `e41*_official_bench.json` | v4 breaks on official data; lock fix; forced-platform costs |
| E43 | `e43_blind_official.py` | `e43*_blind_official.json` | first blind study on official data |
| E44/E45 | `e45_official_v5c.py` | `e45_official_v5c.json` | the two dropout root causes and fixes; 306 missions |
| E46/E46b | `e46_stress_v5c.py`, `e46b_gap_law.py` | `e46*_*.json` | 29-condition stress suite; gap-law sweep (k₂) |
| E47 | `e47_official_tune.py` | `e47_official_tune.json` | grouped CV of k₂, governor, guard |
| E48 | `e48_learned_official.py` (train / eval) | `e48_learned_official.json`, `logs/e48_*.log` | learned-velocity threat, in-pool vs LOEO |
| E49 | `e49_prior_cv.py` | `e49_prior_cv.json` | motion-prior grouped CV |
| E50 | `research/poc/e50_numpy_feasibility.py` | `research/poc/e50_mac_dec4.log` | NumPy kernel costs; import footprint |
| E51 | `e51_evo_official.py` | `e51_evo_official.json` | evo cross-check and output-format audit |
| E52 | `e52_final_confirm.py` | `e52_final_confirm.json` | frozen pipeline, fresh seeds, 578 missions |
| E53 | `e53_blind_final.py`, `e53_analyze.py` | `e53_blind_final.json`, `logs/e53_analyze.log` | blind candidates, oracles, regret, LOEO selection |
| E55/E55b | `e55_gate_windows.py`, `e55b_guard_catch.py` | `e55*_*.json` | gate and guard on 306 windows; threshold 0.035 |
| E56 | `e56_error_budget.py` | `e56_error_budget.json` | oracle error budget |
| E57 | `e57_final_v5b.py` | `e57_final_v5b.json`, `e57_final_v5f.json`, `logs/e57_final_tables.log` | fresh-seed confirmation of the blind rule; final tables |
| E58 | `e58_ablation.py` | `e58_ablation.json` | leave-one-component-out |
| E59 | `e59_baselines.py` | `e59_baselines.json` | classical baselines (strapdown, heave-style high-pass with LOEO cut-off) |
| E60 | `e60_sensor_rotation.py` | `e60_sensor_rotation.json` | sensor-frame rotation invariance |
| E61/E61b | `e61_spline_iters.py`, `e61b_spline_iters_low.py` | `e61*_*.json` | spline pass count 2–20 |
| E63/E63b | `e63_decimated_gravity.py`, `e63b_decimated_gravity_low.py` | `e63*_*.json` | gravity smoother on block means: 50 → 3.1 Hz all within 4 mm ATE |
| E62/E62b | `research/poc/e62_kernel_bench.py`, `research/poc/e62b_fast_kernels.py` | `logs/e62b_mac.log` (+ cloud run in this report) | kernel costs, fast designs, exact renderer |
| E64/E64b/E64c | `research/poc/e64_tight_kernels.py`, `e64b_pass_count.py`, `e64c_cost_kernels.py` | `logs/e64_mac.log`, `logs/e64c_mac.log`, `e64b_pass_count.json` (+ cloud runs in this report) | tight smoother kernels (exact), front-end kernels, pass counts → runtime cost model |
| E65 | `e65_p4_config.py` | `e65_p4_config.json`, `logs/e65_p4_config.log` | accuracy of the submission configuration V5P4 on the E52/E57 protocol |

Reproduce the final tables: `.venv/bin/python research/experiments/e52_final_confirm.py && .venv/bin/python research/experiments/e57_final_v5b.py && .venv/bin/python research/experiments/e65_p4_config.py` (~3–5 min each on 10 cores of the M3 Pro). Reference estimator for the Phase 4 regression tests: `fbs6.estimate_final(tn, w, a, k_drag, cfg=fbs6.V5P4, blind=blind5.final_blind)` with `k_drag = 0.1873` (the submission value; evaluations use leave-one-sequence-out values from `p4.k_loso`).
