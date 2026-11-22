# PHASE 4 — BUILD AND VERIFICATION OF `main.py` (V5P4)

**Five Blind Seconds · Samsung PRISM XR Hackathon 2026 · Track 03 "Path Finder"** · 21–22 Nov 2026
Builds what `PHASE3_FINAL_RESEARCH_AND_ARCHITECTURE_FREEZE.md` §12 specifies. Nothing has been published or submitted.

---

## 0. Bottom line

| Item | Result |
|---|---|
| Deliverable | `release/main.py` v1.0.0: one file, **Python ≥ 3.9 + NumPy ≥ 1.22 only**, 1,296 lines, single thread, no network, deterministic; MD5 `8817f28ca77ee4477913aeb19090ec43` |
| Fidelity to the research reference (`fbs6.estimate_final(..., cfg=fbs6.V5P4, blind=blind5.final_blind)`) | every position within **2.3 × 10⁻¹³ m** and every quaternion within 4.4 × 10⁻¹⁶ on the 34 official-format files; per-mission ATE within 6.4 × 10⁻¹⁴ m on 340 missions; platform, guard and bias-mode decisions identical everywhere (1,309 inputs) |
| Tests T1–T7 (§12.8) | **all pass** (table in §3), T6 in a real `python:3.12-slim` container and in ten fresh-environment runs |
| Memory (complete program, tracemalloc peak) | MH_01 (184 s, worst case): **2.63 MB** own heap, **3.13–3.14 MB** with `main`'s own import (target ≤ 4 MB); 10-minute inputs: **6.86 MB / 7.36 MB** (target ≤ 8 MB) |
| Runtime (complete program, start-up excluded) | MH_01 order B (worst case): **5.2 ms** per IMU-second on the M3 Pro (target ≤ 9), **8.7 ms** on the 2.1 GHz Xeon vCPU (target ≤ 17); start-up (interpreter + NumPy + `main`) **0.05 s** (M3) / **0.09–0.10 s** (Xeon), measured separately |
| Accuracy (T3, 340 missions, organisers' degradation) | EuRoC ATE **2.670 m** (motionless 3.448), TUM-VI **0.451 m** (motionless 1.080) — the frozen V5P4 values; reproduced independently by the release tools `make_samples.py` + `evaluate.py` from the written output files (§3.1) |

Analogy for how exactness was kept while memory fell from 34.5 MB to 2.6 MB: the reference keeps every intermediate signal on
the table at once; `main.py` keeps only the raw recording in a drawer and re-derives each signal, a chapter at a time, whenever
a stage needs it — the same arithmetic in the same order, so the same answer, without the cluttered table.

---

## 1. What was built

`main.py` follows the module map of §12.4; each section of the file is labelled with its module numbers.

| Module (§12.4) | In `main.py` | Notes |
|---|---|---|
| M1–M2 ingest, blind split | `Source`, `iter_rows`, `_parse_fast`, `Data` | own zip reader on `zlib` (the `zipfile` import alone costs 0.9 MB of heap), falls back to `zipfile` for other archive types; raw values stored as int32 multiples of 10⁻⁶ (lossless for the official 6-decimal format, decoded to the same float64 values the reference parses); no per-row array is kept for monotonic inputs (the row → slot map is recomputed on each read) |
| M3 repair | `repair` | chunked despiker and ‖a‖ window check |
| M4–M5 platform statistic and gate | `thrust_lock_fill`, `rest_info` | zero-phase Butterworth by overlap-save convolution with the truncated impulse response: identical to SciPy `sosfiltfilt` (including the ends), not the frequency-domain approximation §12.4 allowed |
| M6 gyro bridge | `Bridge` | OU interpolation, cumulative sums in the reference's order |
| M7 strapdown | `Front.strap`, `qcumprod` | log-depth prefix product in chunks of 2,048 slots, attitude regenerated from per-chunk carries |
| M8 gravity smoother | `GravitySmoother` | 12.5 Hz block means; gain recursion once per input (gravity setting cached), chunked affine scans |
| M9–M10 attitude passes, bias rule | `Front.attitude`, `attitude5b`, `attitude6`, `coarse_bias`, `Bias`, `Att` | spline normal equations accumulated with an order-preserving `bincount` so they equal the reference's single call |
| M11–M13 back-end inputs and measurements | `Meas` | block sums chunk by chunk; drag rows stored for flight blocks only |
| M12 models | `model_matern52`, `model_spring`, `model_rotor`, `van_loan`, `expm`, `lyap` | own `expm` (scaling and squaring, Padé 13, Higham 2005), Lyapunov by the Kronecker vec trick |
| M14 smoother | `_rts`, `smooth_rotor`, `smooth_hand` | √K checkpoints, fixed-shape joint updates, P symmetrised after every update |
| M15–M17 Hermite, guard, blind | `BlockTimes`, `hermite`, `radius`, `Solution` | the guard's hand-held radius at 25 Hz as §12.4 specifies |
| M18 output | `write_output`, `render` | vectorised fixed-point renderer, byte-identical to `%`-formatting (10⁷ random rows + edge cases) |
| CLI and failure handling (§12.2, §12.7) | `main`, `run`, `resolve_output`, `write_motionless` | v5 → hand-held only → motionless; an empty file if the input cannot be read |

**Build order (§12.10), as followed.** (1) contract-perfect motionless program (T1 on day one); (2) `research/phase4/main_v0_exact.py`, a full-array float64 port of the reference, checked stage by stage (valid mask, rest mask, attitude, bias track, gap-filled and rotated specific force, block times, smoother states); (3) hand-held then rotor smoothers, guard and blind; T2/T3 against the frozen reference outputs (`research/phase4/p4_ref_freeze.py`, `data/phase4/ref/*.npz`); (4) the streaming rewrite (v1) for memory, re-checked after every change against the reference on all 34 files; (5) T4/T7; (6) cold start (T6). The research reference itself was not modified.

**Memory engineering, all exact** (outputs byte-identical before and after each step, on the 34 files and the two 10-minute inputs):

| Step | Effect (own heap) |
|---|---|
| v0 full-array port → v1 streaming (int32 raw storage, chunk-wise regeneration of every full-rate signal, √K smoother, re-read for output) | MH_01: 34.5 → 2.95 MB |
| own replacements for `np.median`, `np.percentile`, `np.unique` (each imports `numpy.ma`, 1.9 MB) and for `zipfile` (0.9 MB) | included above |
| drag rows for flight blocks only; accelerometer released before the gyro-only lever-arm sweep | 10-min rotor, measurement stage: 7.56 → 6.51 MB |
| gravity-gain recursion fed and finished in blocks | 10-min rotor, attitude stage: 6.91 → 6.84 MB |
| on long inputs only: block times recomputed on demand (> 2¹⁵ blocks), smoother predictions re-derived instead of stored (> 8,192 steps), smaller renderer pieces (> 2¹⁶ slots) | 10-min hand-held: 7.27 → 6.69 MB; no runtime cost on public-length inputs (M3: +0.4 % / +0.1 %, within noise) |

---

## 2. Deviations from the freeze document, and documentation discrepancies found

Where §12 and the reference code disagreed, `main.py` follows the **code**, because T2 requires agreement with it. None of these changes a result of Phase 3, which was produced by the code.

| # | §12 says | Reference code / `main.py` does | Consequence |
|---|---|---|---|
| D1 | M1: rows with equal timestamps take the first valid row's values | the **last** row in time order wins | none for the official data (duplicates are exact copies); documented |
| D2 | M3: legacy despike floors 2 × 10⁻³ rad/s and 2 × 10⁻² m/s² | floors 1.5 rad/s and 15 m/s² | the despiker is the "no measurable effect" safety item of E58; ported as in the code |
| D3 | M4: low-pass as \|H(f)\|² in the frequency domain | exact `sosfiltfilt` equivalent (overlap-save FIR of the truncated impulse response, 9-sample odd extension) | identical values incl. the ends, so the lock statistic matches to 10⁻¹⁶ instead of "decisions identical" |
| D4 | §12.3 unit guard: median over valid **rows** | median over valid **slots**, decided exactly from counts | identical whenever there are no duplicate rows; never triggers on the official format |
| D5 | §12.3: values as float32, timestamps int64, row → slot map int32 | values int32 micro-units; timestamps and the map not kept for monotonic inputs | lossless instead of float32-rounded; less memory |
| D6 | §12.3: parse in chunks of ~2¹⁸ bytes; renderer chunks ≤ 4,096 rows; M7 chunks of 4,096 | 2¹⁶ bytes; 1,024 (512 beyond 2¹⁶ slots) rows; 2,048 slots | smaller transients; results unchanged (chunking is exact or at round-off) |
| D7 | requirements `numpy>=1.21` | `numpy>=1.22` | NumPy 1.21.6 on macOS arm64 (M3 Pro, macOS 27.2.1) crashed in 2 of 12 runs (SIGSEGV/SIGBUS inside a 17 × 17 matrix product of its bundled BLAS); 1.22.4, 1.23.5, 1.24.4 and 1.26.4: 12/12 clean and byte-identical. On Linux x86-64, 1.21.6 runs correctly |
| D8 | T5 target "≤ 4 MB after `import numpy`" | measured in four windows (§4) | met in every window except those that include NumPy's own import (7.5–8.1 MB alone, unavoidable for any NumPy entry) |

---

## 3. Test results (final build, MD5 `8817f28ca77ee4477913aeb19090ec43`)

Machines: Apple M3 Pro (macOS 27.2.1, Python 3.12.7, NumPy 2.5.4; research venv with SciPy, pandas, evo 1.38.1 for the reference and scoring) and a 2.1 GHz Intel Xeon cloud vCPU (Linux, Python 3.13.18, NumPy 2.5.3).

| Test | Criterion (§12.8) | Result | Evidence (on the Mac, `~/SamsungPRISM_FiveBlindSeconds`) |
|---|---|---|---|
| **T1** contract | rows out = rows in; 8 single-space fields; timestamps = `%d.%09d` of the input ns; decimals; \|‖q‖ − 1\| < 10⁻⁶; finite; `evo_ape tum gt est -a` runs | **34/34** | `logs/phase4/p4_t2_regression.log`, `research/phase4/results/p4_t2_regression.json` |
| **T2** regression | identical decisions 34/34; per-file ATE within max(1 %, 1 cm); mean ATE within 0.5 %; blind rows within 1 cm | **pass**: decisions 34/34; mean ATE 1.8610 vs 1.8610 m (−1.9 × 10⁻⁸ %); largest file difference 5.0 × 10⁻⁷ m = the 6-decimal output rounding; in-process positions ≤ 2.25 × 10⁻¹³ m | same |
| **T3** official protocol | group means within 1 % of E65 (340 missions: seeds 42 + 5–13, orders A/B, LOSO drag) | **pass**: EuRoC 2.6696 m (E65 2.6696), TUM-VI 0.4511 m (E65 0.4511); per-mission \|ΔATE\| ≤ 6.4 × 10⁻¹⁴ m; waypoints 2.4880 / 0.3877, blind 2.7642 / 0.5991; gate decisions 340/340 | `logs/phase4/p4_t3_protocol.log`, `…/p4_t3_protocol.json` |
| **T4** robustness | E46 stress suite, E55 windows, E60 rotations run without error, decisions identical to the reference | **pass**: E46 493/493 (29 conditions × 17 sequences; largest position difference 8.2 × 10⁻⁸ m, on the extreme 0.05 rad/s² gyro ramp); E55 306/306 (2.1 × 10⁻¹³ m); E60 136/136 (2.1 × 10⁻¹³ m); 0 errors in `main.py` or the reference | `logs/phase4/p4_t4_robustness.log`, `…/p4_t4_robustness.json` |
| **T5** efficiency | heap after `import numpy` ≤ 4 MB on MH_01 (both orders), ≤ 8 MB at 10 min; runtime ≤ 9 ms (M3) / ≤ 17 ms (Xeon) per IMU-second on MH_01 B and room2 | **pass** (§4) | `logs/phase4/p4_t5_measure_mac.log`, `research/phase4/results/p4_t5_*.json` |
| **T6** cold start | fresh `python:3.12-slim` + `pip install numpy`, README command on a zipped official-format input, output passes T1 | **pass**: `release/Dockerfile` built without cache and run with the README command (image: Python 3.12.16, NumPy 2.5.5, Linux arm64, glibc 2.41; Docker 29.5.2 via colima on the Mac) on MH_01 B, room2 A, V2_03 A, room5 B: 4/4 T1 pass, every output **byte-identical** to the development build. Also ten fresh-environment runs (empty folder, only `main.py` + `requirements.txt`, venv, NumPy from PyPI): Linux x86-64 — CPython 3.12.13 + NumPy 2.5.5, 3.9.25 + 1.22.4, 3.9.25 + 1.21.6, 3.12.13 + 1.26.4, 3.13.18 + 2.5.5 and two on room2 A; macOS arm64 — 3.13.7 + 2.5.5, 3.10.20 + 1.22.4, 3.10.20 + 1.26.4: all byte-identical | `logs/phase4/p4_t6_docker.log`, `logs/phase4/p4_t6_cold_start_mac.log`, `…/p4_t6_cold_start_*.json` |
| **T7** edge cases | header variants, CRLF, blank lines, spaces, scientific notation, nested zip, no blind window, 1/2-row and all-zero inputs, duplicates, NaN | **37/37** (also: tab-separated, stored and bzip2 zips, folder input, empty file, header only, constant values, non-monotonic time, stamps in s and µs, accelerometer in g, gyro in deg/s, 3 s input, 65 s gap, huge values, extra columns, output to a folder, `-i/-o`, no arguments, missing input) | `logs/phase4/p4_t7_edge_cases.log` |
| renderer | byte-identical to `%`-formatting | 10⁷ random rows + edge cases: 0 mismatches | `logs/phase4/p4_render_test.log` |

Also checked (cloud): six alternative code paths (guard firing, forced rotor on a hand-held room, coarse PI start on rotor and on hand-held, drift-triggered hand-held spline, forced hand-held on a flight) give byte-identical output before and after the memory work.

### 3.1 The release tools, checked against the research harness

`tools/make_samples.py` (the organisers' `degrade_imu.py`, both column orders, any seed) and `tools/evaluate.py` (NumPy only: SE(3) Umeyama ATE, waypoint / blind / dropout proxies, RPE; `--protocol` runs a whole evaluation) let anyone reproduce the accuracy figures without the research code (`research/phase4/p4_tools_check.py`, `logs/phase4/p4_tools_check.log`):

- `make_samples.py` recreates all 34 seed-42 official-format inputs **byte for byte** (CSV content).
- `evaluate.py` on the 34 T2 outputs equals the research scorer `p4.score` exactly for ATE, waypoints, blind and dropout metrics (difference 0.0) and to 7 × 10⁻¹⁵ for the rotation diagnostics; it agrees with `evo_ape -a` to 5 × 10⁻⁷ m.
- `evaluate.py --protocol` (340 missions: inputs made by `make_samples.py`, `main.py` run, output written in the submission format, scored from that file, motionless scored alongside) reproduces T3: EuRoC ATE **2.6696** m (motionless 3.4484), waypoints 2.4880 (3.2478), blind 2.7642 (3.0832), after-dropout 2.2853 (3.0448); TUM-VI ATE **0.4511** (1.0795), waypoints 0.3877 (1.1100), blind 0.5991 (0.6142), after-dropout 0.4079 (1.0246); per-mission difference to T3 ≤ 4.8 × 10⁻⁹ m (the output file's 6-decimal rounding); 0 contract failures, 0 fallbacks.

---

## 4. Efficiency (T5)

### 4.1 Heap (tracemalloc peak of the complete program; MB = 10⁶ bytes)

own = tracing after `import numpy` and `import main` (our data structures and buffers — the FAQ's definition); +main = after `import numpy` only; whole = before `import numpy`; +interp = `python -X tracemalloc`.

| Input | IMU s | M3 Pro: own · +main · whole · +interp | Xeon vCPU: own · +main · whole · +interp |
|---|---|---|---|
| MH_01 order A (rotor) | 184.1 | **2.63** · 3.12 · 10.60 · 12.41 | **2.62** · 3.13 · 11.18 · 13.23 |
| MH_01 order B (rotor, worst case) | 184.1 | **2.63** · 3.13 · 10.60 · 12.41 | **2.63** · 3.14 · 11.19 · 13.23 |
| room2 order A (hand-held) | 144.1 | **2.45** · 2.95 · 10.42 · 12.23 | **2.45** · 2.96 · 11.01 · 13.06 |
| 10-minute rotor input (MH_01 tiled) | 600.0 | **6.86** · 7.35 · 14.82 · 16.63 | **6.85** · 7.36 · 15.41 · 17.45 |
| 10-minute hand-held input (rooms tiled) | 600.0 | **6.69** · 7.19 · 14.66 · 16.47 | **6.69** · 7.20 · 15.24 · 17.29 |

Targets: MH_01 after `import numpy` ≤ 4 MB — met (≤ 3.14 MB); 10-minute input ≤ 8 MB — met (≤ 7.36 MB). Scaling: about 50 bytes per IMU sample beyond a fixed ~0.8 MB, so 8 MB is reached at about 12 minutes of 200 Hz data (about 11 minutes in the +main window), extrapolated linearly from the two measured lengths. **Exception:** if the gravity-health check fails (coarse PI start; 1 of 374 rotor missions in E52), the bias start is a full-rate array of 24 bytes per sample: measured 3.50 MB on MH_01 B but 9.73 MB on the 10-minute rotor input, so on that path the limit is reached at about 8 minutes. Stage peaks are in `metadata.json`.

### 4.2 Runtime (wall clock of the complete run, start-up excluded; ms per IMU-second, min / median of 5)

| Input | M3 Pro: run min / median | M3 Pro: whole command | Xeon vCPU: run min / median | Xeon vCPU: whole command |
|---|---|---|---|---|
| MH_01 order A | 4.11 / 4.13 | 4.52 | 7.20 / 7.64 | 8.50 |
| MH_01 order B (worst case) | **5.16 / 5.23** | 5.58 | **8.69 / 9.65** | 10.68 |
| room2 order A | **4.10 / 4.13** | 4.58 | **7.17 / 7.32** | 8.82 |
| 10-minute rotor | 5.25 / 5.25 | 5.40 | 8.40 / 8.90 | 8.99 |
| 10-minute hand-held | 4.34 / 4.36 | 4.50 | 7.72 / 7.95 | 7.95 |

Targets (§12.8 T5, MH_01 B and room2): ≤ 9 ms on the M3 Pro — met (5.2 / 4.1); ≤ 17 ms on the Xeon — met (8.7–9.7 / 7.2–7.3). The budget is 20. "Whole command" is `python main.py INPUT OUTPUT` timed from outside, start-up included. The Xeon is a shared cloud vCPU: its medians vary by about ±10 % between sessions. On the M3 Pro the memory work cost nothing measurable on public-length inputs (stage timing against the build before it: +0.4 % on MH_01 B, +0.1 % on room2, within noise). The first M3 T5 run was disturbed by the laptop sleeping (lid closed) and was repeated; the table is the repeat.

### 4.3 Start-up, measured separately (median of 5, s)

| Machine | interpreter only | + `import numpy` | + `import main` | whole command, MH_01 B |
|---|---|---|---|---|
| M3 Pro (Python 3.12.7, NumPy 2.5.4) | 0.016 | 0.049 | 0.051 | 1.03 |
| Xeon vCPU (Python 3.13.18, NumPy 2.5.3) | 0.014 | 0.080 | 0.100 | 1.97 |

Start-up is ~0.05 s (M3) / ~0.1 s (Xeon) of fixed cost per run, i.e. 0.3 / 0.5 ms per IMU-second on a 184 s input and 1.7 / 3.3 ms on a 30 s input — not included in the runtime table above.

---

## 5. Open items and risks carried into Phase 5

1. Package per §12.1 and §12.9: the release folder holds `main.py`, `requirements.txt`, `Dockerfile`, a README draft, `metadata.json` (generated from the T5 files), `tools/` (check, measure, make_samples, evaluate) and `tests/`; still to do in Phase 5 — `research/` and `docs/` in the public layout (no dataset files), the `team_name.zip` bundle with an `estimated_trajectory.txt`, the deck, the video and the form.
2. Docker on the Mac is colima + the docker CLI from Homebrew (`colima start` before use, `colima stop` to free the VM's memory).
3. Unchanged unknowns (§14 of the freeze report): the organisers' tracemalloc window (NumPy's import alone exceeds 8 MB if it is inside it), the grading machine, the column order, the metric definitions, whether hidden files are the public sequences.
4. Long-input limits as stated in §4.1.
5. Publication decision (public repository, IP disclosure): taken in Phase 5; this repository is the result.
