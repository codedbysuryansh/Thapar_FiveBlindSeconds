# PHASE 5 — OPTIMISATION PROGRAMME AND CHAMPION v1.1.0

**Five Blind Seconds · Samsung PRISM XR Hackathon 2026 · Track 03 "Path Finder"** · 22 Nov 2026
Follows `PHASE4_BUILD_AND_VERIFICATION_REPORT.md`. Paths such as `opt/`, `champion/` and `rc/` refer to the team's workspace;
the scripts, registry and notes of `opt/` are published in this repository's `opt/` folder (bulky per-mission outputs are not).

---

## 0. Bottom line

| Item | Result |
|---|---|
| New champion | **`main.py` v1.1.0**, SHA-256 `c401d784…44e2` (97,476 bytes), immutable snapshot `champion/v1.1.0/` (66 files, SHA256SUMS, read-only + user-immutable flag) |
| Fallback | `champion/v1.0.0/` untouched (58/58 checksums OK); local `release/` repository and tag `v1.0` untouched |
| Accuracy, official protocol (T3: seeds 42 + 5–13, both orders, 340 missions) | EuRoC ATE 2.670 → **2.294 m** (−14.1 % [−21.2, −4.8]), waypoints −15.8 % [−23.8, −4.8], after-dropout −13.3 % [−21.8, −2.2], blind −3.2 % (n.s.); TUM-VI ATE 0.451 → **0.418 m** (−7.3 % [−13.8, +1.5]), waypoints −11.4 % [−18.1, +0.1]; no metric significantly worse |
| Promotion set (CONF, seeds 3001–3010, never used to choose) | pooled composite −11.21 % [−17.41, −3.52]; STRESS (crops, severity ×0.5/×1.5, ramp shapes) −9.10 % [−13.93, −3.15] |
| Robustness (T4) | 935 Phase 4 inputs + 799 held-out inputs: 0 errors, 0 fallbacks, every family's mean ATE better than v1.0.0 |
| Efficiency (T5) | own heap unchanged (2.64 MB at 3 min, 6.86 MB at 10 min); worst runtime 11.7 ms per IMU-second on the 2.1 GHz Xeon (6.7 on the M3 Pro); +25–27 % on the 10-minute rotor input only |
| Portability (T6) | byte-identical outputs: macOS arm64, python:3.12-slim container (arm64), three fresh macOS environments (Python 3.10/3.13, NumPy 1.22.4/1.26.4/2.5.5), native Linux x86-64 and a genuine x86-64 Docker container |
| Programme | about 160 registered experiments + 21 analyses, all kept with reasons; three accuracy changes and one robustness fix accepted |

Analogy for what changed: version 1.0 treated a drone's drag reading like a kitchen scale that is re-tared every few seconds,
even in a huge hall where the tare never moves, and it kept weighing the floor while the drone stood on it. Version 1.1 stops
weighing the floor, fixes the tare where the room is big, and gives the hand-held tilt a heavier damper when the gyro is clean.

---

## 1. How the programme was run

- **Champion protected.** v1.0.0 frozen in `champion/v1.0.0/` (git archive of tag v1.0, history bundle, Phase 4 logs, reference
  outputs). A separate workspace `opt/` (local git, no remote) holds the harness, the development source and every candidate.
- **Every experiment saved.** `opt/candidates/<ID>/`: its own `main.py` snapshot and `config.json` (SHA-256 of both), `meta.json`
  (parent, hypothesis, time), run logs, per-mission JSON results, paired comparison. The harness refuses a candidate whose files
  changed after registration. `registry.json` / `REGISTRY.md` hold the status and the precise reason of each.
- **Same protocol for everyone.** The organisers' `degrade_imu.py` (unmodified, SHA-256 checked) on the 17 public sequences, both
  column orders; scoring by the Phase 4 scorer (= evo to 5e-7 m); leave-one-sequence-out drag coefficient; ground truth only for
  scoring. Mission sets: DEV (seeds 1001–1004) to explore, DEV2 (2001–2004) to replicate, CONF (3001–3010) only to promote,
  STRESS (crops, severity, ramp shape) for robustness, S42 reported and never selected on.
- **Statistics.** Paired, mission by mission; composite = rubric-weighted (30/20/10) relative change of waypoint, blind and
  after-dropout errors; 95 % intervals from a sequence-level bootstrap stratified by dataset.
- **Promotion rule** (`opt/README.md` §7): pooled composite better with the interval below 0 on CONF; no dataset and no rubric
  component significantly worse; no STRESS loss or new failure; runtime and memory bounds; all regression tests. Two rules were
  written down in round 2: changes below 0.5 % composite are not pursued (7a), and the T4 criterion for a *new* champion (7b:
  no errors or fallbacks, every decision change listed and explained, family means not worse, tail cases listed, a held-out
  robustness set for anything designed after looking at T4).

## 2. Linux x86-64

The cloud VM is a real 2.1 GHz Xeon (x86-64), so no QEMU emulation was needed. Docker Hub and other registries are blocked by the
network proxy, so the x86-64 container is the release Dockerfile on a debootstrapped Ubuntu 24.04 minbase (Python 3.12.3) with
the NumPy 2.5.5 wheel installed offline, run with `--network none`. v1.0.0 (round 1) and v1.1.0 (now): 34/34 official-format
outputs byte-identical to macOS arm64. The same VM gives the slow-machine timing (T5).

## 3. What was tried, by area

| Area | Experiments | Outcome |
|---|---|---|
| Process priors (rotor spring-damper, Matérn, drag noise, gap laws, ZUPT noise, lever arm) | 66 one-at-a-time sweeps (P01–P17) + 12 re-tunes with the final gravity (X116, X117) | none adopted: the V5P4 values sit on flat optima; the best re-tune (Matérn τ_b 40 s, TUM-VI −0.42 %) is below the 0.5 % practical threshold |
| Drone velocity estimation (drag structure, drag offset, ground contact, likelihood/oracle k, thrust coupling) | ~30 + analyses AN02, AN04–AN10 | **adopted:** rest detector 50 %/0.06 rad/s (X041g), no drag rows at rest, radius-switched steady drag offset (X044) → package X107f; rejected: fixed tight priors (V1 +3..+9 %), likelihood selection (anti-correlated with accuracy), own-k oracle (no gain), thrust coupling, quadratic/per-axis drag |
| Hand-held tilt and gravity | 20 (X001–X003, X107, X108, X110b/c) | **adopted:** 24 s final gravity when the gyro is clean (X108b); rejected: acceleration-compensated gravity, longer τ everywhere (fragile under gyro faults: T4 +30 %), 12 s / 8 s (equal accuracy, no robustness gain) |
| Platform classification / guard | X043, X110–X112, AN16–AN17, AN22, AN25 | **adopted:** guard on the loose solution with revert (X110d); rejected: soft blends, ratio gate (MH_03 overlaps the misfires), drag-offset-wander gate (AN25: better on official DEV, −14.8 % vs −13.0 %, but switches small rooms on under accelerometer ramps, +150..+308 %) |
| Smoothing models, heteroscedastic noise | X005, combos | no gain (−0.02 %) |
| Dropout recovery | P11, P12 (gap rotation law, bias jump) | champion values kept |
| Blind-window prediction | X030 (rotor damping 0.3–1.2 s) | no effect: information-limited (Phase 3 E53) |
| Compact learned / hybrid estimators | X060a–h + AN13 | **rejected** (§6) |
| Computational and memory | AN12 profile, AN19 compile transient, T5 on two machines | no change adopted; facts recorded (§7) |
| Interactions | combos X100–X106, X107/X108 splits, X110 × switch | the switch × guard interaction was the one real defect found (§4) |

## 4. The accepted changes and the release-candidate history

1. **Ground handling** (X041g + no drag rows at rest): at rest the accelerometer measures the floor's push; the old model learned
   about −0.65 m/s² there and needed ~20 s after take-off to unlearn it (AN08). Alone −2.4 % EuRoC (DEV); it pays mainly together
   with (2).
2. **Steady drag offset in a large workspace** (X044): loose rotor radius ≥ 1.7 m → re-smooth with drag-offset prior sd 0.03 m/s²,
   τ 1000 s, initial sd 0.15. MH −18.7 %, V1 +0.5 % (always-tight: V1 +8.9 %).
3. **Clean-gyro long gravity** (X108b): hand-held, constant bias mode, |μ| ≤ 0.002 rad/s, sample noise ≤ 0.04 rad/s → final
   attitude with τ = 24 s. TUM-VI −6.5 % (CONF).
4. **Guard order** (X110d): T4 on the release file v1.1.0b found 24 radius-guard flips on Machine Hall flights (AN15): the steady
   prior enlarges the rotor radius 1.5–1.8× toward the true size, so the unchanged 12× guard (calibrated on loose solutions)
   rejected sound solutions (+10 % ATE each). Checking the guard on the loose solution and reverting a steady solution that fails
   it gives identical outputs on every official set and the best T4 balance (mean −5.27 % vs −4.89 %, tail 8 cases vs 8, decision
   changes 8 vs 24). X110a (no revert) improved the mean but doubled the tail (17): the guard had been protecting some bad steady
   solutions by accident.

Builds: RC 1.1.0a (failed T4: unconditioned long gravity, +60 % on short windows) → 1.1.0b (clean-gyro condition; guard defect
found by its own T4) → **1.1.0c = X110d, promoted**. 1.1.0c is byte-identical to the research build on 986/986 harness missions
and to 1.1.0b on all 340 T3 missions.

## 5. Regression of v1.1.0 (the release file itself)

| Test | Result |
|---|---|
| T1 contract | 34/34 on macOS arm64 and Linux x86-64, outputs identical across machines |
| T2 regression | 986/986 byte-identical to the registered research build (S42, DEV, DEV2, CONF, STRESS) |
| T3 protocol | table in §0; per sequence MH_03–MH_05 −23..−30 %, V rooms within ±2 %, room4 −15 %, room6 −20 %, room1 −7 %, room3 +6 %, room5 +2 %; 0 contract failures, 0 fallbacks, guard fired 0 |
| T4 (935) | E46 −2.83 %, E55 −7.19 %, E60 −13.44 % mean ATE; 1 bias-mode change (half-rate room1, from the rest detector, +5 %); strict Phase 4 identity FAIL for that one change, rule 7b PASS |
| T4 held-out (799) | E46 stress seed 1 −3.28 %, F55 (windows of seed-7 files) −6.05 %; 10 cases > 25 % worse (same count as v1.1.0b, whose mean was −3.62 %) |
| T5 | M3 Pro (Python 3.12.7, NumPy 2.5.4): 4.1–6.7 ms per IMU-second; Xeon (3.13.18, 2.5.3): 7.2–11.7; own heap 1.72–2.64 MB (≤ 184 s), 6.69–6.86 MB (600 s) |
| T6 | container python:3.12-slim arm64 6/6; fresh macOS envs 3/3; x86-64 native 34/34 and container 34/34; all byte-identical |
| T7 / renderer | 37/37; 10^7 rows, 0 mismatches |

Cases more than 25 % worse than v1.0.0 (all outside the official envelope): accelerometer bias ramp to 1 m/s² (more than ten times
the official bias walk) triggering the steady offset in small rooms (+88 to +182 %, three cases) and +29 to +51 % in some Machine
Hall and hand-held cases; slow gyro ramps on hand-held data with the long gravity (up to +79 %); recordings starting at 60 % of a
room5 file (+25 to +49 %); a 15 % prefix of MH_03 (on the ground, classified hand-held: 0.15 → 0.27 m).

## 6. The learned-model threat, retested properly (AN13)

1-D CNN (≈73 k weights) on 1 s windows at 100 Hz plus the gravity direction, body-frame velocity target, heteroscedastic loss;
trained on TRAIN missions, tested on DEV; folds leave-one-environment-out (MH, V1, V2) and leave-one-room-out; fused as velocity
rows with the predicted variance × s².

- Held-out velocity RMSE: net 0.58–0.98 m/s vs our physics 0.10–0.32 m/s (worse than predicting zero in room4 and room6);
  over-confident by 1.3–3.2×.
- Fusion, held-out: s = 1/2/4/8 → +53.7 / +26.4 / +9.0 / +2.6 % composite (never better; it only fades as it is switched off).
- Fusion, in-pool (the net has seen the test missions: memorisation): best −9.3 % [−13.4, −6.0] at s = 4.

Meaning: a compact learned model gives a team nothing on an unseen environment; memorising the public sequences could give
about 9 % on DEV *if* the hidden files are those sequences. We do not do that.

## 7. Facts and risks

1. **Memory-window risk (AN19, also true for v1.0.0).** If a grader starts tracemalloc *before* `main.py` is compiled (a first
   import with no `__pycache__`, or `runpy` under tracing, with NumPy already imported), compiling our 97 KB file alone peaks at
   10.3–11.2 MB (CPython 3.10–3.13) and then releases all but 0.4–0.9 MB; with cached bytecode the same window is 3.1 MB. Only two
   remedies exist: (a) a loader that compiles the code in sections from string literals with `exec` (bounds the peak near 1.3 MB,
   but the file stops looking like ordinary Python and some graders flag `exec`), or (b) several modules (against the "standalone
   main.py" rule). Not done. The FAQ wording ("data structures, buffers, and trajectory estimation logic") favours the own-heap
   window, where we use 2.6 MB.
2. **TUM-VI gain is not significant** (6 rooms; from room4, room6, room1); room3 +6 %.
3. **Anchored blind displacement** slightly worse (EuRoC +2.2 %, TUM-VI +1.1 %, both n.s.); absolute blind error better.
4. **Runtime** +25–27 % on 10-minute rotor input (second smoother pass); worst 11.7 ms per IMU-second, limit 20.
5. **Out-of-envelope stress tails** (§5).
6. Unknowns unchanged: tracemalloc window, grading machine, column order, exact metric definitions, whether hidden files are the
   public sequences.

## 8. Convergence

The last 15 experiments (X110b/c, X111a/b, X116a–f, X117a–f) produced nothing above the 0.5 % threshold; the only material
change of round 2 was a robustness fix with identical official accuracy. Ideas left, with honest expected value:

- **Drift-aware switch statistic** (tell absorbed velocity from real offset drift): could recover MH_01 (−12 % if switched) without
  the accelerometer-ramp failures; needs a new statistic and fresh validation. Medium effort, small expected gain on the official
  data, real risk.
- **Faster smoother** (per-step NumPy overhead dominates, AN12): pays only if runtime is scored relatively; we are at ≤ 58 % of the
  budget on the slow machine.
- **Compile-transient loader** (§7.1): decision, not research.

## 9. Decisions taken for the submission

1. **v1.1.0** is the submission build (`src/main.py` in this repository; SHA-256 in `metadata.json`). v1.0.0 remains the archived
   fallback.
2. The compile-transient remedy (§7.1) was not adopted: the file stays ordinary, standalone Python.
3. Publication: this public repository, with `docs/` and the research code and no dataset files; the bundle
   `Thapar_FiveBlindSeconds.zip`; the deck in `presentation/`; the demo video linked from the top-level README.
