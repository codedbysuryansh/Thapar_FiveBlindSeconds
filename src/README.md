# Five Blind Seconds — IMU-only trajectory estimation under sensor degradation

Samsung PRISM XR Hackathon 2026 · Track 03 "Path Finder: Trajectory Estimation Under Sensor Degradation" · team **Thapar_FiveBlindSeconds** · version **1.1.0**

`main.py` reads one degraded IMU recording (EuRoC format, `imu0/data.csv`, zipped or not) and writes `estimated_trajectory.txt`
(TUM format): one 6-DoF pose for **every input row**, including duplicated rows and the rows of the final five blind seconds.
It is a single file that needs **Python ≥ 3.9 and NumPy ≥ 1.22 only**, runs on one CPU thread and uses no network, no other
input and no randomness.

## Run it

```bash
pip install -r requirements.txt                      # numpy>=1.22
python main.py path/to/input.zip estimated_trajectory.txt
```

| Argument form | Meaning |
|---|---|
| `python main.py INPUT [OUTPUT]` (also `-i/--input`, `-o/--output`) | `INPUT`: a `.zip` (searched for `*/imu0/data.csv`, else any `data.csv`, else the only `.csv`), a folder (same search) or a CSV file |
| `OUTPUT` a folder | writes `OUTPUT/estimated_trajectory.txt` |
| no arguments | searches the working folder for an input, writes `./estimated_trajectory.txt` |

The program prints the path it wrote and exits with code 0. The `estimated_trajectory.txt` shipped in the submission bundle is
this program's output for the official-format input `MH_01_easy`, column order A, seed 42 (made from the public EuRoC sequence
with the organisers' script by `tools/make_samples.py`; 35,629 rows; the last 1,002 carry no valid IMU data: the blind window). It never leaves without a valid file: if estimation fails it falls
back to the hand-held branch, then to a motionless trajectory with gravity-levelled orientation, still one pose per row
(the fallback is reported on stderr).

Docker (the same two steps inside `python:3.12-slim`):

```bash
docker build -t fiveblindseconds .
docker run --rm -v "$PWD":/data fiveblindseconds input.zip estimated_trajectory.txt
```

Output lines are `"%d.%09d %.6f %.6f %.6f %.7f %.7f %.7f %.7f"`: timestamp in seconds written from the integer nanoseconds
(no floating-point rounding), position in metres, unit quaternion `qx qy qz qw` (body to world, z up).

## How it works

A strapdown integrator is a blindfolded walker counting steps: every miscount stays forever, so position error grows at least
with the square of time. This program instead asks which smooth, physically plausible path of a drone or of a hand-held device best
explains *every* accelerometer sample, before and after each moment, the way a detective reconstructs a route from all the
clues at once.

```
imu0/data.csv
 1 Ingest         uniform slot grid at the median time step; duplicated rows share a slot (one output row each);
                  zero, NaN and exact-repeat rows are invalid; missing rows become invalid slots
 2 Blind split    the trailing run of invalid slots is the blind window (predicted, never fitted)
 3 Repair         implausible |a| windows invalid; isolated spikes interpolated
 4 Platform gate  rigidity of the thrust axis (gap-filled, one 1 Hz low-pass): rotor if the lock statistic < 0.035
 5 Attitude       Ornstein-Uhlenbeck gyro bridge across gaps -> strapdown -> integrated-random-walk gravity smoother
                  (12.5 Hz block means, time constant 3 s; rotation uncertainty min(k2*T^2/2, 1 rad) across a gap of length T)
                  -> tilt correction; gyro bias: rotor -> penalised 20 s linear spline; hand-held -> constant / rest-anchored
                  unless a drift test fires; hand-held with a clean gyro -> final attitude with a 24 s time constant
 6 Position       world-frame specific force is a MEASUREMENT of a stochastic motion prior; Kalman filter + RTS smoother
                    rotor     : spring-damper prior + accelerometer bias + rotor-drag rows + lever arm (17 states, 25 Hz)
                    hand-held : Matern-5/2 stationary prior + accelerometer bias (100 Hz)
                  zero-velocity rows at genuine rest (quietest half of the 0.5 s windows with gyro sd < 0.06 rad/s, which
                  includes a drone on the ground with its rotors spinning); no drag rows at rest;
                  accelerometer-bias variance + 0.25*min(1, T/2 s) after every gap
 7 Guard, switch  rotor solution's RMS radius > 12x the hand-held one -> hand-held solution; otherwise, if the radius is
                  >= 1.7 m (large workspace), smooth again with a steady drag-row offset and keep that solution only if it
                  passes the same guard
 8 Blind window   rotor: smoothed velocity at failure, decaying with a 0.5 s time constant; hand-held: the smoother's prediction;
                  orientation held
 9 Output         one pose per input row, input order
```

Why these choices: the organisers' degradation produces thousands of 5 ms drops; a filter that resets its uncertainty at every
gap (as most designs do) breaks on that pattern, and a per-segment low-pass breaks the platform statistic. Steps 4–6 are built
around gap length. The full study of version 1.0 is in the repository's `docs/PHASE3_FINAL_RESEARCH_AND_ARCHITECTURE_FREEZE.md`; the build and
optimisation reports are next to it.

### What changed in 1.1.0, and why

Version 1.1.0 comes out of a registered optimisation programme: about 160 candidates (the rejected ones are kept with their
reasons) ran on the same official degradation and were compared mission by mission with the previous version; a change was
screened on one set of seeds, replicated on a second and confirmed on a third that was never used for a choice. Three changes
survived:

1. **Ground handling for drones.** A drone standing on the ground with its rotors spinning now counts as "at rest", and the
   rotor-drag rows are dropped there. On the ground the accelerometer measures the push of the floor, not air drag: the old
   model learned an offset of about −0.65 m/s² there and needed about 20 s after take-off to unlearn it.
2. **A steady drag offset in large workspaces.** Think of the drag-row offset as the tare of a kitchen scale. In the large
   Machine Hall it stays put for the whole flight, so the program fixes the tare and every drag row then carries velocity
   information; in the small Vicon rooms it wanders, so the program keeps re-taring as before. The choice is made on the size
   of the first (loose) solution after the radius guard has checked that solution, and the steady solution must pass the same
   guard. (An earlier build checked the guard after the switch; the robustness suite showed the guard then rejecting sound
   Machine Hall solutions, so the order was fixed before release.)
3. **A calmer tilt estimate for hand-held devices.** When the gyro is clean (constant bias, residual bias ≤ 0.002 rad/s, sample
   noise ≤ 0.04 rad/s), the final attitude uses a 24 s gravity smoother instead of 3 s: a spirit level with a heavier damper,
   so a quick arm swing no longer tips it and less motion leaks into the tilt.

Tested and not adopted, among others: a learned velocity model (a 1-D CNN trained leave-one-environment-out is worse than the
physics at every weighting; trained on the test sequences themselves it would gain about 9 % on our development set, which is
memorisation and was not used), soft or ratio-gated drag switches, a drag-offset-wander statistic (better on the official data,
but it switches the small rooms on under strong accelerometer drifts), shorter hand-held time constants and re-tuned motion
priors (no gain above 0.5 %).

## Accuracy (our evaluation, not an official score)

The organisers' `degrade_imu.py`, unmodified, applied to the 17 public sequences (11 EuRoC MAV flights, 6 TUM-VI rooms), both
plausible column orders, seeds 42 and 5–13 (340 missions; never used to choose anything in version 1.1.0); SE(3) Umeyama
alignment over all output rows, as the FAQ describes. "Motionless" (all positions zero) is the honest floor every IMU-only method
must beat. Change: relative change of the mean, with a 95 % bootstrap interval that resamples whole sequences.

| Group (missions) | Metric (m) | Motionless | v1.0.0 | **v1.1.0** | Change v1.0.0 → v1.1.0 |
|---|---|---|---|---|---|
| EuRoC (220) | ATE | 3.448 | 2.670 | **2.294** | −14.1 % [−21.2, −4.8] |
| | waypoints (8 per mission) | 3.248 | 2.488 | **2.096** | −15.8 % [−23.8, −4.8] |
| | blind 5 s, mean error | 3.083 | 2.764 | **2.675** | −3.2 % [−13.2, +4.8] |
| | 1 s after each dropout of ≥ 5 samples | 3.045 | 2.285 | **1.982** | −13.3 % [−21.8, −2.2] |
| TUM-VI rooms (120) | ATE | 1.080 | 0.451 | **0.418** | −7.3 % [−13.8, +1.5] |
| | waypoints | 1.110 | 0.388 | **0.343** | −11.4 % [−18.1, +0.1] |
| | blind 5 s | 0.614 | 0.599 | **0.593** | −1.0 % [−6.9, +5.2] |
| | after dropouts | 1.025 | 0.408 | **0.387** | −5.2 % [−10.4, +2.4] |

Per sequence (ATE in m, 20 missions each):

| Sequence | Motionless | v1.0.0 | v1.1.0 | | Sequence | Motionless | v1.0.0 | v1.1.0 |
|---|---|---|---|---|---|---|---|---|
| MH_01_easy | 4.33 | 4.14 | 3.91 | | V2_01_easy | 2.30 | 2.13 | 2.12 |
| MH_02_easy | 4.66 | 4.30 | 3.89 | | V2_02_medium | 2.11 | 1.32 | 1.32 |
| MH_03_medium | 3.58 | 2.60 | 2.00 | | V2_03_difficult | 2.06 | 1.26 | 1.26 |
| MH_04_difficult | 6.81 | 4.93 | 3.47 | | room1 | 1.00 | 0.34 | 0.32 |
| MH_05_difficult | 6.87 | 5.16 | 3.81 | | room2 | 1.22 | 0.30 | 0.30 |
| V1_01_easy | 1.85 | 1.60 | 1.57 | | room3 | 1.25 | 0.41 | 0.44 |
| V1_02_medium | 1.78 | 0.98 | 0.96 | | room4 | 1.08 | 0.81 | 0.69 |
| V1_03_difficult | 1.58 | 0.96 | 0.95 | | room5 / room6 | 0.93 / 0.99 | 0.41 / 0.43 | 0.42 / 0.34 |

0 contract failures, 0 fallbacks, and the radius guard fired on none of the 340 missions.

Reproduce (needs your own EuRoC and TUM-VI copies, the organisers' `degrade_imu.py` and pandas, which that script uses;
a few minutes on 10 cores):

```bash
python tools/evaluate.py --protocol --degrade path/to/degrade_imu.py --euroc path/to/euroc --tumvi path/to/tumvi \
       --seeds 42,5,6,7,8,9,10,11,12,13 --procs 10
```

`--euroc` expects `<name>/mav0/` folders, `--tumvi` the `dataset-<room>_512_16/mav0/` folders. Each mission is degraded, run through
`main.py`, written in the submission format and scored from that file; the rotor-drag coefficient of each flight is the mean of
the other ten flights' fits (`--k-drag loso`, the default). To score one output: `python tools/evaluate.py estimated_trajectory.txt
input.zip mav0/state_groundtruth_estimate0/data.csv` (`--gt-tum gt.txt` also writes the ground truth for `evo_ape tum gt.txt
estimated_trajectory.txt -a`, which agrees with our ATE to 5e-7 m).

## Efficiency (measured on this `main.py`, complete program)

Heap is the tracemalloc peak; the organisers' measurement window is not published, so four windows are reported:
**own** = after `import numpy` and `import main` (our data structures and buffers, the FAQ's definition);
**+main** = after `import numpy` only, with `main` loaded from its cached bytecode; **whole** = from before `import numpy`
(NumPy's import alone is 7.5–8.1 MB); **+interp** = `python -X tracemalloc`. Runtime excludes start-up, which is listed
separately. 1 MB = 10^6 bytes.

| Input (IMU seconds) | Machine | own | +main | whole | +interp | ms per IMU-second | start-up s |
|---|---|---|---|---|---|---|---|
| MH_01 order B (184 s, rotor, worst public file) | Apple M3 Pro | 2.63 | 3.13 | 10.61 | 12.42 | 5.0 | 0.05 |
| | 2.1 GHz Xeon cloud vCPU | 2.64 | 3.15 | 11.19 | 13.24 | 9.2 | 0.08 |
| room2 order A (144 s, hand-held) | Apple M3 Pro | 2.45 | 2.95 | 10.42 | 12.23 | 4.4 | 0.05 |
| | 2.1 GHz Xeon cloud vCPU | 2.45 | 2.97 | 11.01 | 13.06 | 8.0 | 0.08 |
| 10-minute rotor input (600 s) | Apple M3 Pro | 6.86 | 7.36 | 14.83 | 16.64 | 6.7 | 0.05 |
| | 2.1 GHz Xeon cloud vCPU | 6.86 | 7.37 | 15.42 | 17.46 | 11.7 | 0.09 |
| 10-minute hand-held input (600 s) | Apple M3 Pro | 6.69 | 7.19 | 14.66 | 16.47 | 4.5 | 0.05 |
| | 2.1 GHz Xeon cloud vCPU | 6.69 | 7.20 | 15.25 | 17.30 | 8.4 | 0.08 |

Runtime is the median of five runs (Apple M3 Pro: macOS, Python 3.12.7, NumPy 2.5.4; Xeon: Linux, Python 3.13.18, NumPy 2.5.3).
Against v1.0.0 the heap is unchanged (within ±0.03 MB). Runtime is 6–10 % lower on MH_01, within −7 to +6 % on room2, MH_04
and the 10-minute hand-held input, and 25–27 % higher on the 10-minute rotor input (the second smoother pass of the
large-workspace switch). If tracing is already on when `main.py` itself is compiled (a first import without `__pycache__`, or
`runpy` under tracing), the compilation alone peaks at 10.3–11.2 MB on CPython 3.10–3.13, as in v1.0.0, and releases all but
0.4–0.9 MB before estimation starts. `metadata.json` holds the full measurements (both machines, all windows, per-stage time and
heap, versions). Reproduce with `python tools/measure.py INPUT`.

## Tests (all run on this `main.py`; its SHA-256 is in `metadata.json`)

| Test | What it checks | Result |
|---|---|---|
| T1 contract | rows out = rows in, 8 single-space fields, timestamps = `%d.%09d` of the input ns, decimals, unit quaternions (`tools/check_output.py`) | 34/34 official-format files, on macOS arm64 and on Linux x86-64 |
| T2 regression | byte identity with the registered research build of the same configuration | 986/986 harness missions (5 mission sets, both orders, 17 sequences) |
| T3 protocol | 340 missions, the accuracy table above | 0 contract failures, 0 fallbacks |
| T4 robustness | the 935 inputs of version 1.0 (29-condition stress suite, official-degradation windows, rotated sensor frames) and 799 held-out inputs (the stress suite with a new seed, windows of seed-7 files) | 0 errors, 0 fallbacks; mean ATE vs v1.0.0: −2.8 % (stress), −7.2 % (windows), −13.4 % (rotations), −3.3 % / −6.1 % on the held-out sets; one bias-mode decision changes (intended, from the new rest detector); the cases more than 25 % worse are listed under Limitations |
| T5 efficiency | heap and runtime (§ Efficiency) | own heap ≤ 2.64 MB at 3 min, 6.86 MB at 10 min; worst runtime 11.7 ms per IMU-second (2.1 GHz Xeon) |
| T6 cold start | the `Dockerfile` built and run (`python:3.12-slim`, Python 3.12.16, NumPy 2.5.5, Linux arm64); three fresh macOS environments with only `main.py`, `requirements.txt` and NumPy from PyPI (Python 3.13 + NumPy 2.5.5, Python 3.10 + NumPy 1.22.4 and 1.26.4); Linux x86-64 (Python 3.13.18, NumPy 2.5.3) | every output byte-identical to the development build |
| T7 edge cases | 37 inputs: headers, CRLF, blank lines, spaces, scientific notation, nested zips, no blind window, 1–2 rows, all-zero, duplicates, NaN, long gaps, non-monotonic time, unit errors, CLI forms | 37/37 (`tests/test_edge_cases.py`) |
| renderer | the output formatter against `%`-formatting | 10^7 random rows and the edge cases: 0 mismatches (`tests/test_render.py`) |

## Limitations (stated plainly)

- **Physics.** Low-frequency position is not observable from an IMU. On the Machine Hall flights (20–40 m paths in a large hall)
  the ATE stays at about 2.0–3.9 m, 10–49 % below the motionless floor; the blind-window error is dominated by the error
  already present when the IMU dies (a perfect predictor of the blind motion would lower it by only 6–8 % on EuRoC).
- **Where 1.1.0 is not better.** The TUM-VI gain comes from three of the six rooms (room4 −15 %, room6 −20 %, room1 −7 %;
  room3 +6 %, room5 +2 %) and is not significant with six rooms. The anchored blind displacement (the motion predicted inside the
  window, measured from the last pose) is slightly worse on average (EuRoC +2.2 %, TUM-VI +1.1 %, both within noise). The
  10-minute rotor input runs about 26 % longer.
- **Robustness beyond the official script.** In our stress suites, an accelerometer bias ramp to 1 m/s² (more than ten times
  the official bias walk) triggers the steady drag offset in small-room flights (ATE +88 to +182 % in three cases) and costs
  +29 to +51 % in some Machine Hall and hand-held cases; a slow gyro ramp on a hand-held recording with the long tilt smoother
  costs up to +79 %; recordings that start at 60 % of a room5 file +25 to +49 %; a 15 % prefix of a drone file (mostly on the
  ground, classified hand-held) up to +80 % (0.15 → 0.27 m). None of these situations occurs in a full recording degraded by
  the organisers' script, where no sequence is more than 6 % worse on average.
- **Validation scope.** Four environments, two platforms, two IMUs, sequences of at most 184 s. Hidden data from another platform
  (for example a head-mounted device) could sit near the platform-gate threshold; the radius guard is the second line of defence.
- **Unknown evaluation details.** The tracemalloc window, the grading machine, the column order fed to the degradation script and
  the exact waypoint / blind / dropout definitions are not published; every number above is our own measurement.
- **Very long inputs.** Every large structure is linear in the input length (about 50 bytes per IMU sample, 10 kB per
  IMU-second at 200 Hz): 2.6 MB own heap at 3 minutes, 6.9 MB at 10 minutes, so 8 MB is reached at about 12 minutes (about
  11 minutes in the stricter after-NumPy window), extrapolated from the two measurements. In the rare case where the
  gravity-health check fails (1 of 374 rotor missions in our tests) the gyro-bias start is a full-rate array (24 more bytes per
  sample): 3.5 MB at 3 minutes but 9.7 MB at 10 minutes, so on that path the limit is reached at about 8 minutes.
- **Sensor-frame sensitivity.** Under the column-order-B degradation (gyro noise ~0.18 rad/s) the rotor estimate moves by
  0.1–0.3 m between rotated sensor frames, without a systematic penalty.

## Files

| Path | Content |
|---|---|
| `main.py` | the estimator (everything needed at inference) |
| `requirements.txt`, `Dockerfile` | environment |
| `tools/check_output.py` | output-contract checker (standard library only) |
| `tools/measure.py` | heap (four windows), runtime and start-up measurement |
| `tools/make_samples.py` | official-format inputs from local EuRoC / TUM-VI copies with the organisers' `degrade_imu.py` (both column orders, any seeds) |
| `tools/evaluate.py` | SE(3) Umeyama ATE, waypoint, blind and dropout proxies, RPE; `--protocol` reproduces the accuracy table |
| `tests/test_edge_cases.py`, `tests/test_render.py` | edge cases (T7); output renderer vs `%`-formatting on 10^7 random rows |
| `metadata.json` | measured memory and runtime, machines, versions |

## Data and licences

No dataset file is included. EuRoC MAV: M. Burri et al., "The EuRoC micro aerial vehicle datasets", IJRR 2016; ETH Research
Collection, DOI 10.3929/ethz-b-000690084, "In Copyright – Non-Commercial Use Permitted". TUM-VI: D. Schubert et al., "The TUM VI
Benchmark for Evaluating Visual-Inertial Odometry", IROS 2018; data licensed CC BY 4.0. The organisers' `degrade_imu.py` was used
unmodified (SHA-256 `cc7398db…2f31`) and is not redistributed here.
