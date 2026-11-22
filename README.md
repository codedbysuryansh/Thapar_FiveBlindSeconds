# Five Blind Seconds

**IMU-only 6-DoF trajectory estimation under sensor degradation, with a five-second blind prediction.**
Samsung PRISM XR Hackathon 2026 · Track 03 "Path Finder: Trajectory Estimation Under Sensor Degradation"

Team **Thapar_FiveBlindSeconds** · Thapar Institute of Engineering & Technology, Patiala
Suryansh Singh · Sudiksha Sharma · Arpit Kumar · Vishwas Sethi

**[▶ Live demo — run main.py in your browser (no install)](https://codedbysuryansh.github.io/Thapar_FiveBlindSeconds/)**

---

## What is in this repository

| Deliverable | Where |
|---|---|
| Solution: one standalone Python file (Python ≥ 3.9 + NumPy ≥ 1.22, one CPU thread) | [`src/main.py`](src/main.py), version 1.1.0 |
| How to run it, how it works, every test and its limitations | [`src/README.md`](src/README.md) |
| Submission bundle (`estimated_trajectory.txt`, `metadata.json`, `src/`) | [`Thapar_FiveBlindSeconds.zip`](Thapar_FiveBlindSeconds.zip) |
| Sample output and metadata (the bundle's top level, also unpacked here) | [`estimated_trajectory.txt`](estimated_trajectory.txt), [`metadata.json`](metadata.json) |
| Presentation (`CollegeName_TeamName`) | [`presentation/Thapar_FiveBlindSeconds.pdf`](presentation/Thapar_FiveBlindSeconds.pdf) and [`.pptx`](presentation/Thapar_FiveBlindSeconds.pptx) |
| Docker | [`src/Dockerfile`](src/Dockerfile) |
| Research, build and optimisation reports | [`docs/`](docs/) |
| Research and optimisation code, results and registry | [`research/`](research/), [`opt/`](opt/) |

The `estimated_trajectory.txt` here is `main.py`'s output for the official-format input MH_01_easy (column order A, seed 42),
made from the public EuRoC sequence with the organisers' degradation script. No dataset files are included.

## Quick start

```bash
git clone https://github.com/codedbysuryansh/Thapar_FiveBlindSeconds
cd Thapar_FiveBlindSeconds/src
pip install -r requirements.txt                        # numpy>=1.22, nothing else
python main.py path/to/input.zip estimated_trajectory.txt
```

`INPUT` is an EuRoC-format IMU recording (`imu0/data.csv`), zipped or not, or the CSV itself. The output is
`estimated_trajectory.txt` in TUM format (`timestamp tx ty tz qx qy qz qw`, timestamps in seconds written from the input's
integer nanoseconds): exactly one pose per input row, including duplicated rows and the rows of the final five blind seconds.

With Docker (run from `src/`, input in the same folder):

```bash
docker build -t fiveblindseconds .
docker run --rm -v "$PWD":/data fiveblindseconds input.zip estimated_trajectory.txt
```

Check the output contract and score against ground truth (scoring only, never at inference):

```bash
python tools/check_output.py input.zip estimated_trajectory.txt
python tools/evaluate.py estimated_trajectory.txt input.zip path/to/mav0/state_groundtruth_estimate0/data.csv
```

## Results (our evaluation, not an official score)

The organisers' `degrade_imu.py`, unmodified, on the 17 public sequences (11 EuRoC MAV flights, 6 TUM-VI rooms), both
plausible column orders, seeds 42 and 5–13: 340 missions. SE(3) Umeyama alignment over all output rows, as the FAQ describes.
"Standing still" (every position zero) is the floor any IMU-only method must beat.

| Error (m) | EuRoC, standing still | EuRoC, **ours** | TUM-VI, standing still | TUM-VI, **ours** |
|---|---|---|---|---|
| ATE (RMSE) | 3.448 | **2.294** | 1.080 | **0.418** |
| 8 waypoints | 3.248 | **2.096** | 1.110 | **0.343** |
| 5 s blind window, mean | 3.083 | **2.675** | 0.614 | **0.593** |
| 1 s after each dropout | 3.045 | **1.982** | 1.025 | **0.387** |

Under the same protocol, textbook double integration (with our own attitude) gives 25.9 m / 29.9 m and the best classical
baseline we found (a high-pass "heave" integrator, cut-off chosen on held-out environments) 3.263 m / 0.550 m.

Efficiency, measured on this `main.py` (full table in [`src/README.md`](src/README.md#efficiency-measured-on-this-mainpy-complete-program)):
own heap 2.64 MB on a 3-minute flight and 6.86 MB on a 10-minute one (limit 8 MB); 5.0 ms per IMU-second on an Apple M3 Pro
and 9.2 ms on a 2.1 GHz Xeon for that flight, worst input 11.7 ms (limit 20 ms); one CPU thread, no network.

Tests: output contract on 34/34 official-format files, byte-identical on macOS arm64, Linux x86-64 and Docker; 986/986
regression missions; 1,734 robustness inputs with 0 crashes and 0 fallbacks; 37/37 edge cases.

## How it works, in one paragraph

A strapdown integrator is a blindfolded walker counting steps: every miscount stays forever, so position error grows at least
with the square of time. The task is offline, so `main.py` instead asks which smooth, physically plausible motion of a drone or
of a hand-held device best explains every accelerometer sample, before and after each moment. It rebuilds the time grid (zero,
NaN and repeated rows are invalid, missing rows are re-inserted, the dead tail is the blind window), estimates attitude with a
gravity smoother whose uncertainty grows with the square of each gap's length, decides from the rigidity of the thrust axis
whether the device is a drone or hand-held, and then runs a Kalman filter and Rauch–Tung–Striebel smoother on a motion prior:
a 17-state spring-damper with rotor drag and lever arm for drones, a Matérn-5/2 prior for hand-held devices, zero velocity at
genuine rest. The five blind seconds use the drone's last smoothed velocity decaying with a 0.5 s time constant, or the hand-held
smoother's own prediction. The nine stages are listed in [`src/README.md`](src/README.md#how-it-works).

## Repository layout

```
Thapar_FiveBlindSeconds/
├── src/                          solution: main.py, README.md, requirements.txt, Dockerfile, metadata.json
│   ├── tools/                    check_output.py, evaluate.py, make_samples.py, measure.py
│   └── tests/                    test_edge_cases.py, test_render.py
├── Thapar_FiveBlindSeconds.zip   submission bundle: estimated_trajectory.txt, metadata.json, src/
├── estimated_trajectory.txt      sample output (MH_01_easy, order A, seed 42)
├── metadata.json                 version, SHA-256 of main.py, measured memory and runtime
├── presentation/                 Thapar_FiveBlindSeconds.pdf / .pptx
├── docs/                         Phase 3 research and architecture freeze, Phase 4 build, Phase 5 optimisation; figures
├── research/                     research harness, experiment scripts and result summaries (E01–E65, Phase 4 tests)
└── opt/                          Phase 5 optimisation harness, experiment registry, analyses, learned-model study
```

## Reproducing the evaluation

You need your own copies of EuRoC MAV and TUM-VI and the organisers' `degrade_imu.py` (part of the participant package; not
redistributed here). Then, from `src/`:

```bash
python tools/make_samples.py --degrade path/to/degrade_imu.py --euroc path/to/euroc --tumvi path/to/tumvi --out inputs --seeds 42
python tools/evaluate.py --protocol --degrade path/to/degrade_imu.py --euroc path/to/euroc --tumvi path/to/tumvi \
       --seeds 42,5,6,7,8,9,10,11,12,13 --procs 10
```

The first command rebuilds the 34 official-format inputs (their CSV content byte for byte); the second runs the whole 340-mission protocol through
`main.py` and scores the written files (our ATE equals `evo_ape tum gt.txt estimated_trajectory.txt -a` to 5×10⁻⁷ m).

## Limitations, briefly

Slow position drift is not observable from an IMU: in the large Machine Hall flights the ATE stays at 2.0–3.9 m, only 10–49 %
below standing still. The blind window is information-limited (hand-held 0.593 m against 0.614 m standing still). The
organisers' memory-measurement window is not published: our own heap is 2.64 MB, but if tracing starts before `main.py` is
compiled, compilation alone peaks at 10.3–11.2 MB. The full list is in [`src/README.md`](src/README.md#limitations-stated-plainly).

## Data and credits

- EuRoC MAV: M. Burri et al., "The EuRoC micro aerial vehicle datasets", *International Journal of Robotics Research*, 2016.
- TUM-VI: D. Schubert et al., "The TUM VI Benchmark for Evaluating Visual-Inertial Odometry", *IROS*, 2018.
- Degradation: the organisers' `degrade_imu.py`. Scoring cross-checked against the open-source `evo` package.

Both datasets were used for development and scoring only; no dataset files are redistributed in this repository.
