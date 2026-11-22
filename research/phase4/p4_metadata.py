#!/usr/bin/env python3
"""Builds release/metadata.json (the optional submission metadata: measured memory and runtime, machines, versions) from the
T5 measurement files research/phase4/results/p4_t5_*.json (one per machine, written by p4_t5_measure.py / tools/measure.py).
Nothing is estimated here: every number is copied from a measurement file."""
import os, sys, json, glob, hashlib, datetime
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REL = os.path.join(ROOT, "release"); RES = os.path.join(ROOT, "research", "phase4", "results")
def sha256(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def r3(x): return round(float(x), 3)
def entry(r):
    h = r["heap_MB"]; s = r["startup_s"]
    return dict(rows=r["rows"], imu_seconds=r3(r["imu_seconds"]),
                heap_peak_MB=dict(own_after_imports=r3(h["after_imports"]), after_numpy_import=r3(h["after_numpy"]),
                                  whole_program=r3(h["whole_program"]), with_interpreter_startup=r3(h["interpreter"])),
                runtime_ms_per_imu_second=dict(min=r3(r["ms_per_imu_second"]["min"]), median=r3(r["ms_per_imu_second"]["median"])),
                run_seconds=dict(min=r3(r["run_s"]["min"]), median=r3(r["run_s"]["median"])),
                startup_seconds_median=dict(python_only=r3(s["python_only"]["median"]), with_import_numpy=r3(s["import_numpy"]["median"]),
                                            with_import_main=r3(s["import_main"]["median"]), end_to_end_command=r3(s["end_to_end"]["median"])),
                end_to_end_ms_per_imu_second=r3(r["end_to_end_ms_per_imu_second"]),
                stage_seconds={k: r3(v) for k, v in r["stage_s"].items()}, stage_heap_peak_MB={k: r3(v) for k, v in r["stage_heap_MB"].items()})
if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(RES, "p4_t5_*.json")))
    if not files: sys.exit("no p4_t5_*.json measurement files")
    machines = []
    for f in files:
        d = json.load(open(f)); first = next(iter(d.values())); m = first["machine"]
        machines.append(dict(source=os.path.relpath(f, ROOT), cpu=m.get("cpu", "?"), platform=m["platform"], python=m["python"], numpy=m["numpy"],
                             inputs={k: entry(v) for k, v in d.items()}))
    meta = dict(team="Thapar_FiveBlindSeconds", track="03 Path Finder: Trajectory Estimation Under Sensor Degradation",
                entry_point="main.py", main_py_sha256=sha256(os.path.join(REL, "main.py")), requirements=open(os.path.join(REL, "requirements.txt")).read().split(),
                configuration="v5 / V5P4 (gravity smoother 12.5 Hz, rotor smoother 25 Hz, hand-held smoother 100 Hz); k_drag 0.1873 1/s",
                single_thread=True, network=False, generated=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                definitions=dict(MB="10^6 bytes", heap="tracemalloc peak of the complete program run (main.main on the input, output file written)",
                                 own_after_imports="tracing starts after `import numpy` and `import main`: the estimator's data structures and buffers",
                                 after_numpy_import="tracing starts after `import numpy`, before `import main`",
                                 whole_program="tracing starts before `import numpy` (NumPy's import alone is 7.5-8.1 MB)",
                                 with_interpreter_startup="`python -X tracemalloc`",
                                 runtime="wall clock of main.main inside a process whose imports are done (start-up excluded), single thread",
                                 startup="interpreter start, + import numpy, + import main, and the full `python main.py INPUT OUTPUT` command, medians"),
                machines=machines)
    out = os.path.join(REL, "metadata.json"); json.dump(meta, open(out, "w"), indent=1); print("wrote", out)
