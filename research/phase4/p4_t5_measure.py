#!/usr/bin/env python3
"""Phase 4 test T5 driver: release/tools/measure.py on MH_01 (orders A and B; rotor path, worst case), room2 A (hand-held path)
and the two 10-minute synthetic inputs (p4_make_long.py, built here if missing). Targets (freeze report 12.8): heap after the
NumPy import <= 4 MB on MH_01 and <= 8 MB on a 10-minute input; runtime excluding the import <= 9 ms per IMU-second on the
M3 Pro, <= 17 ms on the 2.1 GHz Xeon cloud vCPU (MH_01 B, room2). Writes research/phase4/results/p4_t5_<host>.json."""
import os, sys, json, subprocess, platform
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "release", "tools")); import measure as ME                          # noqa: E402
INP = os.path.join(ROOT, "data", "phase4", "inputs"); LONG = os.path.join(ROOT, "data", "phase4", "long")
if __name__ == "__main__":
    reps = int(os.environ.get("P4_REPEATS", "5"))
    if not os.path.exists(os.path.join(LONG, "long_rotor_600s.zip")):
        subprocess.run([sys.executable, os.path.join(ROOT, "research", "phase4", "p4_make_long.py"), INP, LONG, "600"], check=True)
    files = [os.path.join(INP, "MH_01_easy_A42.zip"), os.path.join(INP, "MH_01_easy_B42.zip"), os.path.join(INP, "room2_A42.zip"),
             os.path.join(LONG, "long_rotor_600s.zip"), os.path.join(LONG, "long_hand_600s.zip")]
    out = {}; rows = []
    for f in files:
        r = ME.measure(f, reps); out[os.path.basename(f)] = r; h = r["heap_MB"]; s = r["startup_s"]
        rows.append("%-22s %6.1f s  heap MB: own %.3f | +main import %.3f | whole %.3f | +interp %.3f   run %.2f ms/IMU-s (min %.2f)   start-up %.3f s   end-to-end %.2f ms/IMU-s"
                    % (os.path.basename(f), r["imu_seconds"], h["after_imports"], h["after_numpy"], h["whole_program"], h["interpreter"], r["ms_per_imu_second"]["median"],
                       r["ms_per_imu_second"]["min"], s["import_main"]["median"], r["end_to_end_ms_per_imu_second"]))
        print(rows[-1], flush=True)
    host = platform.node().split(".")[0] or "host"; m = out[os.path.basename(files[0])]["machine"]
    print("machine:", m)
    json.dump(out, open(os.path.join(ROOT, "research", "phase4", "results", "p4_t5_%s.json" % host), "w"), indent=1)
