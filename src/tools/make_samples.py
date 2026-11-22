#!/usr/bin/env python3
"""Make official-format test inputs from local EuRoC MAV / TUM-VI copies with the organisers' degrade_imu.py (unmodified).

    python tools/make_samples.py --degrade path/to/degrade_imu.py --euroc DIR --tumvi DIR --out DIR [--orders A,B] [--seeds 42]
                                 [--max-level 1.0] [--gamma 1.7] [--sequences MH_01_easy,room2,...]

Writes OUT/<sequence>_<order><seed>.zip containing imu0/data.csv in the format of the evaluation data (slide 15 of the
organisers' deck): EuRoC header and column order (timestamp [ns], gyro xyz [rad/s], accel xyz [m/s^2]).

Column order. The script's docstring expects (timestamp, accel, gyro), while EuRoC files are (timestamp, gyro, accel); which
one the organisers fed it is not published, so both are offered: order A re-orders the columns as the docstring says before
degrading; order B passes the EuRoC file as it is (the script then treats the gyro columns as accelerometer columns). The
degraded values are written the way degrade_imu.py writes them (pandas to_csv). The script fixes SEED = 42 in its own main();
other seeds are passed to its degrade() function. Nothing is downloaded; dataset files stay where they are (EuRoC licence).
Requires NumPy and pandas (the organisers' script needs pandas)."""
import os, sys, io, json, hashlib, zipfile, argparse, importlib.util
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"          # one thread, as main.py runs (set before NumPy loads its BLAS; inherited by worker processes)
import numpy as np, pandas as pd
EUROC = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult", "V1_01_easy", "V1_02_medium", "V1_03_difficult",
         "V2_01_easy", "V2_02_medium", "V2_03_difficult"]
TUMVI = ["room%d" % i for i in range(1, 7)]
EUROC_COLS = ["#timestamp [ns]", "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]", "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"]
DOC_COLS = ["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
DEGRADE_SHA256 = "cc7398db47efe8cefb581c13501f43687db2a5a13e7ed89697dc98afcd4b2f31"   # the organisers' degrade_imu.py we used
def load_degrade(path):
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if h != DEGRADE_SHA256: sys.stderr.write("warning: %s has SHA-256 %s, not the version we used (%s)\n" % (path, h, DEGRADE_SHA256))
    spec = importlib.util.spec_from_file_location("degrade_imu", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def seq_dir(name, euroc, tumvi):
    """mav0 folder of a sequence: EuRoC DIR/<name>/mav0, TUM-VI DIR/dataset-<room>_512_16/mav0."""
    return os.path.join(tumvi, "dataset-%s_512_16" % name, "mav0") if name.startswith("room") else os.path.join(euroc, name, "mav0")
def degraded_csv(mav0, order, seed, deg, max_level=1.0, gamma=1.7):
    """(csv text in the evaluation format, degrade statistics) for one sequence, column order and seed."""
    imu = pd.read_csv(os.path.join(mav0, "imu0", "data.csv"), comment="#", header=None)
    tn = imu[0].values.astype(np.int64); w = imu.iloc[:, 1:4].values.astype(float); a = imu.iloc[:, 4:7].values.astype(float)
    if order == "A": df = pd.DataFrame(dict(zip(DOC_COLS, [tn, a[:, 0], a[:, 1], a[:, 2], w[:, 0], w[:, 1], w[:, 2]])))
    elif order == "B": df = pd.DataFrame(dict(zip(EUROC_COLS, [tn, w[:, 0], w[:, 1], w[:, 2], a[:, 0], a[:, 1], a[:, 2]])))
    else: raise ValueError("order must be A or B")
    cfg = dict(deg.CONFIG); cfg["max_level"] = max_level; cfg["gamma"] = gamma
    out, st = deg.degrade(df, cfg, seed); tno = out.iloc[:, 0].to_numpy(np.int64)
    if order == "A": a_o = out.iloc[:, 1:4].to_numpy(float); w_o = out.iloc[:, 4:7].to_numpy(float)
    else: w_o = out.iloc[:, 1:4].to_numpy(float); a_o = out.iloc[:, 4:7].to_numpy(float)
    cols = [tno, w_o[:, 0], w_o[:, 1], w_o[:, 2], a_o[:, 0], a_o[:, 1], a_o[:, 2]]
    return pd.DataFrame(dict(zip(EUROC_COLS, cols))).to_csv(index=False), st
def write_zip(path, csv_text):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z: z.writestr("imu0/data.csv", csv_text)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--degrade", required=True, help="path to the organisers' degrade_imu.py"); ap.add_argument("--euroc", default=""); ap.add_argument("--tumvi", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--orders", default="A,B"); ap.add_argument("--seeds", default="42")
    ap.add_argument("--max-level", type=float, default=1.0); ap.add_argument("--gamma", type=float, default=1.7); ap.add_argument("--sequences", default="")
    a = ap.parse_args(); deg = load_degrade(a.degrade); os.makedirs(a.out, exist_ok=True)
    names = [s for s in a.sequences.split(",") if s] or [n for n in EUROC + TUMVI if os.path.exists(os.path.join(seq_dir(n, a.euroc, a.tumvi), "imu0", "data.csv"))]
    if not names: sys.exit("no sequences found (give --euroc and/or --tumvi)")
    meta = {}
    for n in names:
        for o in a.orders.split(","):
            for sd in (int(x) for x in a.seeds.split(",")):
                tag = "%s_%s%d" % (n, o, sd); txt, st = degraded_csv(seq_dir(n, a.euroc, a.tumvi), o, sd, deg, a.max_level, a.gamma)
                write_zip(os.path.join(a.out, tag + ".zip"), txt); meta[tag] = st; print("%-24s rows %6d -> %6d" % (tag, st["rows_in"], st["rows_out"]), flush=True)
    json.dump(meta, open(os.path.join(a.out, "degrade_stats.json"), "w"), indent=1)
