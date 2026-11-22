"""Long synthetic inputs for the Phase 4 memory / runtime test (T5): official-format degraded files (seed 42, order B) of one
platform concatenated end to end with contiguous timestamps, cut to DURATION seconds, and closed by a 5 s blind window of
zero rows (as in the official format). Content is real IMU data; there is no ground truth for the long file (not scored).
usage: p4_make_long.py INPUT_DIR OUT_DIR [DURATION=600]"""
import sys, os, zipfile, io, numpy as np
SETS = {"rotor": ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult", "V1_01_easy", "V2_01_easy"],
        "hand": ["room1", "room2", "room3", "room4", "room5", "room6"]}
HEADER = "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"
def read(path):
    with zipfile.ZipFile(path) as z: raw = z.read("imu0/data.csv").decode()
    lines = [l for l in raw.splitlines()[1:] if l.strip()]; ts = np.array([int(l.split(",", 1)[0]) for l in lines], np.int64)
    return ts, [l.split(",", 1)[1] for l in lines]
def build(inp, names, dur):
    T = []; V = []; t_next = 1_600_000_000_000_000_000
    for n in names:
        ts, vals = read(os.path.join(inp, n + "_B42.zip")); d = np.diff(ts); med = int(np.median(d[d > 0]))
        T.append(ts - ts[0] + t_next); V += vals; t_next = int(T[-1][-1]) + med
        if (np.concatenate(T)[-1] - T[0][0]) * 1e-9 >= dur: break
    ts = np.concatenate(T); keep = (ts - ts[0]) * 1e-9 < dur - 5.0; ts = ts[keep]; V = [v for v, k in zip(V, keep) if k]
    med = int(np.median(np.diff(ts)[np.diff(ts) > 0])); nb = int(round(5.0 / (med * 1e-9)))
    tb = ts[-1] + med * np.arange(1, nb + 1); out = [HEADER] + ["%d,%s" % (a, v) for a, v in zip(ts.tolist(), V)] + ["%d,0.0,0.0,0.0,0.0,0.0,0.0" % a for a in tb.tolist()]
    return "\n".join(out) + "\n", len(out) - 1, (tb[-1] - ts[0]) * 1e-9
if __name__ == "__main__":
    inp, outd = sys.argv[1], sys.argv[2]; dur = float(sys.argv[3]) if len(sys.argv) > 3 else 600.0; os.makedirs(outd, exist_ok=True)
    for kind, names in SETS.items():
        txt, rows, span = build(inp, names, dur); p = os.path.join(outd, "long_%s_%ds.zip" % (kind, int(dur)))
        with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z: z.writestr("imu0/data.csv", txt)
        print("%s: %d rows, %.1f s -> %s" % (kind, rows, span, p))
