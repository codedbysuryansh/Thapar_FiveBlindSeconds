#!/usr/bin/env python3
"""Phase 4 test T7: input edge cases. Every case runs the real command line in a fresh process and must exit 0 with an output that
passes tools/check_output.py (one row per input row, format, timestamps, unit quaternions); format-only variants must give
byte-identical output to the plain file. Uses a synthetic IMU stream (no dataset needed).

    python tests/test_edge_cases.py            (exit code 0 when all cases pass)"""
import os, sys, io, json, shutil, subprocess, tempfile, zipfile, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REL = os.path.dirname(HERE); MAIN = os.path.join(REL, "main.py")
sys.path.insert(0, os.path.join(REL, "tools")); import check_output as CO                                   # noqa: E402
HEADER = "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"
T0 = 1403636579758555392
def synth(seconds=40.0, fs=200.0, blind=5.0, seed=0):
    """Smooth random rotation and motion of a hand-held-like device; gyro (rad/s) and specific force (m/s^2) in the body frame."""
    rng = np.random.default_rng(seed); n = int(seconds * fs); t = np.arange(n) / fs; w = np.zeros((n, 3)); acc = np.zeros((n, 3))
    for j in range(3):
        for _ in range(3):
            A, f, ph = rng.uniform(0.1, 0.4), rng.uniform(0.05, 0.8), rng.uniform(0, 2 * np.pi); w[:, j] += A * np.sin(2 * np.pi * f * t + ph)
            B, g, ps = rng.uniform(0.1, 0.6), rng.uniform(0.05, 0.5), rng.uniform(0, 2 * np.pi); acc[:, j] += B * np.sin(2 * np.pi * g * t + ps)
    q = np.array([0, 0, 0, 1.0]); f_b = np.empty((n, 3)); grav = np.array([0, 0, 9.81])
    for i in range(n):
        x, y, z, s = q; R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - s * z), 2 * (x * z + s * y)], [2 * (x * y + s * z), 1 - 2 * (x * x + z * z), 2 * (y * z - s * x)],
                                      [2 * (x * z - s * y), 2 * (y * z + s * x), 1 - 2 * (x * x + y * y)]])
        f_b[i] = R.T @ (acc[i] + grav); r = w[i] / fs; th = np.linalg.norm(r); k = np.sin(th / 2) / th if th > 1e-12 else 0.5; d = np.array([r[0] * k, r[1] * k, r[2] * k, np.cos(th / 2)])
        q = np.array([s * d[0] + x * d[3] + y * d[2] - z * d[1], s * d[1] - x * d[2] + y * d[3] + z * d[0], s * d[2] + x * d[1] - y * d[0] + z * d[3], s * d[3] - x * d[0] - y * d[1] - z * d[2]]); q /= np.linalg.norm(q)
    W = np.round(w + rng.normal(0, 2e-3, (n, 3)), 6); F = np.round(f_b + rng.normal(0, 2e-2, (n, 3)), 6); nb = int(blind * fs)
    if nb: W[-nb:] = 0.0; F[-nb:] = 0.0
    ts = T0 + np.round(t * 1e9).astype(np.int64); return ts, W, F
def csv(ts, W, F, header=HEADER, fmt="%r", sep=",", nl="\n", ts_fmt="%d"):
    out = [header] if header else []
    for a, w, f in zip(ts.tolist(), W.tolist(), F.tolist()): out.append(sep.join([ts_fmt % a] + [fmt % x for x in w + f]))
    return (nl.join(out) + nl).encode()
def run(inp, out, args=None, cwd=None):
    p = subprocess.run([sys.executable, MAIN] + (args if args is not None else [inp, out]), capture_output=True, text=True, cwd=cwd); return p.returncode, p.stderr
def main():
    tmp = tempfile.mkdtemp(prefix="fbs_t7_"); ts, W, F = synth(); base = os.path.join(tmp, "base.csv"); open(base, "wb").write(csv(ts, W, F))
    rc, err = run(base, os.path.join(tmp, "base.out")); ref = open(os.path.join(tmp, "base.out"), "rb").read(); results = []
    def case(name, data=None, path=None, same=False, args=None, cwd=None, expect_rows=None, out=None):
        ip = path or os.path.join(tmp, name + ".csv")
        if data is not None: open(ip, "wb").write(data)
        op = out or os.path.join(tmp, name + ".out"); rc, err = run(ip, op, args, cwd)
        if os.path.isdir(op): op = os.path.join(op, "estimated_trajectory.txt")
        r = dict(case=name, exit=rc, log=(err.strip().splitlines() or [""])[-1])
        if rc != 0 or not os.path.exists(op): r["ok"] = False; r["why"] = "exit %d / no output" % rc; results.append(r); return r
        if expect_rows is not None: n = open(op, "rb").read().count(b"\n"); r["ok"] = n == expect_rows; r["why"] = "rows %d (expected %d)" % (n, expect_rows)
        else:
            c = CO.check(ip, op); r["ok"] = c["ok"]; r["why"] = "; ".join(c["problems"]); r["rows"] = c["rows_out"]
        if same and r["ok"]: r["ok"] = open(op, "rb").read() == ref; r["why"] = (r["why"] + " identical-to-plain=%s" % r["ok"]).strip()
        results.append(r); return r
    results.append(dict(case="plain", exit=rc, ok=rc == 0 and CO.check(base, os.path.join(tmp, "base.out"))["ok"], why="", log=(err.strip().splitlines() or [""])[-1]))
    case("crlf", csv(ts, W, F, nl="\r\n"), same=True)
    case("trailing_blank_lines", csv(ts, W, F) + b"\n\n  \n", same=True)
    case("blank_lines_inside", csv(ts, W, F).replace(b"\n", b"\n\n", 50), same=True)
    case("spaces_after_commas", csv(ts, W, F, sep=", "), same=True)
    case("no_header", csv(ts, W, F, header=None), same=True)
    case("header_words", csv(ts, W, F, header="timestamp,gx,gy,gz,ax,ay,az"), same=True)
    case("comment_lines", csv(ts, W, F, header=HEADER + "\n# a comment\n# another"), same=True)
    case("scientific_notation", csv(ts, W, F, fmt="%.6e"))
    case("tab_separated", csv(ts, W, F, sep="\t"), same=True)
    zp = os.path.join(tmp, "nested.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z: z.writestr("mav0/imu0/data.csv", csv(ts, W, F)); z.writestr("mav0/imu0/sensor.yaml", "rate_hz: 200\n")
    case("zip_nested_folder", path=zp, same=True)
    zs = os.path.join(tmp, "stored.zip")
    with zipfile.ZipFile(zs, "w", zipfile.ZIP_STORED) as z: z.writestr("data.csv", csv(ts, W, F))
    case("zip_stored", path=zs, same=True)
    zb = os.path.join(tmp, "bzip.zip")
    with zipfile.ZipFile(zb, "w", zipfile.ZIP_BZIP2) as z: z.writestr("imu0/data.csv", csv(ts, W, F))
    case("zip_bzip2_fallback", path=zb, same=True)
    fo = os.path.join(tmp, "folder", "mav0", "imu0"); os.makedirs(fo); open(os.path.join(fo, "data.csv"), "wb").write(csv(ts, W, F))
    case("folder_input", path=os.path.join(tmp, "folder"), same=True)
    t2, W2, F2 = synth(blind=0.0); case("no_blind_window", csv(t2, W2, F2))
    case("one_row", csv(ts[:1], W[:1], F[:1])); case("two_rows", csv(ts[:2], W[:2], F[:2])); case("ten_rows", csv(ts[:10], W[:10], F[:10]))
    case("all_zero_values", csv(ts, W * 0, F * 0)); case("empty_file", b"", expect_rows=0); case("header_only", (HEADER + "\n").encode(), expect_rows=0)
    case("constant_values", csv(ts, W * 0 + 0.01, F * 0 + 9.81))
    idx = np.sort(np.concatenate([np.arange(len(ts)), np.arange(100, len(ts), 97)])); case("duplicated_timestamps", csv(ts[idx], W[idx], F[idx]))
    b = csv(ts, W, F).split(b"\n"); b[200] = b[200].replace(b",", b",nan,", 1).rsplit(b",", 1)[0]; b[300] = b[300].replace(b",", b",,", 1).rsplit(b",", 1)[0]
    b[400] = b"%d,1.0,2.0" % ts[399]; b[500] = b"%d,abc,0,0,0,0,0" % ts[499]; case("nan_and_broken_fields", b"\n".join(b))
    p = np.arange(len(ts)); p[1000:1010] = p[1000:1010][::-1]; p[[3000, 3001]] = p[[3001, 3000]]; case("non_monotonic_timestamps", csv(ts[p], W[p], F[p]))
    case("timestamps_in_seconds", csv(ts / 1e9, W, F, ts_fmt="%.6f")); case("timestamps_in_microseconds", csv(ts // 1000, W, F))
    case("accelerometer_in_g", csv(ts, W, np.round(F / 9.80665, 6))); case("gyro_in_deg_per_s", csv(ts, np.round(np.degrees(W), 4), F))
    t3, W3, F3 = synth(seconds=3.0, blind=0.5); case("very_short_3s", csv(t3, W3, F3))
    t4, W4, F4 = synth(seconds=100.0); keep = (np.arange(len(t4)) < 2000) | (np.arange(len(t4)) >= 15000); case("gap_65s", csv(t4[keep], W4[keep], F4[keep]))
    Wh = W.copy(); Wh[[700, 701, 2500]] = 5000.0; case("huge_values", csv(ts, Wh, F))
    ex = [b"%s,1.5,2.5" % ln if ln and not ln.startswith(b"#") else ln for ln in csv(ts, W, F).split(b"\n")]; case("extra_columns", b"\n".join(ex))
    od = os.path.join(tmp, "outdir"); os.makedirs(od); case("output_is_a_folder", path=base, out=od + os.sep)
    case("flags_i_o", path=base, args=["-i", base, "-o", os.path.join(tmp, "flags.out")], out=os.path.join(tmp, "flags.out"))
    wd = os.path.join(tmp, "cwd"); os.makedirs(wd); shutil.copy(base, os.path.join(wd, "data.csv")); case("no_arguments", path=os.path.join(wd, "data.csv"), args=[], cwd=wd, out=os.path.join(wd, "estimated_trajectory.txt"))
    case("missing_input", path=os.path.join(tmp, "does_not_exist.csv"), expect_rows=0)
    ok = all(r["ok"] for r in results)
    for r in results: print("%-28s %s  %s  %s" % (r["case"], "PASS" if r["ok"] else "FAIL", r.get("why", ""), r.get("log", "")))
    print("T7: %d/%d cases pass -> %s" % (sum(r["ok"] for r in results), len(results), "PASS" if ok else "FAIL"))
    if "--json" in sys.argv: json.dump(results, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
    shutil.rmtree(tmp, ignore_errors=True); return 0 if ok else 1
if __name__ == "__main__":
    sys.exit(main())
