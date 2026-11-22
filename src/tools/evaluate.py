#!/usr/bin/env python3
"""Score estimated trajectories the way the Track 3 FAQ describes (SE(3) Umeyama alignment over all output rows, then the
Absolute Trajectory Error), plus our waypoint, blind-window and after-dropout proxies. NumPy only (the protocol mode also
needs pandas, for the organisers' degradation script).

One file:
    python tools/evaluate.py EST INPUT GROUND_TRUTH [--json out.json] [--gt-tum gt.txt]
        EST           estimated_trajectory.txt written by main.py (TUM format, one row per input row)
        INPUT         the degraded input that main.py was run on (zip, folder or csv) - row timestamps and the blind window
        GROUND_TRUTH  EuRoC mav0/state_groundtruth_estimate0/data.csv or TUM-VI mav0/mocap0/data.csv
        --gt-tum      also write the ground truth at the scored rows in TUM format (for `evo_ape tum gt.txt EST -a`)

Whole protocol (inputs made with the organisers' script, main.py run in-process, motionless reference scored alongside):
    python tools/evaluate.py --protocol --degrade degrade_imu.py --euroc DIR --tumvi DIR [--orders A,B] [--seeds 42,5,6,7,8,9,10,11,12,13]
                             [--k-drag loso|default|VALUE] [--procs N] [--json out.json]

Metric definitions (our proxies; the official waypoint, blind and dropout definitions are not published):
  ate     RMS position error over every row with ground truth, after one SE(3) alignment fitted on those rows
  wp      mean error at 8 evenly spaced times before the blind window ("waypoints")
  b_abs   mean error over the blind rows (the last 5 s); b_end the last row; b_anc the error of the blind-window displacement
          measured from the last valid pose (what a predictor alone controls)
  drop    mean error in the 1 s after every gap of >= 5 missing samples
  ori     RMS orientation error after the same alignment (diagnostic only: the alignment is fitted to positions)
  rpe_t / rpe_r   evo-style relative pose error over 1 s (translation m, rotation deg)
The motionless reference (all positions zero) is the floor any IMU-only method must beat."""
import os, sys, io, json, time, zipfile, tempfile, argparse
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"          # one thread, as main.py runs (set before NumPy loads its BLAS; inherited by worker processes)
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REL = os.path.dirname(HERE)
# rotor-drag coefficients fitted per EuRoC flight against ground-truth velocity (Phase 2, E05). Evaluations use, for each flight,
# the mean of the OTHER flights (leave-one-sequence-out); main.py's default 0.1873 is the mean of all eleven.
K_FIT = {"MH_01_easy": 0.18185813357413066, "MH_02_easy": 0.18420352909988819, "MH_03_medium": 0.19455260281774267, "MH_04_difficult": 0.2122813980204592,
         "MH_05_difficult": 0.20130547872117885, "V1_01_easy": 0.18789887710347664, "V1_02_medium": 0.18694253796390697, "V1_03_difficult": 0.1644647025618498,
         "V2_01_easy": 0.16621865226961258, "V2_02_medium": 0.19842399911304787, "V2_03_difficult": 0.1817155877371721}
def k_loso(name): return float(np.mean([v for k, v in K_FIT.items() if k != name]))
# ---------------------------------------------------------------- quaternions (x, y, z, w), Hamilton, body -> world
def qmul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]; bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)
def qconj(q): return q * np.array([-1.0, -1.0, -1.0, 1.0])
def qrot(q, v):
    u = q[..., :3]; w = q[..., 3:4]; c = np.cross(u, v); return v + 2 * w * c + 2 * np.cross(u, c)
def qangle(q): return np.degrees(2 * np.arctan2(np.linalg.norm(q[..., :3], axis=-1), np.abs(q[..., 3])))
def qexp(rv):
    th = np.linalg.norm(rv, axis=-1, keepdims=True); k = np.where(th > 1e-12, np.sin(th / 2) / np.maximum(th, 1e-300), 0.5); return np.concatenate([rv * k, np.cos(th / 2)], -1)
def qlog(q):
    q = q * np.where(q[..., 3:4] < 0, -1.0, 1.0); n = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    return q[..., :3] * np.where(n > 1e-12, 2 * np.arctan2(n, q[..., 3:4]) / np.maximum(n, 1e-300), 2.0)
def mat2quat(R):
    """Unit quaternion (x, y, z, w) of a rotation matrix (Shepperd's method)."""
    tr = np.trace(R); i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2], tr]))
    if i == 3: w = np.sqrt(1 + tr) / 2; q = [(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w]
    else:
        j, k = (i + 1) % 3, (i + 2) % 3; s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]); q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / (2 * s)]
        q[i] = s / 2; q[j] = (R[j, i] + R[i, j]) / (2 * s); q[k] = (R[k, i] + R[i, k]) / (2 * s)
    q = np.array(q); return q / np.linalg.norm(q)
def slerp(tq, tk, qk):
    """Spherical interpolation of the key rotations qk (times tk) at times tq (clamped to the key range)."""
    tc = np.clip(tq, tk[0], tk[-1]); i = np.clip(np.searchsorted(tk, tc, "right") - 1, 0, len(tk) - 2); u = ((tc - tk[i]) / (tk[i + 1] - tk[i]))[:, None]
    q0 = qk[i]; d = qmul(qconj(q0), qk[i + 1]); return qmul(q0, qexp(u * qlog(d)))
# ---------------------------------------------------------------- readers
def _lines_of(path):
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path); names = [n for n in z.namelist() if n.endswith("imu0/data.csv")] or [n for n in z.namelist() if n.endswith(".csv")]
        return io.TextIOWrapper(z.open(names[0]), encoding="utf-8", errors="replace")
    if os.path.isdir(path):
        for root, _d, files in os.walk(path):
            if "data.csv" in files and root.endswith("imu0"): return open(os.path.join(root, "data.csv"), encoding="utf-8", errors="replace")
        raise FileNotFoundError("no imu0/data.csv under " + path)
    return open(path, encoding="utf-8", errors="replace")
def read_input_ns(path):
    """Row timestamps (int ns) of an input CSV, in row order (comment and header lines skipped)."""
    out = []
    for ln in _lines_of(path):
        s = ln.strip()
        if not s or s[0] == "#" or s[0].isalpha(): continue
        out.append(int(s.split(",")[0].strip()))
    return np.array(out, np.int64)
def read_estimate(path):
    ns, p, q = [], [], []
    for ln in open(path):
        s = ln.split()
        if not s or s[0].startswith("#"): continue
        a, _dot, b = s[0].partition("."); ns.append(int(a) * 1000000000 + int((b + "000000000")[:9]) if b else int(a) * 1000000000)
        p.append([float(x) for x in s[1:4]]); q.append([float(x) for x in s[4:8]])
    return np.array(ns, np.int64), np.array(p, float), np.array(q, float)
def read_gt(path):
    """EuRoC state_groundtruth_estimate0 / TUM-VI mocap0 CSV: t [ns], p xyz, q wxyz, ... -> (ns, p, q xyzw), strictly increasing."""
    ns, g = [], []
    for ln in open(path):
        s = ln.strip()
        if not s or s[0] == "#": continue
        f = s.split(","); ns.append(int(f[0])); g.append([float(x) for x in f[1:8]])
    ns = np.array(ns, np.int64); g = np.array(g, float); keep = np.concatenate([[True], np.diff(ns) > 0])
    return ns[keep], g[keep, 0:3], g[keep][:, [4, 5, 6, 3]]
# ---------------------------------------------------------------- one mission
class Mission: pass
def mission(tno, gt):
    """Everything the score needs about the input rows: ground truth at each row, validity, blind window, gaps."""
    gns, gp, gq = gt; m = Mission(); t0 = tno[0]; m.tn = tno; m.t = (tno - t0) * 1e-9; m.T = (tno[-1] - t0) * 1e-9; m.blind = m.t >= m.T - 5.0
    gtt = (gns - t0) * 1e-9; tt = m.t; idx = np.clip(np.searchsorted(gtt, tt), 1, len(gtt) - 1)
    m.ok = (tt >= gtt[0]) & (tt <= gtt[-1]) & ((gtt[idx] - gtt[idx - 1]) < 0.05)
    m.gp = np.stack([np.interp(tt, gtt, gp[:, i]) for i in range(3)], 1); m.gq = slerp(tt, gtt, gq / np.linalg.norm(gq, axis=1, keepdims=True))
    d = np.diff(tno); med = np.median(d[d > 0]); miss = np.round(d / med).astype(int) - 1; m.gap_after = np.zeros(len(tno), int); m.gap_after[1:] = np.maximum(miss, 0)
    m.fs = 1e9 / med; return m
def umeyama(est, gt):
    """Rigid SE(3) alignment (no scale) est -> gt; a rank-deficient estimate (e.g. motionless) gets R = I."""
    me, mg = est.mean(0), gt.mean(0); H = (est - me).T @ (gt - mg)
    if np.linalg.matrix_rank(H, tol=1e-9) < 2: return np.eye(3), mg - me
    U, S, Vt = np.linalg.svd(H); D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]); Rm = Vt.T @ D @ U.T; return Rm, mg - Rm @ me
KEYS = ["ate", "wp", "b_abs", "b_end", "b_anc", "pre_ate", "drop", "rpe_t", "rpe_r", "ori", "ori_b"]
def score(m, p, q=None, ncp=8):
    out = dict.fromkeys(KEYS, float("nan")); p = np.asarray(p, float); fin = np.isfinite(p).all(1); A = m.ok & fin
    if A.sum() < 10: return out
    Rm, tt = umeyama(p[A], m.gp[A]); er = np.linalg.norm(p @ Rm.T + tt - m.gp, axis=1); out["ate"] = float(np.sqrt(np.mean(er[A] ** 2)))
    pre = A & ~m.blind; b = A & m.blind; Tp = m.T - 5.0; ci = [int(np.argmin(np.abs(np.where(pre, m.t, 1e9) - Tp * (k + 1) / (ncp + 1)))) for k in range(ncp)]
    out["wp"] = float(er[ci].mean()); out["pre_ate"] = float(np.sqrt(np.mean(er[pre] ** 2)))
    if b.any():
        bi = np.where(b)[0]; L = int(np.where(pre)[0][-1]); out["b_abs"] = float(er[bi].mean()); out["b_end"] = float(er[bi[-1]])
        out["b_anc"] = float(np.linalg.norm((p[bi] - p[L]) @ Rm.T - (m.gp[bi] - m.gp[L]), axis=1).mean())
    g = np.where(m.gap_after >= 5)[0]; near = np.zeros(len(p), bool)
    for i in g: near[i:np.searchsorted(m.t, m.t[i] + 1.0)] = True
    near &= pre; out["drop"] = float(er[near].mean()) if near.any() else float("nan")
    if q is not None:
        q = np.asarray(q, float); q = q / np.linalg.norm(q, axis=1, keepdims=True); idx = np.where(A)[0]; qa = mat2quat(Rm)
        qe, qg = q[idx], m.gq[idx]; ang = qangle(qmul(qconj(qmul(np.broadcast_to(qa, qe.shape), qe)), qg))
        out["ori"] = float(np.sqrt(np.mean(ang ** 2))); bb = m.blind[idx]; out["ori_b"] = float(np.sqrt(np.mean(ang[bb] ** 2))) if bb.any() else float("nan")
        d = int(round(m.fs)); k0 = np.arange(0, len(idx) - d, d); k1 = k0 + d; pe, pg = p[idx], m.gp[idx]
        if len(k0):
            te = qrot(qconj(qe[k0]), pe[k1] - pe[k0]); tg = qrot(qconj(qg[k0]), pg[k1] - pg[k0]); dRg = qmul(qconj(qg[k0]), qg[k1]); dRe = qmul(qconj(qe[k0]), qe[k1])
            out["rpe_t"] = float(np.sqrt(np.mean(np.sum((te - tg) ** 2, 1)))); out["rpe_r"] = float(np.sqrt(np.mean(qangle(qmul(qconj(dRg), dRe)) ** 2)))
    return out
def score_files(est, inp, gt_path, gt_tum=None):
    tno = read_input_ns(inp); ens, p, q = read_estimate(est)
    contract = dict(rows_in=int(len(tno)), rows_out=int(len(ens)), timestamps_equal=bool(len(tno) == len(ens) and (tno == ens).all()))
    if not contract["timestamps_equal"]: return dict(contract=contract, error="the estimate does not have one row per input row with the input's timestamps")
    m = mission(tno, read_gt(gt_path)); s = score(m, p, q); s0 = score(m, np.zeros_like(p), np.tile([0.0, 0.0, 0.0, 1.0], (len(p), 1)))
    if gt_tum:
        with open(gt_tum, "w") as f:
            for i in np.where(m.ok)[0]: f.write("%d.%09d %.6f %.6f %.6f %.7f %.7f %.7f %.7f\n" % ((tno[i] // 1000000000, tno[i] % 1000000000) + tuple(m.gp[i]) + tuple(m.gq[i])))
    return dict(contract=contract, score=s, motionless=s0, blind_rows=int(m.blind.sum()), rows_with_gt=int(m.ok.sum()))
# ---------------------------------------------------------------- the protocol
def _job(job):
    name, order, seed, opts = job; sys.path.insert(0, os.path.join(REL, "tools")); sys.path.insert(0, REL)
    import make_samples as MS, main as MAIN
    deg = MS.load_degrade(opts["degrade"]); mav0 = MS.seq_dir(name, opts["euroc"], opts["tumvi"]); txt, st = MS.degraded_csv(mav0, order, seed, deg)
    kd = k_loso(name) if opts["k_drag"] == "loso" else (MAIN.K_DRAG if opts["k_drag"] == "default" else float(opts["k_drag"]))
    gtf = os.path.join(mav0, "mocap0" if name.startswith("room") else "state_groundtruth_estimate0", "data.csv")
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "input.zip"); est = os.path.join(td, "estimated_trajectory.txt"); MS.write_zip(inp, txt)
        t0 = time.perf_counter(); _path, info = MAIN.run(inp, est, k_drag=kd); sec = time.perf_counter() - t0
        r = score_files(est, inp, gtf)
    return dict(name=name, order=order, seed=seed, k_drag=kd, seconds=sec, rotor=bool(info.get("rotor")), guard=bool(info.get("guard")), mode=info.get("mode"),
                fallback=info.get("fallback", ""), **r)
def protocol(a):
    sys.path.insert(0, os.path.join(REL, "tools")); import make_samples as MS
    names = [s for s in a.sequences.split(",") if s] or [n for n in MS.EUROC + MS.TUMVI if os.path.exists(os.path.join(MS.seq_dir(n, a.euroc, a.tumvi), "imu0", "data.csv"))]
    opts = dict(degrade=os.path.abspath(a.degrade), euroc=os.path.abspath(a.euroc) if a.euroc else "", tumvi=os.path.abspath(a.tumvi) if a.tumvi else "", k_drag=a.k_drag)
    jobs = [(n, o, int(sd), opts) for sd in a.seeds.split(",") for o in a.orders.split(",") for n in names]
    if a.procs > 1:
        from multiprocessing import Pool
        with Pool(a.procs) as pool: res = pool.map(_job, jobs, chunksize=1)
    else: res = [_job(j) for j in jobs]
    keys = ("ate", "wp", "b_abs", "b_anc", "drop", "rpe_t", "ori")
    print("%d missions; main.py vs motionless (metres; ori and rpe_r in degrees)" % len(res))
    for gname, sel in (("EuRoC", lambda r: not r["name"].startswith("room")), ("TUM-VI", lambda r: r["name"].startswith("room")), ("all", lambda r: True)):
        for o in sorted({r["order"] for r in res}) + ["both"]:
            rr = [r for r in res if sel(r) and (o == "both" or r["order"] == o) and "score" in r]
            if not rr: continue
            print("  %-6s order %-4s n=%-4d " % (gname, o, len(rr)) + "  ".join("%s %.4f/%.4f" % (k, np.nanmean([r["score"][k] for r in rr]), np.nanmean([r["motionless"][k] for r in rr])) for k in keys))
    bad = [r for r in res if "score" not in r or r["fallback"]]
    print("contract failures or fallbacks: %d; platform rotor on %d EuRoC / %d TUM-VI missions; guard fired %d" % (len(bad), sum(r["rotor"] for r in res if not r["name"].startswith("room")),
          sum(r["rotor"] for r in res if r["name"].startswith("room")), sum(r["guard"] for r in res)))
    if a.json: json.dump(res, open(a.json, "w"), indent=1)
    return 0 if not bad else 1
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score main.py outputs (SE(3) Umeyama ATE and proxies).")
    ap.add_argument("files", nargs="*", help="EST INPUT GROUND_TRUTH"); ap.add_argument("--json"); ap.add_argument("--gt-tum")
    ap.add_argument("--protocol", action="store_true"); ap.add_argument("--degrade"); ap.add_argument("--euroc", default=""); ap.add_argument("--tumvi", default="")
    ap.add_argument("--orders", default="A,B"); ap.add_argument("--seeds", default="42"); ap.add_argument("--sequences", default="")
    ap.add_argument("--k-drag", default="loso"); ap.add_argument("--procs", type=int, default=1)
    a = ap.parse_args()
    if a.protocol:
        if not a.degrade: sys.exit("--protocol needs --degrade path/to/degrade_imu.py")
        sys.exit(protocol(a))
    if len(a.files) != 3: ap.error("give EST INPUT GROUND_TRUTH, or --protocol")
    r = score_files(*a.files, gt_tum=a.gt_tum)
    if "error" in r: print(r["error"], r["contract"]); sys.exit(1)
    s, s0 = r["score"], r["motionless"]
    print("rows %d (with ground truth %d, blind %d); main.py | motionless" % (r["contract"]["rows_in"], r["rows_with_gt"], r["blind_rows"]))
    for k in KEYS: print("  %-6s %9.4f | %9.4f" % (k, s[k], s0[k]))
    if a.json: json.dump(r, open(a.json, "w"), indent=1)
