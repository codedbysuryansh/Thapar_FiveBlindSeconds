#!/usr/bin/env python3
"""E50 - FEASIBILITY PROBE for the Python-only rule (Track 3 FAQ, 21 Nov): can the v5 computation fit <= 8 MB of Python heap
(tracemalloc, the FAQ's 'algorithm / heap level') and <= 20 ms per IMU-second on one thread, with numpy only?
NOT the estimator: it runs the cost-dominating kernels the way a numpy implementation would, on a zipped official-format input:
  parse (zip -> int64 timestamps + float32 channels, chunked) | timeline + validity (vectorised) | gyro integration
  (log-depth quaternion prefix scan in chunks) | block averaging to the smoother rate | gravity smoother (3 columns, Python loop
  at the low rate) | 17-state Kalman filter + checkpointed RTS at the low rate (numpy per step) | output writing (exact timestamps).
Usage: python e50_numpy_feasibility.py <evaluation.zip> <out.txt> [dec ...]"""
import sys, os, time, tracemalloc, zipfile, io
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(_k, "1")
tracemalloc.start(); T0 = time.perf_counter()
import numpy as np  # noqa: E402
T_IMPORT = time.perf_counter() - T0; MEM_IMPORT = tracemalloc.get_traced_memory()
def qmul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]; bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)
def parse(path):
    zf = zipfile.ZipFile(path); name = [n for n in zf.namelist() if n.endswith("imu0/data.csv")][0]; ts = []; X = []
    with zf.open(name) as fh:
        txt = io.TextIOWrapper(fh, encoding="utf-8")
        while True:
            lines = txt.readlines(1 << 18)
            if not lines: break
            lines = [l for l in lines if l.strip() and l[0] != "#"]
            if not lines: continue
            ts.append(np.array([int(l.split(",", 1)[0]) for l in lines], np.int64)); X.append(np.loadtxt(lines, delimiter=",", usecols=(1, 2, 3, 4, 5, 6), dtype=np.float32, ndmin=2))
    return np.concatenate(ts), np.concatenate(X)
def timeline(ts, X):
    d = np.diff(ts); med = np.median(d[d > 0]); st = np.maximum(np.rint(d / med).astype(np.int64), 0); rows = np.concatenate([[0], np.cumsum(st)]).astype(np.int32); N = int(rows[-1]) + 1
    U = np.zeros((N, 6), np.float32); U[rows] = X; valid = np.zeros(N, bool); valid[rows] = True; valid &= ~(U == 0).all(1); rep = np.zeros(N, bool); rep[1:] = (U[1:] == U[:-1]).all(1); valid &= ~rep
    return rows, U, valid, float(med) * 1e-9
def integrate(w, dt, chunk=4096):
    N = len(w); q = np.empty((N, 4), np.float32); q0 = np.array([0, 0, 0, 1.0])
    for s in range(0, N, chunk):
        wc = w[s:s + chunk].astype(np.float64) * dt; th = np.linalg.norm(wc, axis=1, keepdims=True); k = np.where(th > 1e-12, np.sin(th / 2) / np.maximum(th, 1e-12), 0.5); o = np.concatenate([wc * k, np.cos(th / 2)], 1); st = 1
        while st < len(o): o[st:] = qmul(o[:-st], o[st:]); st *= 2
        o = qmul(np.broadcast_to(q0, o.shape), o); o /= np.linalg.norm(o, axis=1, keepdims=True); q[s:s + chunk] = o; q0 = o[-1]
    return q
def rotate_block_mean(q, f, valid, d, chunk=8192):
    K = len(q) // d; out = np.zeros((K, 3)); cnt = np.zeros(K)
    for s in range(0, K * d, chunk):
        e = min(K * d, s + chunk); u = q[s:e, :3].astype(np.float64); w = q[s:e, 3:4].astype(np.float64); v = f[s:e].astype(np.float64); t = 2 * np.cross(u, v); r = v + w * t + np.cross(u, t)
        m = valid[s:e, None]; kb = (np.arange(s, e) // d); np.add.at(out, kb, r * m); np.add.at(cnt, kb, m[:, 0])
    return out / np.maximum(cnt, 1)[:, None], cnt > 0
def irw_smooth(x, ok, dt, tau=3.0):
    q = dt / tau ** 4; q11, q12, q22 = q * dt ** 3 / 3, q * dt ** 2 / 2, q * dt; K = len(x); m0 = x[0].copy(); m1 = np.zeros(3); p00, p01, p11 = 1e2, 0.0, 1e2; XF = np.empty((K, 2, 3)); PF = np.empty((K, 3)); PP = np.empty((K, 3))
    for i in range(K):
        if i: m0 = m0 + dt * m1; p00, p01, p11 = p00 + 2 * dt * p01 + dt * dt * p11 + q11, p01 + dt * p11 + q12, p11 + q22
        PP[i] = (p00, p01, p11)
        if ok[i]: s = p00 + 1.0; k0, k1 = p00 / s, p01 / s; r = x[i] - m0; m0 = m0 + k0 * r; m1 = m1 + k1 * r; p00, p01, p11 = p00 - k0 * p00, p01 - k0 * p01, p11 - k1 * p01
        XF[i, 0] = m0; XF[i, 1] = m1; PF[i] = (p00, p01, p11)
    out = np.empty((K, 3)); s0, s1 = XF[-1, 0].copy(), XF[-1, 1].copy(); out[-1] = s0
    for i in range(K - 2, -1, -1):
        f00, f01, f11 = PF[i]; b00, b01, b10, b11 = f00 + dt * f01, f01, f01 + dt * f11, f11; d00, d01, d11 = PP[i + 1]; det = d00 * d11 - d01 * d01
        c00 = (b00 * d11 - b01 * d01) / det; c01 = (-b00 * d01 + b01 * d00) / det; c10 = (b10 * d11 - b11 * d01) / det; c11 = (-b10 * d01 + b11 * d00) / det
        e0 = s0 - (XF[i, 0] + dt * XF[i, 1]); e1 = s1 - XF[i, 1]; s0 = XF[i, 0] + c00 * e0 + c01 * e1; s1 = XF[i, 1] + c10 * e0 + c11 * e1; out[i] = s0
    return out
def smoother(A, Q, y, ok, P0, n=17, ckpt=64):
    K = len(y); H4 = np.array([0, 0, 1.0, 1.0]); Hs = np.zeros((5, n)); Hs[0, 0:4] = H4; Hs[1, 4:8] = H4; Hs[2, 8:12] = H4; Hs[3, 1] = -0.18; Hs[3, 12] = 1; Hs[4, 5] = -0.18; Hs[4, 13] = 1; Rr = np.array([1.0, 1.0, 1.0, 2.0, 2.0]); cps = {}; At = A.T.copy()
    def step(x, P, k):
        xp = A @ x; Pp = A @ P @ At + Q
        if ok[k]:
            yk = np.concatenate([y[k], y[k, :2] * 0.1]); PHt = Pp @ Hs.T; S = Hs @ PHt; S[np.diag_indices(5)] += Rr; Kg = np.linalg.solve(S, PHt.T).T; return xp, Pp, xp + Kg @ (yk - Hs @ xp), Pp - Kg @ PHt.T
        return xp, Pp, xp, Pp
    x = np.zeros(n); P = P0.copy()
    for k in range(K):
        if k % ckpt == 0: cps[k] = (x.copy(), P.copy())
        _, _, x, P = step(x, P, k)
    out = np.empty((K, 3)); xs = x; out[-1] = xs[[0, 4, 8]]; nxt = None
    for b0 in sorted(cps, reverse=True):
        x, P = cps[b0]; b1 = min(K, b0 + ckpt); XF, PF, XP, PPl = [], [], [], []
        for k in range(b0, b1): xp, Pp, x, P = step(x, P, k); XF.append(x); PF.append(P); XP.append(xp); PPl.append(Pp)
        for j in range(b1 - b0 - 1, -1, -1):
            k = b0 + j
            if k == K - 1: continue
            xpn, Ppn = nxt if j == b1 - b0 - 1 else (XP[j + 1], PPl[j + 1])
            C = np.linalg.solve(Ppn, A @ PF[j]).T; xs = XF[j] + C @ (xs - xpn); out[k] = xs[[0, 4, 8]]
        nxt = (XP[0], PPl[0])
    return out
def write(path, ts, P, Q, chunk=8192):
    sec, ns = np.divmod(ts, 10 ** 9)
    with open(path, "w") as fh:
        fh.write("# timestamp tx ty tz qx qy qz qw\n")
        for s in range(0, len(ts), chunk):
            fh.write("".join("%d.%09d %.6f %.6f %.6f %.7f %.7f %.7f %.7f\n" % (a, b, *p, *q) for a, b, p, q in zip(sec[s:s + chunk].tolist(), ns[s:s + chunk].tolist(), P[s:s + chunk].tolist(), Q[s:s + chunk].tolist())))
if __name__ == "__main__":
    zpath, outp = sys.argv[1], sys.argv[2]; decs = [int(x) for x in sys.argv[3:]] or [8]; T = {}; base = tracemalloc.get_traced_memory()[0]; tracemalloc.reset_peak()
    t0 = time.perf_counter(); ts, X = parse(zpath); T["parse"] = time.perf_counter() - t0
    t0 = time.perf_counter(); rows, U, valid, dt = timeline(ts, X); T["timeline"] = time.perf_counter() - t0; N = len(U); dur = N * dt
    t0 = time.perf_counter(); w = U[:, :3].copy(); w[~valid] = 0; q = integrate(w, dt); del w; T["gyro integration (1 pass)"] = time.perf_counter() - t0
    for d in decs:
        t0 = time.perf_counter(); fb, ok = rotate_block_mean(q, U[:, 3:], valid, d); T["rotate + block mean (dec %d)" % d] = time.perf_counter() - t0
        t0 = time.perf_counter(); g = irw_smooth(fb, ok, dt * d); T["gravity smoother 1 pass (dec %d)" % d] = time.perf_counter() - t0
        F = np.zeros((17, 17)); h = dt * d
        for j in range(3): F[4 * j:4 * j + 3, 4 * j + 1:4 * j + 4] += np.eye(3) * h
        A = np.eye(17) + F; Qd = np.eye(17) * 1e-4; P0 = np.eye(17)
        t0 = time.perf_counter(); pos = smoother(A, Qd, (fb - g), ok, P0); T["17-state KF + checkpointed RTS (dec %d)" % d] = time.perf_counter() - t0
    t0 = time.perf_counter(); Pf = np.repeat(pos, decs[-1], 0); Pf = np.vstack([Pf, np.repeat(Pf[-1:], max(0, N - len(Pf)), 0)])[:N][rows]; write(outp, ts, Pf, q[rows]); T["write output"] = time.perf_counter() - t0
    cur, peak = tracemalloc.get_traced_memory()
    print("input rows %d, timeline %d, %.1f s of IMU; numpy %s; python %s; 'import numpy': %.3f s, %.2f MB traced" % (len(ts), N, dur, np.__version__, sys.version.split()[0], T_IMPORT, MEM_IMPORT[1] / 1e6))
    for k, v in T.items(): print("  %-44s %7.3f s  %6.2f ms per IMU-second" % (k, v, 1e3 * v / dur))
    print("  peak traced heap during the pipeline: %.2f MB above the post-import baseline; %.2f MB counting the numpy import" % ((peak - base) / 1e6, peak / 1e6))
