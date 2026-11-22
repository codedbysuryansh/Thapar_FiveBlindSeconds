#!/usr/bin/env python3
"""E62 - kernel benchmark for the Phase 4 numpy design (feasibility only; NOT the estimator). Synthetic inputs with the size and
gap structure of the longest public sequence (MH_01: 184 s at 200 Hz, ~6 % single drops + bursts). Measures, single thread:
  K1  gravity smoother: IRW, shared 2x2 covariance recursion (data-independent) + 9 columns (3 specific force + 6 regressor), at 25 Hz,
      (a) numpy per step, (b) pure-Python floats
  K2  hand-held smoother: 4-state Matern-5/2 + OU bias per axis, ONE covariance recursion shared by the 3 axes, 100 Hz, full RTS
  K3  rotor smoother: 17 states, joint update of 3 accel + 2 drag + up to 3 ZUPT rows, checkpointed RTS, at 50 Hz and 25 Hz
  K4  output formatting of 35,629 rows with exact '%d.%09d' timestamps (6 / 4 decimals)
Usage: python e62_kernel_bench.py"""
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ.setdefault(_k, "1")
import sys, time, tracemalloc, platform
import numpy as np
DUR = 184.1; FS = 200.0; rng = np.random.default_rng(0)

def gaps_mask(n_steps, p_drop=0.06, p_burst=0.0015):
    ok = rng.random(n_steps) > p_drop
    for i in np.where(rng.random(n_steps) < p_burst)[0]: ok[i:i + rng.integers(5, 50)] = False
    return ok

# ------------------------------------------------------------------ K1 gravity smoother (IRW), 25 Hz, shared covariance
def k1_numpy(y, ok, dt, tau=3.0, r=1.0):
    q = 1.0 / tau ** 4; K = len(y); m = y[0].copy() * 0; m = np.vstack([y[0], np.zeros(y.shape[1])]); P = np.array([[1e2, 0], [0, 1e2]]); A = np.array([[1, dt], [0, 1.0]])
    Qd = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]]); XF = np.empty((K, 2, y.shape[1])); PF = np.empty((K, 2, 2)); PP = np.empty((K, 2, 2))
    for k in range(K):
        if k: m = A @ m; P = A @ P @ A.T + Qd
        PP[k] = P
        if ok[k]: s = P[0, 0] + r; g = P[:, 0] / s; m = m + np.outer(g, y[k] - m[0]); P = P - np.outer(g, P[0])
        XF[k] = m; PF[k] = P
    out = np.empty_like(y); s_ = XF[-1].copy(); out[-1] = s_[0]
    for k in range(K - 2, -1, -1):
        C = PF[k] @ A.T @ np.linalg.inv(PP[k + 1]); s_ = XF[k] + C @ (s_ - A @ XF[k]); out[k] = s_[0]
    return out

def k1_python(y, ok, dt, tau=3.0, r=1.0):
    q = 1.0 / tau ** 4; q11, q12, q22 = q * dt ** 3 / 3, q * dt ** 2 / 2, q * dt; K, C = y.shape; Y = y.tolist(); okl = ok.tolist()
    p00, p01, p11 = 1e2, 0.0, 1e2; g0s = [0.0] * K; g1s = [0.0] * K; pf = [None] * K; pp = [None] * K
    for k in range(K):   # data-independent covariance pass, once
        if k: p00, p01, p11 = p00 + 2 * dt * p01 + dt * dt * p11 + q11, p01 + dt * p11 + q12, p11 + q22
        pp[k] = (p00, p01, p11)
        if okl[k]: s = p00 + r; g0, g1 = p00 / s, p01 / s; p00, p01, p11 = p00 - g0 * p00, p01 - g0 * p01, p11 - g1 * p01
        else: g0 = g1 = 0.0
        g0s[k] = g0; g1s[k] = g1; pf[k] = (p00, p01, p11)
    cs = [None] * K
    for k in range(K - 1):   # smoother gains, once
        f00, f01, f11 = pf[k]; b00, b01, b10, b11 = f00 + dt * f01, f01, f01 + dt * f11, f11; d00, d01, d11 = pp[k + 1]; det = d00 * d11 - d01 * d01
        cs[k] = ((b00 * d11 - b01 * d01) / det, (-b00 * d01 + b01 * d00) / det, (b10 * d11 - b11 * d01) / det, (-b10 * d01 + b11 * d00) / det)
    out = []
    for c in range(C):   # per column: forward mean, backward mean
        m0, m1 = Y[0][c], 0.0; xf0 = [0.0] * K; xf1 = [0.0] * K
        for k in range(K):
            if k: m0 = m0 + dt * m1
            g0 = g0s[k]
            if g0: e = Y[k][c] - m0; m0 += g0 * e; m1 += g1s[k] * e
            xf0[k] = m0; xf1[k] = m1
        s0, s1 = xf0[-1], xf1[-1]; col = [0.0] * K; col[-1] = s0
        for k in range(K - 2, -1, -1):
            c00, c01, c10, c11 = cs[k]; a0, a1 = xf0[k], xf1[k]; e0 = s0 - (a0 + dt * a1); e1 = s1 - a1; s0 = a0 + c00 * e0 + c01 * e1; s1 = a1 + c10 * e0 + c11 * e1; col[k] = s0
        out.append(col)
    return np.array(out).T

# ------------------------------------------------------------------ K2 hand-held smoother, 100 Hz, shared covariance across axes
def model_matern(ell=1.0, sp=2.0, tau_b=20.0, sb=0.1, dt=0.01):
    lam = np.sqrt(5.0) / ell; F = np.zeros((4, 4)); F[0, 1] = 1; F[1, 2] = 1; F[2, :3] = [-lam ** 3, -3 * lam ** 2, -3 * lam]; F[3, 3] = -1 / tau_b
    Qc = np.zeros((4, 4)); Qc[2, 2] = 16 / 3 * sp ** 2 * lam ** 5; Qc[3, 3] = 2 * sb ** 2 / tau_b
    M = np.block([[-F, Qc], [np.zeros((4, 4)), F.T]]) * dt; E = expm(M); A = E[4:, 4:].T; Q = A @ E[:4, 4:]; return A, 0.5 * (Q + Q.T)
def expm(M):   # scaling and squaring with a 12-term Taylor series (numpy only)
    n = max(0, int(np.ceil(np.log2(max(np.abs(M).sum(1).max(), 1e-12)))) + 4); X = M / 2 ** n; E = np.eye(len(M)); T = np.eye(len(M))
    for i in range(1, 13): T = T @ X / i; E = E + T
    for _ in range(n): E = E @ E
    return E
def k2(y, ok, zupt, A, Q, P0, r=0.04, rz=4e-4):
    K = len(y); H = np.array([0, 0, 1.0, 1.0]); Hz = np.array([0, 1.0, 0, 0]); x = np.zeros((4, 3)); P = P0.copy(); At = A.T
    ck = 128; cps = {}
    def fwd(x, P, k):
        xp = A @ x; Pp = A @ P @ At + Q; x, P = xp, Pp
        if ok[k]: PH = P @ H; s = H @ PH + r; g = PH / s; x = x + np.outer(g, y[k] - H @ x); P = P - np.outer(g, PH)
        if zupt[k]: PH = P @ Hz; s = Hz @ PH + rz; g = PH / s; x = x - np.outer(g, Hz @ x); P = P - np.outer(g, PH)
        return xp, Pp, x, P
    for k in range(K):
        if k % ck == 0: cps[k] = (x.copy(), P.copy())
        _, _, x, P = fwd(x, P, k)
    out = np.empty((K, 3)); xs = x; out[-1] = xs[0]; nxt = None
    for b0 in sorted(cps, reverse=True):
        x, P = cps[b0]; b1 = min(K, b0 + ck); seg = []
        for k in range(b0, b1): xp, Pp, x, P = fwd(x, P, k); seg.append((xp, Pp, x, P))
        for j in range(b1 - b0 - 1, -1, -1):
            k = b0 + j
            if k == K - 1: continue
            xpn, Ppn = nxt if j == b1 - b0 - 1 else seg[j + 1][:2]; C = np.linalg.solve(Ppn, A @ seg[j][3]).T; xs = seg[j][2] + C @ (xs - xpn); out[k] = xs[0]
        nxt = seg[0][:2]
    return out

# ------------------------------------------------------------------ K3 rotor smoother, 17 states, joint update, checkpointed RTS
def k3(y, ok, fl, zupt, ew, A, Q, P0, ck=96):
    K = len(y); n = 17; H = np.zeros((8, n)); R = np.array([0.04] * 3 + [1.96] * 2 + [4e-4] * 3)
    for j in range(3): H[j, 4 * j + 2] = 1; H[j, 4 * j + 3] = 1; H[5 + j, 4 * j + 1] = 1
    H[3, 12] = 1; H[4, 13] = 1; At = A.T; idx_acc = np.arange(3); cps = {}
    def fwd(x, P, k):
        xp = A @ x; Pp = A @ P @ At + Q; x, P = xp, Pp
        rows = []
        if ok[k]: rows += [0, 1, 2]
        if fl[k]:
            for i in range(2): H[3 + i, [1, 5, 9]] = -0.187 * ew[k, i]
            rows += [3, 4]
        if zupt[k]: rows += [5, 6, 7]
        if rows:
            Hk = H[rows]; PH = P @ Hk.T; S = Hk @ PH; S[np.diag_indices(len(rows))] += R[rows]; G = np.linalg.solve(S, PH.T).T; yk = np.zeros(len(rows)); na = 3 if ok[k] else 0
            if na: yk[:3] = y[k]
            x = x + G @ (yk - Hk @ x); P = P - G @ PH.T
        return xp, Pp, x, P
    x = np.zeros(n); P = P0.copy()
    for k in range(K):
        if k % ck == 0: cps[k] = (x.copy(), P.copy())
        _, _, x, P = fwd(x, P, k)
    out = np.empty((K, 6)); xs = x; out[-1] = xs[[0, 4, 8, 1, 5, 9]]; nxt = None
    for b0 in sorted(cps, reverse=True):
        x, P = cps[b0]; b1 = min(K, b0 + ck); seg = []
        for k in range(b0, b1): xp, Pp, x, P = fwd(x, P, k); seg.append((xp, Pp, x, P))
        for j in range(b1 - b0 - 1, -1, -1):
            k = b0 + j
            if k == K - 1: continue
            xpn, Ppn = nxt if j == b1 - b0 - 1 else seg[j + 1][:2]; C = np.linalg.solve(Ppn, A @ seg[j][3]).T; xs = seg[j][2] + C @ (xs - xpn); out[k] = xs[[0, 4, 8, 1, 5, 9]]
        nxt = seg[0][:2]
    return out

# ------------------------------------------------------------------ K4 output formatting
def k4(ts, P, Q, dec_p=6, dec_q=7, chunk=4096):
    sec, ns = np.divmod(ts, 10 ** 9); fmt = "%%d.%%09d %%.%df %%.%df %%.%df %%.%df %%.%df %%.%df %%.%df\n" % ((dec_p,) * 3 + (dec_q,) * 4); n = 0
    for s in range(0, len(ts), chunk):
        txt = "".join([fmt % (a, b, *p, *q) for a, b, p, q in zip(sec[s:s + chunk].tolist(), ns[s:s + chunk].tolist(), P[s:s + chunk].tolist(), Q[s:s + chunk].tolist())]); n += len(txt)
    return n

def timed(label, fn, *a):
    tracemalloc.start(); t0 = time.perf_counter(); out = fn(*a); dt_ = time.perf_counter() - t0; peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    print("  %-62s %7.3f s  %6.2f ms per IMU-second  peak %.2f MB" % (label, dt_, 1e3 * dt_ / DUR, peak / 1e6)); return out

if __name__ == "__main__":
    print("E62 kernel benchmark: %s, Python %s, numpy %s, %s" % (platform.processor() or platform.machine(), sys.version.split()[0], np.__version__, platform.platform()))
    for hz, lab in ((25, "K1 gravity smoother 25 Hz, 9 columns"),):
        K = int(DUR * hz); y = rng.normal(size=(K, 9)); ok = gaps_mask(K, 0.01, 0.0)
        timed(lab + " - numpy per step", k1_numpy, y, ok, 1 / hz); timed(lab + " - pure Python floats", k1_python, y, ok, 1 / hz)
    A, Q = model_matern(dt=0.01); P0 = np.eye(4); K = int(DUR * 100); y = rng.normal(size=(K, 3)); ok = gaps_mask(K, 0.03); zu = np.zeros(K, bool); zu[:300] = True
    timed("K2 hand-held smoother 100 Hz (4 states x 3 axes, shared P)", k2, y, ok, zu, A, Q, P0)
    A8, Q8 = model_matern(dt=0.04); K8 = int(DUR * 25); timed("K2 hand-held smoother 25 Hz (guard-only run)", k2, y[:K8], ok[:K8], zu[:K8], A8, Q8, P0)
    for hz in (50, 25):
        K = int(DUR * hz); n = 17; A17 = np.eye(n) + np.diag(np.ones(n - 1) * 0.01, 1) * 0.5; Q17 = np.eye(n) * 1e-4; ew = rng.normal(size=(K, 2, 3)); fl = np.ones(K, bool); zu = np.zeros(K, bool); zu[:hz * 3] = True
        timed("K3 rotor smoother %d Hz (17 states, joint update, ckpt RTS)" % hz, k3, rng.normal(size=(K, 3)), gaps_mask(K, 0.02), fl, zu, ew, A17, Q17, np.eye(n))
    N = 35629; ts = 1403636579758555392 + np.cumsum(rng.integers(4_900_000, 5_100_000, N)).astype(np.int64); Pp = rng.normal(size=(N, 3)); Qq = rng.normal(size=(N, 4)); Qq /= np.linalg.norm(Qq, axis=1, keepdims=True)
    timed("K4 output formatting, 35,629 rows (6/7 decimals)", k4, ts, Pp, Qq, 6, 7); timed("K4 output formatting, 35,629 rows (4/6 decimals)", k4, ts, Pp, Qq, 4, 6)
