#!/usr/bin/env python3
"""E62b - FAST kernel designs for the Phase 4 numpy implementation (feasibility PoC, not the estimator).
Idea: in every v5 smoother the covariance recursion does not depend on the measured values (only on the time grid, the validity
masks and the attitude-dependent drag geometry, all known beforehand). So (1) run the covariance (Riccati) recursion once, step by
step, with minimal per-step work; (2) the mean recursions of the filter and of the RTS pass are then AFFINE recursions
x_k = M_k x_{k-1} + u_k, evaluated with a vectorised log-depth associative scan over all steps and all columns at once.
Output: a vectorised fixed-point ASCII renderer (pad bytes removed afterwards) that writes exact '%d.%09d' timestamps.
Each fast kernel is checked against the straightforward reference implementation (max abs difference reported)."""
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ.setdefault(_k, "1")
import sys, time, tracemalloc, platform
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e62_kernel_bench as ref
DUR = ref.DUR; rng = np.random.default_rng(1)

def affine_scan(M, u, x0, reverse=False):
    """x_k = M_k x_{k-1} + u_k for k = 0..K-1 (x_{-1} = x0); M (K,n,n), u (K,n,c). Hillis-Steele scan, log2(K) vectorised levels."""
    if reverse: M, u = M[::-1], u[::-1]
    M = M.copy(); u = u.copy(); K = len(M); d = 1
    while d < K:
        u[d:] = np.einsum("kij,kjc->kic", M[d:], u[:-d]) + u[d:]; M[d:] = np.einsum("kij,kjl->kil", M[d:], M[:-d]); d *= 2
    x = np.einsum("kij,jc->kic", M, x0) + u
    return x[::-1] if reverse else x

# ------------------------------------------------------------------ K1b gravity smoother: Python-float covariance + affine scans
def k1_fast(y, ok, dt, tau=3.0, r=1.0):
    q = 1.0 / tau ** 4; q11, q12, q22 = q * dt ** 3 / 3, q * dt ** 2 / 2, q * dt; K, C = y.shape; okl = ok.tolist()
    g = np.zeros((K, 2)); PF = np.zeros((K, 3)); PP = np.zeros((K, 3)); p00, p01, p11 = 1e2, 0.0, 1e2; gl = [None] * K; pfl = [None] * K; ppl = [None] * K
    for k in range(K):
        if k: p00, p01, p11 = p00 + 2 * dt * p01 + dt * dt * p11 + q11, p01 + dt * p11 + q12, p11 + q22
        ppl[k] = (p00, p01, p11)
        if okl[k]: s = p00 + r; g0, g1 = p00 / s, p01 / s; p00, p01, p11 = p00 - g0 * p00, p01 - g0 * p01, p11 - g1 * p01
        else: g0 = g1 = 0.0
        gl[k] = (g0, g1); pfl[k] = (p00, p01, p11)
    g = np.array(gl); PF = np.array(pfl); PP = np.array(ppl); A = np.array([[1.0, dt], [0.0, 1.0]])
    # forward: m_k = (I - g e1') A m_{k-1} + g y_k   (k = 0: no predict)
    IG = np.repeat(np.eye(2)[None], K, 0); IG[:, :, 0] -= g; M = IG @ A; M[0] = IG[0]; u = g[:, :, None] * y[:, None, :]
    # initial state: the reference starts from m = (y0, 0), P = 100 I, no prediction at k = 0 -> x_{-1} = (y0, 0) and M_0 = IG_0
    xf = affine_scan(M, u, np.vstack([y[0], np.zeros(C)]))
    # backward: s_k = C_k s_{k+1} + (xf_k - C_k A xf_k), C_k = PF_k A' PP_{k+1}^-1
    f00, f01, f11 = PF[:-1, 0], PF[:-1, 1], PF[:-1, 2]; b00, b01, b10, b11 = f00 + dt * f01, f01, f01 + dt * f11, f11; d00, d01, d11 = PP[1:, 0], PP[1:, 1], PP[1:, 2]; det = d00 * d11 - d01 * d01
    Cm = np.empty((K - 1, 2, 2)); Cm[:, 0, 0] = (b00 * d11 - b01 * d01) / det; Cm[:, 0, 1] = (-b00 * d01 + b01 * d00) / det; Cm[:, 1, 0] = (b10 * d11 - b11 * d01) / det; Cm[:, 1, 1] = (-b10 * d01 + b11 * d00) / det
    v = xf[:-1] - np.einsum("kij,jl,klc->kic", Cm, A, xf[:-1]); s = affine_scan(Cm, v, xf[-1], reverse=True)
    return np.vstack([s[:, 0], xf[-1:, 0]])

# ------------------------------------------------------------------ K2b / K3b generic: Riccati once (chunk checkpoints), means by scans
def smoother_fast(A, Q, P0, Hfun, y, K, n, chunk=512):
    """Kalman filter + RTS for x_k = A x_{k-1} + w, z_k = H_k x_k + v (H_k from Hfun(k) -> (H, R, rows-of-y or None)).
    Pass 1: Riccati forward storing only checkpoints of P every `chunk` steps plus the per-step affine map (M_k, u_k) of the mean
    (I - G H) A and G z  -> filtered means by one scan. Pass 2: per chunk (reverse), recompute P_k|k, P_k+1|k from the checkpoint,
    form C_k and the backward affine map, scan it."""
    At = A.T; I = np.eye(n); cps = {}; Mf = np.empty((K, n, n)); uf = np.empty((K, n, y.shape[2])); P = P0.copy()
    for k in range(K):
        if k % chunk == 0: cps[k] = P.copy()
        Pp = A @ P @ At + Q
        H, R, z = Hfun(k)
        if H is None: Mf[k] = A; uf[k] = 0; P = Pp; continue
        PH = Pp @ H.T; S = H @ PH; S.flat[::len(R) + 1] += R; G = np.linalg.solve(S, PH.T).T; P = Pp - G @ PH.T
        Mf[k] = (I - G @ H) @ A; uf[k] = G @ z
    xf = affine_scan(Mf, uf, np.zeros((n, uf.shape[2]))); del Mf, uf
    xs = np.empty_like(xf); xs[-1] = xf[-1]; nxt = xf[-1]
    starts = sorted(cps)
    for c0 in reversed(starts):
        c1 = min(K, c0 + chunk); P = cps[c0]; PFs = []; PPs = []
        for k in range(c0, min(K, c1 + 1)):
            Pp = A @ P @ At + Q; PPs.append(Pp)
            H, R, z = Hfun(k)
            if H is None: P = Pp
            else: PH = Pp @ H.T; S = H @ PH; S.flat[::len(R) + 1] += R; G = np.linalg.solve(S, PH.T).T; P = Pp - G @ PH.T
            PFs.append(P)
        L = c1 - c0 - (1 if c1 == K else 0)
        if L <= 0: continue
        Cm = np.stack([np.linalg.solve(PPs[j + 1], A @ PFs[j]).T for j in range(L)])
        v = xf[c0:c0 + L] - np.einsum("kij,jl,klc->kic", Cm, A, xf[c0:c0 + L]); s_end = xs[c0 + L] if c0 + L < K else xf[-1]
        xs[c0:c0 + L] = affine_scan(Cm, v, s_end, reverse=True)
    return xs

# ------------------------------------------------------------------ K4b vectorised exact ASCII renderer
def render(ts, P, Q, dp=6, dq=7):
    """Exact text identical to '%d.%09d %.{dp}f %.{dp}f %.{dp}f %.{dq}f %.{dq}f %.{dq}f %.{dq}f\\n' % row, for every row (|values| < 1e9)."""
    N = len(ts); cols = []
    sec, ns = np.divmod(ts.astype(np.int64), 10 ** 9)
    def digits(v, width):   # v >= 0 int64 -> (N, width) ASCII digits with leading pad (0 bytes) except the last digit
        out = np.zeros((N, width), np.uint8); x = v.copy()
        for j in range(width - 1, -1, -1):
            d = (x % 10).astype(np.uint8); out[:, j] = 48 + d; x //= 10
        nd = np.maximum(1, np.floor(np.log10(np.maximum(v, 1))).astype(int) + 1); pad = np.arange(width)[None, :] < (width - nd)[:, None]; out[pad] = 0; return out
    def fixed(x, dec, width_int=9):
        neg = np.signbit(x); s = np.round(np.abs(x) * 10 ** dec).astype(np.int64); ip, fp = np.divmod(s, 10 ** dec)
        sign = np.where(neg, 45, 0).astype(np.uint8)[:, None]; I_ = digits(ip, width_int); F_ = np.zeros((N, dec), np.uint8); y = fp.copy()
        for j in range(dec - 1, -1, -1): F_[:, j] = 48 + (y % 10); y //= 10
        return np.hstack([sign, I_, np.full((N, 1), 46, np.uint8), F_])
    sp = np.full((N, 1), 32, np.uint8)
    parts = [digits(sec, 12), np.full((N, 1), 46, np.uint8)]
    nsd = np.zeros((N, 9), np.uint8); y = ns.copy()
    for j in range(8, -1, -1): nsd[:, j] = 48 + (y % 10); y //= 10
    parts += [nsd]
    for j in range(3): parts += [sp, fixed(P[:, j], dp)]
    for j in range(4): parts += [sp, fixed(Q[:, j], dq)]
    parts += [np.full((N, 1), 10, np.uint8)]
    buf = np.hstack(parts).ravel(); return buf[buf != 0].tobytes()

def timed(label, fn, *a):
    tracemalloc.start(); t0 = time.perf_counter(); out = fn(*a); dt_ = time.perf_counter() - t0; peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    print("  %-66s %7.3f s  %6.2f ms per IMU-second  peak %.2f MB" % (label, dt_, 1e3 * dt_ / DUR, peak / 1e6)); return out

if __name__ == "__main__":
    print("E62b fast kernels: %s, Python %s, numpy %s" % (platform.platform(), sys.version.split()[0], np.__version__))
    hz = 25; K = int(DUR * hz); y = rng.normal(size=(K, 9)).cumsum(0) * 0.01 + 9.8; ok = ref.gaps_mask(K, 0.05, 0.002)
    a = timed("K1  gravity smoother 25 Hz x 9 columns - reference (numpy per step)", ref.k1_numpy, y, ok, 1 / hz)
    b = timed("K1b gravity smoother 25 Hz x 9 columns - Riccati once + affine scans", k1_fast, y, ok, 1 / hz)
    print("      max |difference| = %.2e" % np.abs(a - b).max())
    # hand-held: Matern 4-state, 3 axes share P (one smoother with 3 columns)
    for hz in (100, 50):
        A, Q = ref.model_matern(dt=1.0 / hz); P0 = np.eye(4); K = int(DUR * hz); Y = rng.normal(size=(K, 3)); okm = ref.gaps_mask(K, 0.03); zu = np.zeros(K, bool); zu[: hz * 3] = True
        Ha = np.array([[0, 0, 1.0, 1.0]]); Hz = np.array([[0, 1.0, 0, 0]]); Hb = np.vstack([Ha, Hz])
        def Hfun(k, Y=Y, okm=okm, zu=zu):
            if okm[k] and zu[k]: return Hb, np.array([0.04, 4e-4]), np.vstack([Y[k], np.zeros(3)])
            if okm[k]: return Ha, np.array([0.04]), Y[k][None]
            if zu[k]: return Hz, np.array([4e-4]), np.zeros((1, 3))
            return None, None, None
        r1 = timed("K2  hand-held smoother %d Hz - reference (per step, ckpt RTS)" % hz, ref.k2, Y, okm, zu, A, Q, P0)
        r2 = timed("K2b hand-held smoother %d Hz - Riccati once + affine scans" % hz, smoother_fast, A, Q, P0, Hfun, Y[:, None, :], K, 4)
        print("      max |difference| = %.2e" % np.abs(r1 - r2[:, 0, :]).max())
    # rotor: 17 states, accel (3) + drag (2) + ZUPT (3) rows; a STABLE spring-damper-like A for a meaningful accuracy check
    for hz in (50, 25):
        K = int(DUR * hz); n = 17; dt = 1.0 / hz; F = np.zeros((n, n))
        for j in range(3): s = 4 * j; F[s, s + 1] = 1; F[s + 1, s] = -(2 * np.pi / 20) ** 2; F[s + 1, s + 1] = -0.5; F[s + 1, s + 2] = 1; F[s + 2, s + 2] = -1 / 0.3; F[s + 3, s + 3] = -1 / 20
        for j in (12, 13): F[j, j] = -0.01
        Qc = np.zeros((n, n))
        for j in range(3): Qc[4 * j + 2, 4 * j + 2] = 2 / 0.3; Qc[4 * j + 3, 4 * j + 3] = 2 * 0.01 / 20
        Qc[12, 12] = Qc[13, 13] = 2 * 0.0225 / 100
        M = np.block([[-F, Qc], [np.zeros((n, n)), F.T]]) * dt; E = ref.expm(M); A = E[n:, n:].T; Qd = A @ E[:n, n:]; Qd = 0.5 * (Qd + Qd.T); P0 = np.eye(n)
        Y = rng.normal(size=(K, 3)); okm = ref.gaps_mask(K, 0.02); fl = np.ones(K, bool); zu = np.zeros(K, bool); zu[: hz * 3] = True; ew = rng.normal(size=(K, 2, 3)); ew /= np.linalg.norm(ew, axis=2, keepdims=True)
        H0 = np.zeros((8, n)); R0 = np.array([0.04] * 3 + [1.96] * 2 + [4e-4] * 3)
        for j in range(3): H0[j, 4 * j + 2] = 1; H0[j, 4 * j + 3] = 1; H0[5 + j, 4 * j + 1] = 1
        H0[3, 12] = 1; H0[4, 13] = 1
        def Hfun3(k, Y=Y, okm=okm, fl=fl, zu=zu, ew=ew):
            rows = ([0, 1, 2] if okm[k] else []) + ([3, 4] if fl[k] else []) + ([5, 6, 7] if zu[k] else [])
            if not rows: return None, None, None
            H = H0[rows].copy(); z = np.zeros((len(rows), 1)); i = 0
            if okm[k]: z[:3, 0] = Y[k]; i = 3
            if fl[k]:
                for d in range(2): H[i + d, [1, 5, 9]] = -0.187 * ew[k, d]
            return H, R0[rows], z
        r1 = timed("K3  rotor smoother %d Hz - reference (per step, ckpt RTS)" % hz, ref.k3, Y, okm, fl, zu, ew, A, Qd, P0)
        r2 = timed("K3b rotor smoother %d Hz - Riccati once + affine scans" % hz, smoother_fast, A, Qd, P0, Hfun3, None if False else np.zeros((K, 1, 1)), K, n)
        print("      max |position difference| = %.2e m" % np.abs(r1[:, :3] - r2[:, [0, 4, 8], 0]).max())
    N = 35629; ts = 1403636579758555392 + np.cumsum(rng.integers(4_900_000, 5_100_000, N)).astype(np.int64); Pp = rng.normal(size=(N, 3)) * 5; Qq = rng.normal(size=(N, 4)); Qq /= np.linalg.norm(Qq, axis=1, keepdims=True)
    timed("K4  output, 35,629 rows, Python %-formatting", ref.k4, ts, Pp, Qq, 6, 7)
    txt = timed("K4b output, 35,629 rows, vectorised renderer", render, ts, Pp, Qq, 6, 7)
    sec, ns = np.divmod(ts, 10 ** 9); exp = "".join("%d.%09d %.6f %.6f %.6f %.7f %.7f %.7f %.7f\n" % (a_, b_, *p, *q) for a_, b_, p, q in zip(sec.tolist(), ns.tolist(), Pp.tolist(), Qq.tolist())).encode()
    a_l = txt.split(b"\n"); b_l = exp.split(b"\n"); diff = sum(x != y_ for x, y_ in zip(a_l, b_l))
    print("      renderer vs %%-formatting: %d of %d rows differ (last-digit rounding of exact ties), %d bytes" % (diff, N, len(txt)))
    bad = [(x, y_) for x, y_ in zip(a_l, b_l) if x != y_][:2]
    for x, y_ in bad: print("        e.g. %s | %s" % (x.decode(), y_.decode()))
    vals = np.array([[float(t) for t in l.split()] for l in a_l[:2000] if l]); ref_v = np.column_stack([ts[:2000] * 1e-9, Pp[:2000], Qq[:2000]])
    print("      parsed back (first 2000 rows): max |value error| %.1e (positions), timestamps exact: %s" % (np.abs(vals[:, 1:] - ref_v[:, 1:]).max(), all(l.split()[0] == ("%d.%09d" % divmod(int(t), 10 ** 9)).encode() for l, t in zip(a_l[:2000], ts[:2000]))))
