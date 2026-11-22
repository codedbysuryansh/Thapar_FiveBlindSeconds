#!/usr/bin/env python3
"""E64 - TIGHT NumPy kernels for the two position smoothers (the largest runtime items of the Phase 4 build), measured against the
E62 reference kernels (same models, same inputs; exactness reported). Feasibility PoC, not the estimator.
  rotor     : 17 states, accel (3) + drag (2) rows every step in flight, ZUPT (3) rows only at rest -> joint update with a 5- or 8-row
              fixed-shape H (inactive rows zeroed, R = 1, z = 0: exact), RTS backward with a VECTOR solve D = Pp^-1 (xs - xp),
              xs = xf + Pf A' D (no 17 x 17 gain matrix), sqrt(K) checkpoints, references instead of copies.
  hand-held : 4 states, one covariance shared by the 3 axes; scalar update when only the accelerometer row is active (most steps),
              2-row joint update only inside ZUPT windows; same RTS form.
Usage: python e64_tight_kernels.py  (single thread; prints ms per IMU-second for a 184.1 s input and the max difference)"""
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ.setdefault(_k, "1")
import sys, time, tracemalloc, platform
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e62_kernel_bench as ref
DUR = ref.DUR; rng = np.random.default_rng(2)

def rts_chunks(K, ck, cps, fwd, out, take, At, n):
    """Shared backward machinery: recompute each chunk forward from its checkpoint, then RTS backward through it."""
    xs = None; nxt = None
    for c0 in sorted(cps, reverse=True):
        x, P = cps[c0]; c1 = min(K, c0 + ck); XP = []; PP = []; XF = []; PF = []
        for k in range(c0, c1):
            xp, Pp, x, P = fwd(x, P, k); XP.append(xp); PP.append(Pp); XF.append(x); PF.append(P)
        for j in range(c1 - c0 - 1, -1, -1):
            k = c0 + j
            if k == K - 1: xs = XF[j]
            else:
                xpn, Ppn = nxt if j == c1 - c0 - 1 else (XP[j + 1], PP[j + 1])
                xs = XF[j] + PF[j] @ (At @ np.linalg.solve(Ppn, xs - xpn))
            out[k] = take(xs)
        nxt = (XP[0], PP[0])
    return out

def k3_tight(y, ok, fl, zupt, ew, A, Q, P0, kd=0.187, ck=None, sym_every=1):
    K = len(y); n = A.shape[0]; At = A.T; ck = ck or max(16, int(np.sqrt(K)))
    act = np.zeros((K, 8)); act[:, :3] = ok[:, None]; act[:, 3:5] = fl[:, None]; act[:, 5:] = zupt[:, None]
    Z = np.zeros((K, 8)); Z[:, :3] = y * ok[:, None]; R0 = np.array([0.04] * 3 + [1.96] * 2 + [4e-4] * 3); RK = np.where(act > 0, R0, 1.0)
    H = np.zeros((8, n))
    for j in range(3): H[j, 4 * j + 2] = 1; H[j, 4 * j + 3] = 1; H[5 + j, 4 * j + 1] = 1
    H[3, 12] = 1; H[4, 13] = 1; Hd = -kd * ew; rd, cd = np.ix_([3, 4], [1, 5, 9]); m_of = np.where(zupt, 8, 5); anyact = act.any(1)
    def fwd(x, P, k):
        xp = A @ x; Pp = A @ P @ At; Pp += Q
        if not anyact[k]: return xp, Pp, xp, Pp
        m = m_of[k]; H[rd, cd] = Hd[k]; Hk = H[:m] * act[k, :m, None]
        PH = Pp @ Hk.T; S = Hk @ PH; S.flat[::m + 1] += RK[k, :m]; Gt = np.linalg.solve(S, PH.T); P = Pp - PH @ Gt
        return xp, Pp, xp + (Z[k, :m] - Hk @ xp) @ Gt, (0.5 * (P + P.T) if k % sym_every == 0 else P)
    x = np.zeros(n); P = P0.copy(); cps = {}
    for k in range(K):
        if k % ck == 0: cps[k] = (x, P)
        _, _, x, P = fwd(x, P, k)
    idx = np.array([0, 4, 8, 1, 5, 9])
    return rts_chunks(K, ck, cps, fwd, np.empty((K, 6)), lambda xs: xs[idx], At, n)

def k2_tight(Y, okm, zu, A, Q, P0, ck=None):
    K = len(Y); At = A.T; ck = ck or max(16, int(np.sqrt(K))); h = np.array([0, 0, 1.0, 1.0]); hz = np.array([0, 1.0, 0, 0]); Hb = np.vstack([h, hz])
    ra, rz = 0.04, 4e-4; mode = okm.astype(int) + 2 * zu.astype(int); zeros3 = np.zeros(3)
    def fwd(x, P, k):
        xp = A @ x; Pp = A @ P @ At; Pp += Q; md = mode[k]
        if md == 0: return xp, Pp, xp, Pp
        if md == 1:
            PH = Pp @ h; g = PH / (h @ PH + ra); return xp, Pp, xp + np.outer(g, Y[k] - h @ xp), Pp - np.outer(g, PH)
        if md == 2:
            PH = Pp @ hz; g = PH / (hz @ PH + rz); return xp, Pp, xp - np.outer(g, hz @ xp), Pp - np.outer(g, PH)
        PH = Pp @ Hb.T; S = Hb @ PH; S[0, 0] += ra; S[1, 1] += rz; Gt = np.linalg.solve(S, PH.T); P = Pp - PH @ Gt
        return xp, Pp, xp + Gt.T @ (np.vstack([Y[k], zeros3]) - Hb @ xp), 0.5 * (P + P.T)   # joint form: symmetrise (else the skew part grows ~x1.4 per step)
    x = np.zeros((4, 3)); P = P0.copy(); cps = {}
    for k in range(K):
        if k % ck == 0: cps[k] = (x, P)
        _, _, x, P = fwd(x, P, k)
    return rts_chunks(K, ck, cps, fwd, np.empty((K, 3)), lambda xs: xs[0], At, 4)

def timed(label, fn, *a):
    tracemalloc.start(); t0 = time.perf_counter(); out = fn(*a); dt_ = time.perf_counter() - t0; peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    print("  %-64s %7.3f s  %6.2f ms per IMU-second  peak %.2f MB" % (label, dt_, 1e3 * dt_ / DUR, peak / 1e6), flush=True); return out

def rotor_model(hz, n=17):
    dt = 1.0 / hz; F = np.zeros((n, n))
    for j in range(3): s = 4 * j; F[s, s + 1] = 1; F[s + 1, s] = -(2 * np.pi / 20) ** 2; F[s + 1, s + 1] = -0.5; F[s + 1, s + 2] = 1; F[s + 2, s + 2] = -1 / 0.3; F[s + 3, s + 3] = -1 / 20
    for j in (12, 13): F[j, j] = -0.01
    Qc = np.zeros((n, n))
    for j in range(3): Qc[4 * j + 2, 4 * j + 2] = 2 / 0.3; Qc[4 * j + 3, 4 * j + 3] = 2 * 0.01 / 20
    Qc[12, 12] = Qc[13, 13] = 2 * 0.0225 / 100
    M = np.block([[-F, Qc], [np.zeros((n, n)), F.T]]) * dt; E = ref.expm(M); A = E[n:, n:].T; Qd = A @ E[:n, n:]; return A, 0.5 * (Qd + Qd.T), np.eye(n)

if __name__ == "__main__":
    print("E64 tight smoother kernels: %s, Python %s, numpy %s" % (platform.platform(), sys.version.split()[0], np.__version__), flush=True)
    for hz in (100, 50):
        A, Q = ref.model_matern(dt=1.0 / hz); P0 = np.eye(4); K = int(DUR * hz); Y = rng.normal(size=(K, 3)); okm = ref.gaps_mask(K, 0.03); zu = np.zeros(K, bool); zu[: hz * 3] = True; zu[K // 2: K // 2 + hz] = True
        r1 = timed("hand-held %3d Hz - E62 reference" % hz, ref.k2, Y, okm, zu, A, Q, P0)
        r2 = timed("hand-held %3d Hz - tight" % hz, k2_tight, Y, okm, zu, A, Q, P0)
        print("      max |difference| = %.2e" % np.abs(r1 - r2).max(), flush=True)
    for hz in (50, 25):
        A, Qd, P0 = rotor_model(hz); K = int(DUR * hz); Y = rng.normal(size=(K, 3)); okm = ref.gaps_mask(K, 0.02); fl = np.ones(K, bool); zu = np.zeros(K, bool); zu[: hz * 3] = True
        ew = rng.normal(size=(K, 2, 3)); ew /= np.linalg.norm(ew, axis=2, keepdims=True)
        r1 = timed("rotor %3d Hz (17 states) - E62 reference" % hz, ref.k3, Y, okm, fl, zu, ew, A, Qd, P0)
        for se in (1, 4):
            r2 = timed("rotor %3d Hz (17 states) - tight, symmetrise every %d" % (hz, se), k3_tight, Y, okm, fl, zu, ew, A, Qd, P0, 0.187, None, se)
            print("      max |position difference| = %.2e m, velocity %.2e m/s" % (np.abs(r1[:, :3] - r2[:, :3]).max(), np.abs(r1[:, 3:] - r2[:, 3:]).max()), flush=True)
