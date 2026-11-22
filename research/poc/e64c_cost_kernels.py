#!/usr/bin/env python3
"""E64c - the remaining front-end kernels of the Phase 4 design, timed in the form the blueprint prescribes (feasibility PoC):
  G   gravity smoother call (IRW, 2 states) on C columns with CACHED gains: the covariance/gain recursion depends only on the time grid,
      the validity mask and the gap jumps, so it is computed once per input (per measurement-noise setting) and every later call runs
      only the mean recursions; (a) vectorised affine scans, (b) pure-Python loop; rates 25 / 12.5 / 6.25 Hz
  R   the cached gain computation itself (pure-Python floats), once
  S   strapdown pass at 200 Hz (log-depth quaternion prefix scan in chunks of 4,096; E50 design)
  L   zero-phase Butterworth low-pass by FFT (|H|^2, odd-extension padding) on 36,820 x 3 samples; compared with a reference
      forward-backward biquad (pure Python) in the interior
Usage: python e64c_cost_kernels.py"""
import os
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ.setdefault(_k, "1")
import sys, time, tracemalloc, platform
import numpy as np
DUR = 184.1; rng = np.random.default_rng(3)

def gains(K, ok, jump, dt, tau=3.0, r=1.0):
    """IRW Kalman gains and RTS coefficients (pure-Python floats). Returns g (K,2) and the backward map C (K-1,2,2)."""
    q = 1.0 / tau ** 4; q11, q12, q22 = q * dt ** 3 / 3, q * dt ** 2 / 2, q * dt; okl = ok.tolist(); jl = jump.tolist()
    p00, p01, p11 = 1e6, 0.0, 1e2; G = [None] * K; PF = [None] * K; PP = [None] * K
    for k in range(K):
        if k: p00, p01, p11 = p00 + 2 * dt * p01 + dt * dt * p11 + q11, p01 + dt * p11 + q12, p11 + q22
        p00 += jl[k]; p11 += 0.01 * jl[k]; PP[k] = (p00, p01, p11)
        if okl[k]: s = p00 + r; g0, g1 = p00 / s, p01 / s; p00, p01, p11 = p00 - g0 * p00, p01 - g0 * p01, p11 - g1 * p01
        else: g0 = g1 = 0.0
        G[k] = (g0, g1); PF[k] = (p00, p01, p11)
    C = [None] * (K - 1)
    for k in range(K - 1):
        f00, f01, f11 = PF[k]; b00, b01, b10, b11 = f00 + dt * f01, f01, f01 + dt * f11, f11; d00, d01, d11 = PP[k + 1]; det = d00 * d11 - d01 * d01
        C[k] = ((b00 * d11 - b01 * d01) / det, (-b00 * d01 + b01 * d00) / det, (b10 * d11 - b11 * d01) / det, (-b10 * d01 + b11 * d00) / det)
    return np.array(G), np.array(C).reshape(K - 1, 2, 2)

def scan(M, u, x0, reverse=False):
    if reverse: M, u = M[::-1], u[::-1]
    M = M.copy(); u = u.copy(); K = len(M); d = 1
    while d < K:
        u[d:] = np.einsum("kij,kjc->kic", M[d:], u[:-d]) + u[d:]; M[d:] = np.einsum("kij,kjl->kil", M[d:], M[:-d]); d *= 2
    x = np.einsum("kij,jc->kic", M, x0) + u
    return x[::-1] if reverse else x

def call_scan(y, G, C, dt, Mf=None):
    """Mean recursions of one call with cached gains. Mf (forward maps) may be cached too (depends on gains only)."""
    K, Cc = y.shape; A = np.array([[1.0, dt], [0.0, 1.0]])
    if Mf is None: IG = np.repeat(np.eye(2)[None], K, 0); IG[:, :, 0] -= G; Mf = IG @ A; Mf[0] = IG[0]
    xf = scan(Mf, G[:, :, None] * y[:, None, :], np.vstack([y[0], np.zeros(Cc)]))
    v = xf[:-1] - np.einsum("kij,jl,klc->kic", C, A, xf[:-1]); s = scan(C, v, xf[-1], reverse=True)
    return np.vstack([s[:, 0], xf[-1:, 0]])

def call_python(y, G, C, dt):
    K, Cc = y.shape; g0 = G[:, 0].tolist(); g1 = G[:, 1].tolist(); c = C.reshape(K - 1, 4).tolist(); out = np.empty((K, Cc)); Y = y.T.tolist()
    for j in range(Cc):
        yj = Y[j]; m0, m1 = yj[0], 0.0; f0 = [0.0] * K; f1 = [0.0] * K
        for k in range(K):
            if k: m0 += dt * m1
            a = g0[k]
            if a: e = yj[k] - m0; m0 += a * e; m1 += g1[k] * e
            f0[k] = m0; f1[k] = m1
        s0, s1 = f0[-1], f1[-1]; col = [0.0] * K; col[-1] = s0
        for k in range(K - 2, -1, -1):
            c00, c01, c10, c11 = c[k]; a0, a1 = f0[k], f1[k]; e0 = s0 - (a0 + dt * a1); e1 = s1 - a1; s0 = a0 + c00 * e0 + c01 * e1; s1 = a1 + c10 * e0 + c11 * e1; col[k] = s0
        out[:, j] = col
    return out

def qmul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]; bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)
def strapdown(w, dt, chunk=4096):
    N = len(w); q = np.empty((N, 4), np.float32); q0 = np.array([0, 0, 0, 1.0])
    for s in range(0, N, chunk):
        wc = w[s:s + chunk].astype(np.float64) * dt; th = np.linalg.norm(wc, axis=1, keepdims=True); k = np.where(th > 1e-12, np.sin(th / 2) / np.maximum(th, 1e-12), 0.5); o = np.concatenate([wc * k, np.cos(th / 2)], 1); st = 1
        while st < len(o): o[st:] = qmul(o[:-st], o[st:]); st *= 2
        o = qmul(np.broadcast_to(q0, o.shape), o); o /= np.linalg.norm(o, axis=1, keepdims=True); q[s:s + chunk] = o; q0 = o[-1]
    return q

def butter2(fc, fs):
    """2nd-order Butterworth low-pass (bilinear transform with prewarping): b, a."""
    k = np.tan(np.pi * fc / fs); n = 1 + np.sqrt(2) * k + k * k; b = np.array([k * k, 2 * k * k, k * k]) / n; a = np.array([1.0, 2 * (k * k - 1) / n, (1 - np.sqrt(2) * k + k * k) / n]); return b, a
def lowpass_fft(x, fs, fc, pad=None):
    """Zero-phase low-pass = |H(f)|^2 of the 2nd-order Butterworth, applied in the frequency domain after odd-extension padding."""
    N = len(x); pad = pad or min(N - 1, int(6 * fs / fc)); xe = np.concatenate([2 * x[0] - x[pad:0:-1], x, 2 * x[-1] - x[-2:-pad - 2:-1]]); L = len(xe); nf = 1 << int(np.ceil(np.log2(L)))
    b, a = butter2(fc, fs); w = np.exp(-2j * np.pi * np.fft.rfftfreq(nf) ); H = (b[0] + b[1] * w + b[2] * w * w) / (a[0] + a[1] * w + a[2] * w * w)
    y = np.fft.irfft(np.fft.rfft(xe, nf, axis=0) * (np.abs(H) ** 2)[:, None], nf, axis=0)[:L]; return y[pad:pad + N]
def lowpass_ref(x, fs, fc):
    b, a = butter2(fc, fs); out = np.empty_like(x)
    for j in range(x.shape[1]):
        def run(v):
            y = [0.0] * len(v); x1 = x2 = v[0]; y1 = y2 = v[0]
            for i, xi in enumerate(v): yi = b[0] * xi + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2; x2, x1, y2, y1 = x1, xi, y1, yi; y[i] = yi
            return y
        out[:, j] = run(run(x[:, j].tolist())[::-1])[::-1]
    return out

def timed(label, fn, *a, n_rep=1):
    tracemalloc.start(); t0 = time.perf_counter()
    for _ in range(n_rep): out = fn(*a)
    dt_ = (time.perf_counter() - t0) / n_rep; peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
    print("  %-66s %7.4f s  %6.3f ms per IMU-second  peak %.2f MB" % (label, dt_, 1e3 * dt_ / DUR, peak / 1e6), flush=True); return out

if __name__ == "__main__":
    print("E64c front-end kernels: %s, Python %s, numpy %s" % (platform.platform(), sys.version.split()[0], np.__version__), flush=True)
    for hz in (25.0, 12.5, 6.25):
        K = int(DUR * hz); dt = 1 / hz; ok = rng.random(K) > 0.002; jump = np.where(rng.random(K) < 0.2, 1e-8, 0.0); y = np.cumsum(rng.normal(size=(K, 6)), 0) * 0.01 + 9.8
        G, C = timed("R  gain + RTS-coefficient recursion, %5.2f Hz (once per input)" % hz, gains, K, ok, jump, dt)
        a = timed("G  call, 3 columns, %5.2f Hz - affine scans" % hz, call_scan, y[:, :3], G, C, dt, n_rep=3)
        b = timed("G  call, 3 columns, %5.2f Hz - pure-Python loop" % hz, call_python, y[:, :3], G, C, dt, n_rep=3)
        timed("G  call, 6 columns, %5.2f Hz - affine scans" % hz, call_scan, y, G, C, dt, n_rep=3)
        print("      scan vs loop max |difference| %.2e" % np.abs(a - b).max(), flush=True)
    N = 36820; w = (rng.normal(size=(N, 3)) * 0.3).astype(np.float32)
    timed("S  strapdown pass, 36,820 samples (prefix scan, chunks of 4,096)", strapdown, w, 0.005, n_rep=3)
    x = np.cumsum(rng.normal(size=(N, 3)), 0) * 0.05 + np.array([0.0, 0.0, 9.81])
    y1 = timed("L  zero-phase 1 Hz low-pass by FFT, 36,820 x 3", lowpass_fft, x, 200.0, 1.0, n_rep=3)
    y2 = timed("L  zero-phase 15 Hz low-pass by FFT, 36,820 x 3", lowpass_fft, x, 200.0, 15.0, n_rep=3)
    r1 = timed("L  reference forward-backward biquad (pure Python), 1 Hz", lowpass_ref, x, 200.0, 1.0)
    r2 = lowpass_ref(x, 200.0, 15.0); m = slice(2000, N - 2000)
    print("      FFT vs biquad filtfilt, interior max |difference|: 1 Hz %.2e, 15 Hz %.2e (signal std %.2f)" % (np.abs(y1[m] - r1[m]).max(), np.abs(y2[m] - r2[m]).max(), x.std()), flush=True)
