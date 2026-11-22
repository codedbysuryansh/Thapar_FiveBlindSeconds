#!/usr/bin/env python3
"""Five Blind Seconds - IMU-only 6-DoF trajectory estimation under sensor degradation.
Samsung PRISM XR Hackathon 2026, Track 03 "Path Finder: Trajectory Estimation Under Sensor Degradation".

    python main.py [INPUT] [OUTPUT]          (also -i/--input and -o/--output)

INPUT   a .zip (searched for */imu0/data.csv, else data.csv, else the only .csv), a folder (same search) or a CSV file;
        default: the working directory, searched the same way.
OUTPUT  a file path, or a folder (-> FOLDER/estimated_trajectory.txt); default ./estimated_trajectory.txt.

Writes exactly one pose per input row, in input order, duplicated and blind rows included:
    "<sec>.<9-digit ns> tx ty tz qx qy qz qw"   (TUM format, metres, unit quaternion, world frame with z up)
Python + NumPy only; one thread; no network, no other files, no randomness. Never exits without a valid output file.

Method (Phase 3 freeze report, sections 3 and 12; submission configuration V5P4): uniform slot grid with invalid slots for
dropped, zero, NaN and stuck rows; the trailing invalid run is the blind window; integrated-random-walk gravity smoother with a
gap-length-squared rotation-uncertainty law on 12.5 Hz block means; thrust-lock platform gate; gyro bias as a 20 s-knot spline
(rotor) or constant / rest-anchored (hand-held, unless a drift test fires); position as a Kalman smoother of a stationary motion
prior measured by the world-frame specific force (rotor: 17-state spring-damper + rotor drag + lever arm at 25 Hz; hand-held:
Matern-5/2 at 100 Hz), zero-velocity rows at genuine rest, gap-scaled accelerometer-bias variance; radius guard; blind window:
rotor = damped velocity (0.5 s), hand-held = the smoother's own prediction, orientation held."""
import os, sys
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"          # single CPU thread, set before NumPy loads its BLAS
import math, zipfile
import numpy as np

__version__ = "1.0.0"
G = 9.81
CFG = dict(k2=0.25, gate=0.035, thr_hand=2.0e-3, health=0.9, lam2=3.0, knot=20.0, gap_ref=2.0, dec_rotor=8, dec_hand=2, guard=12.0, gs_dec=16)
K_DRAG = 0.1873                                   # rotor-drag coefficient [1/s]: mean of 11 per-flight fits (Phase 2)
ROTOR = (20.0, 2.0, 0.3, 1.0, 20.0, 0.1)          # spring-damper prior: T_p [s], tau_v [s], tau_a [s], sigma_a, tau_b [s], sigma_b
MATERN = (1.0, 2.0, 20.0, 0.1)                    # Matern-5/2 prior: ell [s], sigma_p [m], tau_b [s], sigma_b
SD_DRAG, SZ, BIAS_JUMP, TAU_C, SC, LEVER_PRIOR, TAU_G = 1.4, 0.02, 0.5, 100.0, 0.15, 0.15, 3.0
EZ = np.array([0.0, 0.0, 1.0])
TRACE = None                                      # tests set a dict here to receive intermediate results

def log(msg):
    sys.stderr.write("[main] %s\n" % msg)

# =============================================================================================== quaternions (x, y, z, w)
def qmul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]; bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)
def qinv(q): return q * np.array([-1.0, -1.0, -1.0, 1.0])
def qrot(q, v):
    u = q[..., :3]; w = q[..., 3:4]; c = np.cross(u, v); return v + 2 * w * c + 2 * np.cross(u, c)
def qexp(rv):
    th = np.linalg.norm(rv, axis=-1, keepdims=True); k = np.where(th > 1e-9, np.sin(th / 2) / np.maximum(th, 1e-12), 0.5)
    return np.concatenate([rv * k, np.cos(th / 2)], -1)
def qlog(q):
    q = q * np.where(q[..., 3:4] < 0, -1.0, 1.0); n = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    return q[..., :3] * np.where(n > 1e-9, 2 * np.arctan2(n, q[..., 3:4]) / np.maximum(n, 1e-12), 2.0)
def qcumprod(dq):
    out = dq.copy(); s = 1
    while s < len(out):
        out[s:] = qmul(out[:-s], out[s:]); out /= np.linalg.norm(out, axis=1, keepdims=True); s *= 2
    return out
def integrate_gyro(t, w):
    dt = np.diff(t)[:, None]; dq = qexp(0.5 * (w[:-1] + w[1:]) * dt)
    return qcumprod(np.vstack([[0, 0, 0, 1.0], dq]))
def shortest_arc(u, v):
    c = np.cross(u, v); d = np.sum(u * v, -1, keepdims=True); q = np.concatenate([c, 1.0 + d], -1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)
def rot_rows01(q):
    """Rows 0 and 1 of the rotation matrix of q (scalar last), normalised first; the formulas of scipy's Rotation.as_matrix."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True); x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w; xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    return np.stack([x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw), 2 * (xy + zw), -x2 + y2 - z2 + w2, 2 * (yz - xw)], 1).reshape(-1, 2, 3)

# =============================================================================================== filters and small numerics
def _butter2(fc, fs):
    """2nd-order Butterworth low-pass by the bilinear transform with prewarping (= scipy.signal.butter(2, fc/(fs/2)))."""
    k = math.tan(math.pi * fc / fs); n = 1 + math.sqrt(2) * k + k * k
    return k * k / n, 2 * k * k / n, k * k / n, 2 * (k * k - 1) / n, (1 - math.sqrt(2) * k + k * k) / n
def _homog(a1, a2, v0, v1, L):
    """c_n = e0 . M^n v, n < L, for the biquad state matrix M = [[-a1, 1], [-a2, 0]] (direct form II transposed)."""
    c = [0.0] * L; x0, x1 = v0, v1
    for n in range(L):
        c[n] = x0; x0, x1 = -a1 * x0 + x1, -a2 * x0
    return np.array(c)
def _lfilter_ic(x, h, c1, c2, z1, z2):
    """IIR output with initial DF2T state (z1, z2) per column = convolution with the impulse response + homogeneous response."""
    n = x.shape[0]; L = len(h); nf = 1 << int(math.ceil(math.log2(n + L)))
    y = np.fft.irfft(np.fft.rfft(x, nf, axis=0) * np.fft.rfft(h, nf)[:, None], nf, axis=0)[:n]
    m = min(n, len(c1)); y[:m] += c1[:m, None] * z1[None, :] + c2[:m, None] * z2[None, :]
    return y
def lowpass(x, fs, fc):
    """Zero-phase 2nd-order Butterworth = scipy.signal.sosfiltfilt(butter(2, fc/(fs/2), output='sos'), x, axis=0), defaults
    (odd extension of 9 samples, steady-state initial conditions scaled by the first sample of each pass)."""
    x = np.asarray(x, float); one = x.ndim == 1; X = x[:, None] if one else x; n = X.shape[0]
    b0, b1, b2, a1, a2 = _butter2(fc, fs); edge = min(9, n - 1)
    ext = np.concatenate([2 * X[:1] - X[edge:0:-1], X, 2 * X[-1:] - X[-2:-edge - 2:-1]], 0)
    zi = np.linalg.solve(np.array([[1 + a1, -1.0], [a2, 1.0]]), np.array([b1 - a1 * b0, b2 - a2 * b0]))
    rho = math.sqrt(abs(a2)) if a1 * a1 < 4 * a2 else max(abs((-a1 + s * math.sqrt(a1 * a1 - 4 * a2)) / 2) for s in (-1, 1))
    L = min(len(ext), int(math.ceil(math.log(1e-18) / math.log(rho))) + 16 if 0 < rho < 1 else len(ext))
    u0, u1 = b1 - a1 * b0, b2 - a2 * b0; h = np.concatenate([[b0], _homog(a1, a2, u0, u1, L - 1)])
    c1 = _homog(a1, a2, 1.0, 0.0, L); c2 = _homog(a1, a2, 0.0, 1.0, L)
    y = _lfilter_ic(ext, h, c1, c2, zi[0] * ext[0], zi[1] * ext[0])
    y = _lfilter_ic(y[::-1], h, c1, c2, zi[0] * y[-1], zi[1] * y[-1])[::-1]
    y = y[edge:edge + n]
    return y[:, 0] if one else y
def fill_gaps(t, x, valid):
    if valid.all(): return x
    tv = t[valid]; return np.stack([np.interp(t, tv, x[valid, i]) for i in range(x.shape[1])], 1)
def segments(valid):
    d = np.diff(np.concatenate([[0], valid.astype(np.int8), [0]])); return np.flatnonzero(d == 1), np.flatnonzero(d == -1)
def hf_std(x): return float(np.diff(x, axis=0).std() / np.sqrt(2))

def expm(A):
    """Matrix exponential, scaling and squaring with a [13/13] Pade approximant (Higham 2005)."""
    b = (64764752532480000., 32382376266240000., 7771770303897600., 1187353796428800., 129060195264000., 10559470521600.,
         670442572800., 33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.)
    n = A.shape[0]; I = np.eye(n); nrm = np.abs(A).sum(0).max(); s = max(0, int(math.ceil(math.log2(nrm / 5.371920351148152)))) if nrm > 0 else 0
    A = A / (2 ** s); A2 = A @ A; A4 = A2 @ A2; A6 = A4 @ A2
    U = A @ (A6 @ (b[13] * A6 + b[11] * A4 + b[9] * A2) + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * I)
    V = A6 @ (b[12] * A6 + b[10] * A4 + b[8] * A2) + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * I
    R = np.linalg.solve(V - U, V + U)
    for _ in range(s): R = R @ R
    return R
def lyap(F, Qc):
    """P with F P + P F' + Qc = 0 (stationary covariance), by the Kronecker form."""
    n = F.shape[0]; I = np.eye(n); P = np.linalg.solve(np.kron(I, F) + np.kron(F, I), -Qc.reshape(-1)).reshape(n, n); return 0.5 * (P + P.T)
def van_loan(F, Qc, dt):
    n = F.shape[0]; M = np.zeros((2 * n, 2 * n)); M[:n, :n] = -F; M[:n, n:] = Qc; M[n:, n:] = F.T
    E = expm(M * dt); A = E[n:, n:].T; Q = A @ E[:n, n:]; return A, 0.5 * (Q + Q.T)
def discretise(F, Qc, t, round_s=1e-5):
    dts = np.round(np.diff(t) / round_s).astype(np.int64); u, inv = np.unique(dts, return_inverse=True); A = []; Q = []
    for d in u:
        a, q = van_loan(F, Qc, max(int(d), 0) * round_s); A.append(a); Q.append(q)
    return A, Q, np.concatenate([[0], inv]).astype(np.int64)
def model_matern52(ell, sp, tau_b, sb):
    lam = math.sqrt(5.0) / ell; F = np.zeros((4, 4)); F[0, 1] = 1; F[1, 2] = 1; F[2, :3] = [-lam ** 3, -3 * lam ** 2, -3 * lam]; F[3, 3] = -1.0 / tau_b
    Qc = np.zeros((4, 4)); Qc[2, 2] = 16.0 / 3.0 * sp ** 2 * lam ** 5; Qc[3, 3] = 2 * sb ** 2 / tau_b
    return F, Qc, lyap(F, Qc), np.array([0, 0, 1.0, 1.0])
def model_spring(T_p, tau_v, tau_a, sa, tau_b, sb):
    w0 = 2 * math.pi / T_p; c = 1.0 / tau_v; F = np.zeros((4, 4)); F[0, 1] = 1; F[1, 0] = -w0 ** 2; F[1, 1] = -c; F[1, 2] = 1
    F[2, 2] = -1.0 / tau_a; F[3, 3] = -1.0 / tau_b; Qc = np.zeros((4, 4)); Qc[2, 2] = 2 * sa ** 2 / tau_a; Qc[3, 3] = 2 * sb ** 2 / tau_b
    return F, Qc, lyap(F, Qc), np.array([-w0 ** 2, -c, 1.0, 1.0])

# =============================================================================================== input / output
def _find_member(names):
    names = [n for n in names if not n.startswith("__MACOSX") and not n.endswith("/")]; low = [n.replace("\\", "/").lower() for n in names]
    for want in ("imu0/data.csv", "data.csv"):
        hits = sorted((len(l), n) for n, l in zip(names, low) if l == want or l.endswith("/" + want))
        if hits: return hits[0][1]
    csv = [n for n, l in zip(names, low) if l.endswith(".csv")]
    return csv[0] if len(csv) == 1 else (sorted(csv, key=len)[0] if csv else None)
def open_input(path):
    """A binary stream of the IMU CSV inside INPUT (zip, folder or file)."""
    path = path or "."
    if os.path.isdir(path):
        files = []
        for root, _d, fs in os.walk(path):
            for f in fs: files.append(os.path.relpath(os.path.join(root, f), path))
        m = _find_member([f for f in files if f.lower().endswith(".csv")])
        if m is not None: return open(os.path.join(path, m), "rb")
        zips = [f for f in files if f.lower().endswith(".zip")]
        if len(zips) >= 1: return open_input(os.path.join(path, sorted(zips, key=len)[0]))
        raise FileNotFoundError("no IMU CSV found under %s" % path)
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path); m = _find_member(z.namelist())
        if m is None: raise FileNotFoundError("no CSV inside %s" % path)
        return z.open(m)
    return open(path, "rb")

def _ts_any(tok):
    try: return int(tok)
    except ValueError: v = float(tok)
    s = 1e9 if abs(v) < 1e11 else (1e6 if abs(v) < 1e14 else (1e3 if abs(v) < 1e17 else 1.0)); return int(round(v * s))
def _parse_block(block, TS, VA):
    good = []
    for ln in block.split(b"\n"):
        s = ln.strip()
        if not s: continue
        c = s[0]
        if c == 35 or 65 <= c <= 90 or 97 <= c <= 122: continue          # comment or header line
        good.append(s)
    if not good: return
    toks = b" ".join(good).replace(b",", b" ").split()
    if len(toks) == 7 * len(good):
        try:
            arr = np.array(toks).reshape(-1, 7); TS.append(arr[:, 0].astype(np.int64)); VA.append(arr[:, 1:].astype(np.float64)); return
        except ValueError: pass
    ts = []; va = []
    for s in good:                                                      # slow path: irregular lines, one at a time
        f = [x.strip() for x in s.split(b",")] if b"," in s else s.split()
        try: tv = _ts_any(f[0])
        except (ValueError, OverflowError): continue
        row = []
        for j in range(1, 7):
            try: row.append(float(f[j]) if j < len(f) and f[j] else float("nan"))
            except ValueError: row.append(float("nan"))
        ts.append(tv); va.append(row)
    if ts: TS.append(np.array(ts, np.int64)); VA.append(np.array(va, np.float64))
def read_imu(stream, chunk=1 << 18):
    TS, VA = [], []; rest = b""
    while True:
        buf = stream.read(chunk)
        if not buf: break
        data = rest + buf; cut = data.rfind(b"\n")
        if cut < 0: rest = data; continue
        rest = data[cut + 1:]; _parse_block(data[:cut], TS, VA)
    if rest.strip(): _parse_block(rest, TS, VA)
    if not TS: return np.zeros(0, np.int64), np.zeros((0, 6))
    return np.concatenate(TS), np.concatenate(VA)
def unit_guard(tn, X):
    """Defensive unit checks (never triggered by the official format): ns timestamps, m/s^2, rad/s."""
    d = np.diff(tn); d = d[d > 0]
    if len(d):
        med = float(np.median(d))
        if med < 2e4:                                                   # > 50 kHz in ns: the stamps are in us, ms or s
            for sc in (1e3, 1e6, 1e9):
                if 1e9 / (med * sc) <= 5000: tn = tn * int(sc); log("timestamps rescaled by %g" % sc); break
    ok = np.isfinite(X).all(1) & ~(X == 0).all(1)
    if ok.sum() > 10:
        if np.median(np.linalg.norm(X[ok, 3:], axis=1)) < 2.0: X[:, 3:] *= 9.80665; log("accelerometer in g -> m/s^2")
        if np.median(np.linalg.norm(X[ok, :3], axis=1)) > 20.0: X[:, :3] *= math.pi / 180; log("gyro in deg/s -> rad/s")
    return tn, X

def render(tn, P, Q, dp=6, dq=7):
    """'%d.%09d %.6f %.6f %.6f %.7f %.7f %.7f %.7f\\n' for every row, vectorised (fixed-point digits, pad bytes removed)."""
    N = len(tn); sec, ns = np.divmod(tn.astype(np.int64), 10 ** 9)
    def digits(v, width):
        out = np.zeros((N, width), np.uint8); x = v.copy()
        for j in range(width - 1, -1, -1): out[:, j] = 48 + (x % 10).astype(np.uint8); x //= 10
        nd = np.maximum(1, np.floor(np.log10(np.maximum(v, 1))).astype(int) + 1); out[np.arange(width)[None, :] < (width - nd)[:, None]] = 0; return out
    def fixed(x, dec):
        neg = np.signbit(x); s = np.round(np.abs(x) * 10 ** dec).astype(np.int64); ip, fp = np.divmod(s, 10 ** dec)
        F_ = np.zeros((N, dec), np.uint8); y = fp.copy()
        for j in range(dec - 1, -1, -1): F_[:, j] = 48 + (y % 10); y //= 10
        return np.hstack([np.where(neg, 45, 0).astype(np.uint8)[:, None], digits(ip, 9), np.full((N, 1), 46, np.uint8), F_])
    nsd = np.zeros((N, 9), np.uint8); y = ns.copy()
    for j in range(8, -1, -1): nsd[:, j] = 48 + (y % 10); y //= 10
    sp = np.full((N, 1), 32, np.uint8); parts = [digits(sec, 12), np.full((N, 1), 46, np.uint8), nsd]
    for j in range(3): parts += [sp, fixed(P[:, j], dp)]
    for j in range(4): parts += [sp, fixed(Q[:, j], dq)]
    parts.append(np.full((N, 1), 10, np.uint8)); buf = np.hstack(parts).ravel(); return buf[buf != 0].tobytes()
def write_trajectory(path, tn, P, Q, chunk=4096):
    with open(path, "wb") as f:
        for s in range(0, len(tn), chunk): f.write(render(tn[s:s + chunk], P[s:s + chunk], Q[s:s + chunk]))

# =============================================================================================== M1-M3: timeline, blind split, repair
def ingest2(tn, X):
    """Uniform slot grid at the median timestamp step. Rows with the same timestamp share a slot; zero, NaN and exact-repeat
    rows are invalid (first sample of a stuck run kept); missing rows become invalid slots. Returns t, w, a, valid, rows."""
    tn = np.asarray(tn, np.int64); d = np.diff(tn); med = float(np.median(d[d > 0]))
    st = np.maximum(np.round(d / med).astype(np.int64), 0); rows = np.concatenate([[0], np.cumsum(st)]); N = int(rows[-1]) + 1
    ur, first = np.unique(rows, return_index=True); t = np.interp(np.arange(N), ur, (tn[first] - tn[0]) * 1e-9)
    Xu = np.full((N, 6), np.nan); Xu[rows] = X; bad = np.isnan(Xu).any(1) | (Xu == 0).all(1)
    same = np.zeros(N, bool); same[1:] = (Xu[1:] == Xu[:-1]).all(1) & ~bad[1:]
    valid = ~(bad | same); Xu[~valid] = np.nan
    return t, Xu[:, :3], Xu[:, 3:], valid, rows
def repair(t, w, a, valid, spike_k=10.0, floor_a=15.0, floor_w=1.5, g_lo=5.0, g_hi=15.5, win=1.0):
    """Isolated outliers far above the local robust spread are interpolated (stay valid); windows whose 1 s median |a| is
    outside [g_lo, g_hi] m/s^2 become invalid."""
    w = w.copy(); a = a.copy(); valid = valid.copy(); fs = 1.0 / np.median(np.diff(t)); N = len(t)
    Xf = np.hstack([fill_gaps(t, w, valid), fill_gaps(t, a, valid)]); pad = np.pad(Xf, ((2, 2), (0, 0)), mode="edge")
    med = np.median(np.stack([pad[i:i + N] for i in range(5)]), 0); r = np.abs(Xf - med); n1 = int(fs); m = N // n1
    sig = np.repeat(np.median(r[:m * n1].reshape(m, n1, 6), 1) * 1.4826, n1, 0) if m else np.zeros((0, 6))
    sig = np.vstack([sig, np.repeat(sig[-1:], N - len(sig), 0)]) if len(sig) else np.zeros((N, 6))
    floor = np.array([floor_w] * 3 + [floor_a] * 3); spike = ((r > np.maximum(floor, spike_k * sig)).any(1)) & valid; ns = int(spike.sum())
    if ns:
        good = valid & ~spike
        for j in range(3): w[spike, j] = np.interp(t[spike], t[good], w[good, j]); a[spike, j] = np.interp(t[spike], t[good], a[good, j])
    an = np.linalg.norm(fill_gaps(t, a, valid), axis=1); h = max(1, int(win * fs) // 2); k = max(1, int(fs * 0.1)); idx = np.arange(0, N, k)
    medn = np.array([np.median(an[max(0, i - h):i + h + 1]) for i in idx])
    bad = np.interp(np.arange(N), idx, ((medn < g_lo) | (medn > g_hi)).astype(float)) > 0.0
    if bad.any():
        d = int(0.5 * fs); bad = np.convolve(bad.astype(float), np.ones(2 * d + 1), "same") > 0; bad &= valid; valid = valid & ~bad; w[~valid] = np.nan; a[~valid] = np.nan
    return w, a, valid, dict(spikes=ns, implausible=int(bad.sum()))

# =============================================================================================== M4-M5: platform statistics and gate
def thrust_lock_fill(t, af, valid, win=10.0):
    """Median over 10 s windows (5 s hop) of RMS(f_perp)/g of the 1 Hz low-passed, gap-filled specific force."""
    fs = 1.0 / np.median(np.diff(t)); n = int(win * fs); al = lowpass(fill_gaps(t, af, valid), fs, 1.0); out = []
    for i in range(0, len(t) - n + 1, n // 2):
        v = valid[i:i + n]
        if v.sum() < n // 2: continue
        x = al[i:i + n][v]; zb = x.mean(0); zb = zb / np.linalg.norm(zb); perp = x - np.outer(x @ zb, zb); out.append(np.sqrt((perp ** 2).sum(1).mean()) / G)
    return float(np.median(out)) if out else 1.0
def rest_info(t, w, a, valid, win=0.5, frac=0.05, gate=0.035, min_windows=3):
    """Genuine rest: clusters of >= 3 quiet 0.5 s windows whose gyro means agree. Returns (anchors [(time, bias)], mask)."""
    mask = np.zeros(len(t), bool); fs = 1.0 / np.median(np.diff(t)); n = max(8, int(win * fs)); m = len(t) // n
    idx = np.arange(m * n).reshape(m, n); okw = valid[idx].all(1)
    if okw.sum() < 10: return [], mask
    Wn = np.nan_to_num(w[:m * n]).reshape(m, n, 3); An = np.nan_to_num(a[:m * n]).reshape(m, n, 3)
    gs = Wn.std(1).max(1); as_ = An.std(1).max(1); gs[~okw] = np.inf; as_[~okw] = np.inf
    sc = gs / np.median(gs[okw]) + as_ / np.median(as_[okw]); k = max(3, int(frac * okw.sum())); cand = np.argsort(sc)[:k]; cand = np.sort(cand[gs[cand] < gate]); out = []
    if len(cand) == 0: return out, mask
    for g in np.split(cand, np.where(np.diff(cand) > 4)[0] + 1):
        if len(g) < min_windows: continue
        means = Wn[g].mean(1); spread = means.std(0).max(); allow = 1.0e-3 + 3.0 * gs[g].mean() / np.sqrt(n)
        if spread < allow: out.append((float(t[idx[g]].mean()), np.median(means, 0))); mask[idx[g].ravel()] = True
    return out, mask
def bias_track(t, anchors):
    if not anchors: return None
    ta = np.array([x[0] for x in anchors]); B = np.array([x[1] for x in anchors]); return np.stack([np.interp(t, ta, B[:, i]) for i in range(3)], 1)

# =============================================================================================== M6-M8: gyro bridge, strapdown, gravity smoother
def bridge_gyro(t, w, valid, tau=0.4, mwin=5.0):
    """Ornstein-Uhlenbeck bridge across each interior gap: conditional mean pinned to the two end samples, reverting to the
    local mean (+-5 s); short gaps reduce to linear interpolation."""
    out = fill_gaps(t, w, valid)
    if valid.all(): return out
    N = len(t); fs = 1.0 / np.median(np.diff(t)); k = int(mwin * fs); i0, i1 = segments(~valid); keep = (i0 > 0) & (i1 < N); i0, i1 = i0[keep], i1[keep]
    if not len(i0): return out
    wv = np.where(valid[:, None], np.nan_to_num(w), 0.0); cs = np.vstack([np.zeros((1, 3)), np.cumsum(wv, 0)]); cv = np.concatenate([[0], np.cumsum(valid)])
    lo = np.maximum(0, i0 - k); hi = np.minimum(N, i1 + k)
    mu = 0.5 * ((cs[i0] - cs[lo]) / (cv[i0] - cv[lo])[:, None] + (cs[hi] - cs[i1]) / (cv[hi] - cv[i1])[:, None])
    lens = i1 - i0; g = np.repeat(np.arange(len(i0)), lens); pos = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens); j = i0[g] + pos
    T = (t[i1] - t[i0 - 1])[g]; tt = t[j] - t[i0 - 1][g]; den = 1 - np.exp(-2 * T / tau)
    s0 = np.exp(-tt / tau) * (1 - np.exp(-2 * (T - tt) / tau)) / den; s1 = np.exp(-(T - tt) / tau) * (1 - np.exp(-2 * tt / tau)) / den
    m_ = mu[g]; out = out.copy(); out[j] = m_ + s0[:, None] * (w[i0 - 1][g] - m_) + s1[:, None] * (w[i1][g] - m_); return out

def _scan(m00, m01, m10, m11, u0, u1, x0, x1):
    """x_k = M_k x_{k-1} + u_k (2-vectors, C columns), x_{-1} = (x0, x1): log-depth (Hillis-Steele) composition of affine maps."""
    m00, m01, m10, m11 = m00.copy(), m01.copy(), m10.copy(), m11.copy(); u0, u1 = u0.copy(), u1.copy(); K = len(m00); d = 1
    while d < K:
        a00, a01, a10, a11 = m00[d:, None], m01[d:, None], m10[d:, None], m11[d:, None]; p0, p1 = u0[:-d], u1[:-d]
        n0 = a00 * p0 + a01 * p1 + u0[d:]; n1 = a10 * p0 + a11 * p1 + u1[d:]
        b00, b01, b10, b11 = m00[:-d], m01[:-d], m10[:-d], m11[:-d]; A00, A01, A10, A11 = m00[d:], m01[d:], m10[d:], m11[d:]
        c00 = A00 * b00 + A01 * b10; c01 = A00 * b01 + A01 * b11; c10 = A10 * b00 + A11 * b10; c11 = A10 * b01 + A11 * b11
        u0[d:] = n0; u1[d:] = n1; m00[d:] = c00; m01[d:] = c01; m10[d:] = c10; m11[d:] = c11; d *= 2
    return m00[:, None] * x0 + m01[:, None] * x1 + u0, m10[:, None] * x0 + m11[:, None] * x1 + u1
class GravitySmoother:
    """= fbs6.gravity_smooth_dec (IRW Kalman smoother of each column on block means of d slots, unit measurement variance per
    sample, gap jump (9.81 min(k2 T^2/2, 1)/sig_m)^2 at each re-start). The covariance / gain recursion depends only on the grid,
    the mask and the jumps, so it is computed once per noise setting; every call runs two affine scans over all columns."""
    def __init__(self, t, valid, d=16, tau=TAU_G, k2=0.25):
        N = len(t); self.N, self.d = N, d; dt = float(np.median(np.diff(t))); self.dt = dt; K = (N + d - 1) // d; self.K = K; self.pad = K * d - N
        st = np.flatnonzero(valid[1:] & ~valid[:-1]) + 1; en = np.flatnonzero(~valid[1:] & valid[:-1]) + 1; j = np.searchsorted(en, st) - 1
        T = np.where(j >= 0, t[st] - t[en[np.maximum(j, 0)] - 1], 0.0) if len(st) else np.zeros(0)
        self.jblk = st // d; self.jth = np.minimum(0.5 * k2 * T * T, 1.0)
        vv = np.zeros(K * d); vv[:N] = valid; self.vv = vv; self.cnt = vv.reshape(K, d).sum(1); self.okb = self.cnt > 0
        self.q = dt / tau ** 4; self.dtb = d * dt; self.cache = {}
        self.tb = (np.arange(K) * d + (d - 1) / 2.0) * dt; self.tf = np.arange(N) * dt
    def _gains(self, sig_m):
        if sig_m in self.cache: return self.cache[sig_m]
        K = self.K; dtb = self.dtb; q = self.q; q11, q12, q22 = q * dtb ** 3 / 3, q * dtb ** 2 / 2, q * dtb
        jb = np.zeros(K)
        if len(self.jblk): np.add.at(jb, self.jblk, (9.81 * self.jth / sig_m) ** 2)
        jb = jb.tolist(); ok = self.okb.tolist(); cnt = self.cnt.tolist(); g0 = [0.0] * K; g1 = [0.0] * K; PF = [None] * K; PP = [None] * K
        p00, p01, p11 = 1e6, 0.0, 1e2; first = -1
        for i in range(K):
            if first >= 0: p00, p01, p11 = p00 + 2 * dtb * p01 + dtb * dtb * p11 + q11, p01 + dtb * p11 + q12, p11 + q22
            p00 += jb[i]; p11 += 0.01 * jb[i]; PP[i] = (p00, p01, p11)
            if ok[i]:
                if first < 0: first = i
                s = p00 + 1.0 / cnt[i]; k0, k1 = p00 / s, p01 / s; p00, p01, p11 = p00 - k0 * p00, p01 - k0 * p01, p11 - k1 * p01; g0[i] = k0; g1[i] = k1
            PF[i] = (p00, p01, p11)
        C = np.empty((max(K - 1, 0), 4))
        for i in range(K - 1):
            f00, f01, f11 = PF[i]; b00, b01, b10, b11 = f00 + dtb * f01, f01, f01 + dtb * f11, f11; d00, d01, d11 = PP[i + 1]; det = d00 * d11 - d01 * d01
            C[i] = ((b00 * d11 - b01 * d01) / det, (-b00 * d01 + b01 * d00) / det, (b10 * d11 - b11 * d01) / det, (-b10 * d01 + b11 * d00) / det)
        g0 = np.array(g0); g1 = np.array(g1); r = dict(first=first, g0=g0, g1=g1, C=C)
        f = first                                                       # forward maps M_i = (I - g_i e1') A for i > first
        r["Mf"] = (1 - g0[f + 1:], (1 - g0[f + 1:]) * dtb, -g1[f + 1:], 1 - g1[f + 1:] * dtb)
        c00, c01, c10, c11 = C[:, 0], C[:, 1], C[:, 2], C[:, 3]         # backward: s_i = C_i s_{i+1} + (I - C_i A) xf_i
        r["IA"] = (1 - c00, -(c00 * dtb + c01), -c10, 1 - (c10 * dtb + c11))
        self.cache[sig_m] = r; return r
    def block_means(self, x, valid_mask=None):
        """Block means over valid slots of x (N, C)."""
        N, d, K = self.N, self.d, self.K; C = x.shape[1]; xv = np.zeros((K * d, C)); xv[:N] = np.nan_to_num(x) * self.vv[:N, None]
        return xv.reshape(K, d, C).sum(1) / np.maximum(self.cnt, 1)[:, None]
    def smooth_blocks(self, xb, sig_m=1.5):
        """Smoothed level of each column at the block centres (K, C)."""
        r = self._gains(sig_m); K = self.K; f = r["first"]; C = xb.shape[1]
        if f < 0: return np.zeros((K, C))
        F0 = np.zeros((K, C)); F1 = np.zeros((K, C)); F0[f] = xb[f]
        if f < K - 1:
            u0 = r["g0"][f + 1:, None] * xb[f + 1:]; u1 = r["g1"][f + 1:, None] * xb[f + 1:]
            F0[f + 1:], F1[f + 1:] = _scan(*r["Mf"], u0, u1, F0[f], F1[f])
        if K == 1: return F0
        ia = r["IA"]; v0 = ia[0][:, None] * F0[:-1] + ia[1][:, None] * F1[:-1]; v1 = ia[2][:, None] * F0[:-1] + ia[3][:, None] * F1[:-1]
        C_ = r["C"][::-1]; s0, s1 = _scan(C_[:, 0], C_[:, 1], C_[:, 2], C_[:, 3], v0[::-1], v1[::-1], F0[-1], F1[-1])
        return np.vstack([s0[::-1], F0[-1:]])
    def to_slots(self, sb, idx=None):
        tf = self.tf if idx is None else self.tf[idx]; return np.stack([np.interp(tf, self.tb, sb[:, c]) for c in range(sb.shape[1])], 1)
    def __call__(self, x, sig_m=1.5):
        return self.to_slots(self.smooth_blocks(self.block_means(x), sig_m))

# =============================================================================================== M9-M10: attitude passes and bias rule
class Front:
    def __init__(self, t, wu, au, valid):
        self.t, self.wu, self.au, self.valid = t, wu, au, valid; self.fs = 1.0 / np.median(np.diff(t)); self.N = len(t)
        self.af = fill_gaps(t, au, valid); self.wf = bridge_gyro(t, wu, valid); self.gs = GravitySmoother(t, valid, CFG["gs_dec"], TAU_G, CFG["k2"])
        self.seg = np.cumsum(np.concatenate([[0], np.diff(valid.astype(np.int8)) != 0])); self.step = max(1, int(self.fs * 1.0))
        ii = np.where(valid[:-self.step] & valid[self.step:] & (self.seg[:-self.step] == self.seg[self.step:]))[0][::10]; self.ii = ii
    def attitude(self, bg):
        """One pass: strapdown with the bias track, gravity in the gyro-integrated frame, tilt correction."""
        qg = integrate_gyro(self.t, self.wf - bg); gI = self.gs(qrot(qg, self.af)); gn = np.linalg.norm(gI, axis=1, keepdims=True)
        qc = shortest_arc(gI / gn, np.broadcast_to(EZ, gI.shape)); q = qmul(qc, qg); q /= np.linalg.norm(q, axis=1, keepdims=True); return q, qc, gn[:, 0]
    def tilt_drift(self, qc):
        ii, st = self.ii, self.step; t = self.t
        return -(qlog(qmul(qc[ii + st], qinv(qc[ii]))) / (t[ii + st] - t[ii])[:, None])[:, :2]
    def attitude5b(self, bg_track=None, fit="const", ridge=1e-1, win=60.0, knot=20.0, iters=2):
        t = self.t; N = self.N; bg = np.zeros((N, 3)) if bg_track is None else bg_track.copy(); n_it = iters if fit else 0; hmin = 1.0
        for it in range(n_it + 1):
            q, qc, gn = self.attitude(bg)
            if it == n_it: hmin = float(np.percentile(gn[self.valid], 1) / 9.81); break
            ii = self.ii
            if len(ii) < 50: break
            A = rot_rows01(q[ii]); y = self.tilt_drift(qc)
            if fit == "const":
                A2 = A.reshape(-1, 3); bg = bg + np.linalg.solve(A2.T @ A2 + len(A2) * ridge * np.eye(3), A2.T @ y.reshape(-1))
            else:
                tm = t[ii] + 0.5 * self.step / self.fs; knots = np.arange(t[0], t[-1] + knot, knot); db = np.zeros((len(knots), 3))
                for j, tk in enumerate(knots):
                    wg = np.clip(1 - np.abs(tm - tk) / win, 0, None); sel = wg > 0
                    if sel.sum() < 20: continue
                    Aw = A[sel] * wg[sel][:, None, None]; db[j] = np.linalg.solve(np.einsum("nij,nik->jk", Aw, A[sel]) + 2 * wg[sel].sum() * ridge * np.eye(3), np.einsum("nij,ni->j", Aw, y[sel]))
                bg = bg + np.stack([np.interp(t, knots, db[:, i]) for i in range(3)], 1)
        return q, bg, hmin
    @staticmethod
    def _basis(tm, t0, t1, knot):
        if t1 - t0 < 1.5 * knot: s = (tm - 0.5 * (t0 + t1)) / max(t1 - t0, 1e-9); return np.stack([np.ones_like(tm), s], 1)
        K = int(np.ceil((t1 - t0) / knot)) + 1; u = np.clip((tm - t0) / knot, 0, K - 1 - 1e-9); i = np.floor(u).astype(int); fr = u - i
        Phi = np.zeros((len(tm), K)); r = np.arange(len(tm)); Phi[r, i] = 1 - fr; Phi[r, i + 1] = fr; return Phi
    def attitude6(self, bg0, knot=20.0, lam2=3.0, lam0=0.05, iters=6, tol=3e-4):
        """Gyro-bias increments as a linear spline in time, penalised least squares on the drift of the tilt correction with the
        regressors passed through the same gravity filter and 1 s difference as the response."""
        t = self.t; N = self.N; bg = bg0.copy(); done = False; st = self.step
        for it in range(iters + 1):
            q, qc, _ = self.attitude(bg)
            if it == iters or done: break
            ii = self.ii
            if len(ii) < 50: break
            y = self.tilt_drift(qc); tm = t[ii] + 0.5 * st / self.fs; Phi = self._basis(tm, t[0], t[-1], knot); K = Phi.shape[1]; n = len(ii)
            Rf = rot_rows01(q).reshape(N, 6); Rs = self.gs(Rf, sig_m=1.5 / 9.81)
            cs = np.vstack([np.zeros((1, 6)), np.cumsum(Rs, 0)]); A = ((cs[ii + st] - cs[ii]) / st).reshape(n, 2, 3)
            M = np.einsum("nk,nij->nikj", Phi, A).reshape(2 * n, 3 * K); Nm = M.T @ M; rhs = M.T @ y.reshape(-1); wbar = n / K; Pn = lam0 * wbar * np.eye(3 * K)
            if K > 2: D2 = np.diff(np.eye(K), 2, axis=0); Pn = Pn + lam2 * wbar * np.kron(D2.T @ D2, np.eye(3))
            beta = np.linalg.solve(Nm + Pn, rhs).reshape(K, 3); inc = self._basis(t, t[0], t[-1], knot) @ beta; bg = bg + inc; done = bool(np.abs(inc).max() < tol)
        return q, bg

def _mahony(w, a, valid, dt, kp, ki, b0):
    """Causal PI complementary filter with integral (bias) feedback; returns the bias state at every sample."""
    N = len(w); qx, qy, qz, qs = 0.0, 0.0, 0.0, 1.0; b = [float(b0[0]), float(b0[1]), float(b0[2])]; B = np.empty((N, 3)); init = False
    W = w.tolist(); A = a.tolist(); V = valid.tolist(); e0 = e1 = e2 = 0.0
    for i in range(N):
        if V[i]:
            a0, a1, a2 = A[i]; an = math.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
            if not init and an > 1e-6:
                u0, u1, u2 = a0 / an, a1 / an, a2 / an; cx, cy, cz, dd = u1, -u0, 0.0, u2; nq = math.sqrt(cx * cx + cy * cy + (1 + dd) ** 2)
                if nq > 1e-9: qx, qy, qz, qs = cx / nq, cy / nq, cz / nq, (1 + dd) / nq
                init = True
            if init and 4.0 < an < 16.0:
                x, y, z, s = qx, qy, qz, qs; v0 = 2 * (x * z - s * y); v1 = 2 * (y * z + s * x); v2 = 1 - 2 * (x * x + y * y)
                e0 = (a1 * v2 - a2 * v1) / an; e1 = (a2 * v0 - a0 * v2) / an; e2 = (a0 * v1 - a1 * v0) / an
                b[0] -= ki * e0 * dt; b[1] -= ki * e1 * dt; b[2] -= ki * e2 * dt
            else: e0 = e1 = e2 = 0.0
            w0, w1, w2 = W[i]; r0 = (w0 - b[0] + kp * e0) * dt; r1 = (w1 - b[1] + kp * e1) * dt; r2 = (w2 - b[2] + kp * e2) * dt
            th = math.sqrt(r0 * r0 + r1 * r1 + r2 * r2); k = math.sin(th / 2) / th if th > 1e-9 else 0.5; d0, d1, d2, d3 = r0 * k, r1 * k, r2 * k, math.cos(th / 2)
            x, y, z, s = qx, qy, qz, qs
            qx = s * d0 + x * d3 + y * d2 - z * d1; qy = s * d1 - x * d2 + y * d3 + z * d0; qz = s * d2 + x * d1 - y * d0 + z * d3; qs = s * d3 - x * d0 - y * d1 - z * d2
            n = math.sqrt(qx * qx + qy * qy + qz * qz + qs * qs); qx, qy, qz, qs = qx / n, qy / n, qz / n, qs / n
        B[i] = b
    return B
def coarse_bias(t, w, a, valid, kp=1.0, ki=0.3, smooth=5.0):
    """Average of a causal and an anti-causal PI complementary filter's bias states, low-passed at `smooth` s."""
    dt = float(np.median(np.diff(t))); wf = np.nan_to_num(w); af = np.nan_to_num(a); z = np.zeros(3)
    bf = _mahony(wf, af, valid, dt, kp, ki, z); bb = _mahony(-wf[::-1].copy(), af[::-1].copy(), valid[::-1].copy(), dt, kp, ki, -z)[::-1]
    return lowpass(0.5 * (bf - bb), 1.0 / dt, 1.0 / (2 * np.pi * smooth))

# =============================================================================================== M11-M17: back end
def flight_mask(t, a, frac=0.5):
    fs = 1.0 / np.median(np.diff(t)); e = np.sum(np.diff(a, axis=0, prepend=a[:1]) ** 2, 1); vib = np.sqrt(lowpass(e, fs, 1.0).clip(0)); return vib > frac * np.median(vib)
def thrust_frame(a):
    zb = a.mean(0); zb = zb / np.linalg.norm(zb); h = np.array([0, 1.0, 0]) if abs(zb[1]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(zb, h); e1 /= np.linalg.norm(e1); return zb, e1, np.cross(zb, e1)
def _bsum(x, K, d):
    """Block sums of x (N, ...) over blocks of d rows (zero padded)."""
    N = x.shape[0]; pad = K * d - N
    if pad: x = np.concatenate([x, np.zeros((pad,) + x.shape[1:], x.dtype)])
    return x.reshape((K, d) + x.shape[1:]).sum(1)
def hermite(tk, pk, vk, t):
    i = np.clip(np.searchsorted(tk, t) - 1, 0, len(tk) - 2); h = (tk[i + 1] - tk[i])[:, None]; s = ((t - tk[i]) / (tk[i + 1] - tk[i]))[:, None]
    s2, s3 = s * s, s * s * s; out = (2 * s3 - 3 * s2 + 1) * pk[i] + (s3 - 2 * s2 + s) * h * vk[i] + (-2 * s3 + 3 * s2) * pk[i + 1] + (s3 - s2) * h * vk[i + 1]
    lo = t < tk[0]; hi = t > tk[-1]; out[lo] = pk[0] + (t[lo] - tk[0])[:, None] * vk[0]; out[hi] = pk[-1] + (t[hi] - tk[-1])[:, None] * vk[-1]; return out
def _rts(K, x0, P0, fwd, take, Anext, out):
    """Kalman filter + RTS smoother with sqrt(K) checkpoints: the forward pass keeps one (x, P) per chunk; each chunk is re-run
    forward and smoothed backward (xs_k = xf_k + Pf_k A' Pp_{k+1}^-1 (xs_{k+1} - xp_{k+1}))."""
    ck = max(16, int(math.sqrt(K))); cps = []; x, P = x0, P0
    for k in range(K):
        if k % ck == 0: cps.append((x, P))
        _, _, x, P = fwd(x, P, k)
    xs = None; nxt = None
    for c in range(len(cps) - 1, -1, -1):
        c0 = c * ck; c1 = min(K, c0 + ck); x, P = cps[c]; XP = []; PP = []; XF = []; PF = []
        for k in range(c0, c1):
            xp, Pp, x, P = fwd(x, P, k); XP.append(xp); PP.append(Pp); XF.append(x); PF.append(P)
        for j in range(c1 - c0 - 1, -1, -1):
            k = c0 + j
            if k == K - 1: xs = XF[j]
            else:
                xpn, Ppn = (XP[j + 1], PP[j + 1]) if j < c1 - c0 - 1 else nxt
                xs = XF[j] + PF[j] @ (Anext(k + 1).T @ np.linalg.solve(Ppn, xs - xpn))
            out[k] = take(xs)
        nxt = (XP[0], PP[0])
    return out
def _binf(t, valid, gap_ref):
    """Accelerometer-bias variance injected at the end of every gap: 0.5^2 min(1, T / 2 s)."""
    binf = np.zeros(len(t)); st = np.flatnonzero(valid[1:] & ~valid[:-1]) + 1; en = np.flatnonzero(~valid[1:] & valid[:-1]) + 1
    if len(st):
        j = np.searchsorted(en, st) - 1; T = np.where(j >= 0, t[st] - t[en[np.maximum(j, 0)] - 1], gap_ref); binf[st] = BIAS_JUMP ** 2 * np.minimum(1.0, T / gap_ref)
    return binf
def _grid(t, valid, d):
    N = len(t); K = (N + d - 1) // d; pad = K * d - N
    tt = np.concatenate([t, t[-1] + (t[-1] - t[-2]) * np.arange(1, pad + 1)]) if pad else t
    return K, tt.reshape(K, d).mean(1)
def smooth_hand(t, aw, valid, zmask, binf, d):
    """Matern-5/2 prior per world axis (the three axes share one 4 x 4 covariance), accelerometer + zero-velocity rows, block
    means of d slots. Returns tk, p_k (K, 3), v_k (K, 3)."""
    F4, Qc4, P04, H4 = model_matern52(*MATERN); K, tk = _grid(t, valid, d); As, Qs, ix = discretise(F4, Qc4, tk)
    sn2 = hf_std(aw[valid]) ** 2; ca = _bsum(valid.astype(float), K, d); ya = _bsum(aw * valid[:, None], K, d) / np.maximum(ca, 1)[:, None]
    Ra = _bsum(np.where(valid, sn2, 0.0), K, d) / np.maximum(ca, 1) / np.maximum(ca, 1); cz = _bsum(zmask.astype(float), K, d)
    Rz = _bsum(np.where(zmask, SZ ** 2, 0.0), K, d) / np.maximum(cz, 1) / np.maximum(cz, 1); bk = np.zeros(K * d); bk[:len(binf)] = binf; bk = bk.reshape(K, d).max(1)
    okA = (ca > 0).tolist(); okZ = (cz > 0).tolist(); RaL = Ra.tolist(); RzL = Rz.tolist(); bL = bk.tolist(); ixL = ix.tolist()
    h = H4; hz = np.array([0, 1.0, 0, 0]); Hb = np.vstack([h, hz]); z3 = np.zeros(3)
    def fwd(x, P, k):
        if k:
            A = As[ixL[k]]; xp = A @ x; Pp = A @ P @ A.T + Qs[ixL[k]]
            if bL[k] > 0: Pp[3, 3] += bL[k]
        else: xp, Pp = x, P
        a_, z_ = okA[k], okZ[k]
        if not (a_ or z_): return xp, Pp, xp, 0.5 * (Pp + Pp.T)
        if a_ and not z_:
            PH = Pp @ h; g = PH / (h @ PH + RaL[k]); xn = xp + np.outer(g, ya[k] - h @ xp); Pn = Pp - np.outer(g, PH)
        elif z_ and not a_:
            PH = Pp @ hz; g = PH / (hz @ PH + RzL[k]); xn = xp - np.outer(g, hz @ xp); Pn = Pp - np.outer(g, PH)
        else:
            PH = Pp @ Hb.T; S = Hb @ PH; S[0, 0] += RaL[k]; S[1, 1] += RzL[k]; Gt = np.linalg.solve(S, PH.T); Pn = Pp - PH @ Gt
            xn = xp + Gt.T @ (np.vstack([ya[k], z3]) - Hb @ xp)
        return xp, Pp, xn, 0.5 * (Pn + Pn.T)
    out = _rts(K, np.zeros((4, 3)), P04.copy(), fwd, lambda xs: xs[:2], lambda k: As[ixL[k]], np.empty((K, 2, 3)))
    return tk, out[:, 0], out[:, 1]
def smooth_rotor(t, q, af, aw, wl, valid, zmask, binf, d, k_drag):
    """17-state rotor model: spring-damper prior + OU bias per world axis, 2 OU drag biases, 3 lever-arm states; accelerometer,
    rotor-drag (in flight) and zero-velocity rows, block means of d slots. Returns tk, p_k, v_k."""
    F4, Qc4, P04, H4 = model_spring(*ROTOR); n = 17; F = np.zeros((n, n)); Qc = np.zeros((n, n)); P0 = np.zeros((n, n))
    for j in range(3):
        sl = slice(4 * j, 4 * j + 4); F[sl, sl] = F4; Qc[sl, sl] = Qc4; P0[sl, sl] = P04
    for j in (12, 13): F[j, j] = -1.0 / TAU_C; Qc[j, j] = 2 * SC ** 2 / TAU_C; P0[j, j] = SC ** 2
    for j in (14, 15, 16): P0[j, j] = LEVER_PRIOR ** 2
    K, tk = _grid(t, valid, d); As, Qs, ix = discretise(F, Qc, tk); N = len(t); fs = 1.0 / np.median(np.diff(t))
    sn2 = hf_std(aw[valid]) ** 2; ca = _bsum(valid.astype(float), K, d); c1a = np.maximum(ca, 1)
    ya = _bsum(aw * valid[:, None], K, d) / c1a[:, None]; Ra = _bsum(np.where(valid, sn2, 0.0), K, d) / c1a / c1a
    zb, e1, e2 = thrust_frame(af[valid]); fl = flight_mask(t, af); md = valid & fl; cd = _bsum(md.astype(float), K, d); c1d = np.maximum(cd, 1)
    wd = np.gradient(wl, axis=0) * fs; L = [np.cross(wd, u) + np.cross(wl, np.cross(wl, u)) for u in np.eye(3)]
    HV = np.zeros((K, 2, 3)); HL = np.zeros((K, 2, 3)); YD = np.zeros((K, 2))
    for i, e in enumerate((e1, e2)):
        ew = qrot(q, np.broadcast_to(e, (N, 3))); HV[:, i] = _bsum(-k_drag * ew * md[:, None], K, d) / c1d[:, None]
        HL[:, i] = np.stack([_bsum((L[j] @ e) * md, K, d) for j in range(3)], 1) / c1d[:, None]; YD[:, i] = _bsum((af @ e) * md, K, d) / c1d
    Rd = _bsum(np.where(md, SD_DRAG ** 2, 0.0), K, d) / c1d / c1d; cz = _bsum(zmask.astype(float), K, d); Rz = _bsum(np.where(zmask, SZ ** 2, 0.0), K, d) / np.maximum(cz, 1) / np.maximum(cz, 1)
    bk = np.zeros(K * d); bk[:len(binf)] = binf; bk = bk.reshape(K, d).max(1)
    okA = (ca > 0).tolist(); okD = (cd > 0).tolist(); okZ = (cz > 0).tolist(); bL = bk.tolist(); ixL = ix.tolist()
    H = np.zeros((8, n))
    for j in range(3): H[j, 4 * j:4 * j + 4] = H4; H[5 + j, 4 * j + 1] = 1.0
    H[3, 12] = 1.0; H[4, 13] = 1.0; rows3 = (np.array([3, 3, 3]), np.array([1, 5, 9])); rows4 = (np.array([4, 4, 4]), np.array([1, 5, 9]))
    def fwd(x, P, k):
        if k:
            A = As[ixL[k]]; xp = A @ x; Pp = A @ P @ A.T + Qs[ixL[k]]
            if bL[k] > 0: Pp[3, 3] += bL[k]; Pp[7, 7] += bL[k]; Pp[11, 11] += bL[k]
        else: xp, Pp = x, P
        if not (okA[k] or okD[k] or okZ[k]): return xp, Pp, xp, 0.5 * (Pp + Pp.T)
        m = 8 if okZ[k] else 5; Hk = H[:m].copy(); z = np.zeros(m); r = np.ones(m)
        if okA[k]: z[:3] = ya[k]; r[:3] = Ra[k]
        else: Hk[:3] = 0.0
        if okD[k]:
            Hk[rows3] = HV[k, 0]; Hk[rows4] = HV[k, 1]; Hk[3, 14:] = HL[k, 0]; Hk[4, 14:] = HL[k, 1]; z[3:5] = YD[k]; r[3:5] = Rd[k]
        else: Hk[3:5] = 0.0
        if m == 8: r[5:] = Rz[k]
        PH = Pp @ Hk.T; S = Hk @ PH; S.flat[::m + 1] += r; Gt = np.linalg.solve(S, PH.T); Pn = Pp - PH @ Gt; xn = xp + (z - Hk @ xp) @ Gt
        return xp, Pp, xn, 0.5 * (Pn + Pn.T)
    out = _rts(K, np.zeros(n), P0, fwd, lambda xs: xs[[0, 4, 8, 1, 5, 9]], lambda k: As[ixL[k]], np.empty((K, 6)))
    return tk, out[:, :3], out[:, 3:]
def radius(p): return float(np.sqrt(np.mean(np.sum((p - p.mean(0)) ** 2, 1))))

# =============================================================================================== end to end
def estimate(tn, X, k_drag=K_DRAG, force=None):
    """v5 / V5P4 on one stream (rows in input order, X = (gyro, accel)). Returns p (rows, 3), q (rows, 4), info."""
    t, wu, au, valid, rows = ingest2(tn, X); N = len(t); L = int(np.flatnonzero(valid)[-1]); nt = N - 1 - L
    t0, wu0, au0, v0 = t[:L + 1], wu[:L + 1], au[:L + 1], valid[:L + 1]
    wu0, au0, v0, ri = repair(t0, wu0, au0, v0)
    af0 = fill_gaps(t0, au0, v0); lock = thrust_lock_fill(t0, af0, v0); anchors, zmask = rest_info(t0, wu0, au0, v0)
    rotor = lock < CFG["gate"] if force is None else force == "rotor"
    fe = Front(t0, wu0, au0, v0); base = bias_track(t0, anchors)
    if base is None: q3, b3, h = fe.attitude5b(fit="const"); base = b3
    else: q3, _b, h = fe.attitude5b(bg_track=base, fit=None)
    D = float("nan"); coarse = False
    def spline():
        if h < CFG["health"]:
            bc = coarse_bias(t0, fe.wf, fe.af, v0); return fe.attitude6(bc, knot=CFG["knot"], lam2=CFG["lam2"], iters=10), True
        return fe.attitude6(base, knot=CFG["knot"], lam2=CFG["lam2"]), False
    if rotor: (q, bg), coarse = spline(); mode = "spline"
    else:
        if h < CFG["health"]: D = float("inf")
        else:
            _q4, b4, _ = fe.attitude5b(bg_track=base, fit="knots", win=60.0, knot=20.0); dd = b4 - base; D = float(np.sqrt(((dd - dd.mean(0)) ** 2).sum(1).mean()))
        if D > CFG["thr_hand"]: (q, bg), coarse = spline(); mode = "spline"
        else: q, bg, mode = q3, base, "const"
    # back-end inputs (prep_from), extended over the blind window (estimate_v6)
    fs = fe.fs; dtn = 1.0 / fs; af = fe.af; wc = fe.wf - bg; wl = lowpass(wc, fs, 15.0); aw = qrot(q, af); aw = aw - aw[v0].mean(0)
    ext = lambda x: np.concatenate([x, np.repeat(x[-1:], nt, 0)]); te = np.concatenate([t0, t0[-1] + dtn * np.arange(1, nt + 1)])
    ve = np.concatenate([v0, np.zeros(nt, bool)]); ze = np.concatenate([zmask, np.zeros(nt, bool)]); qe, afe, wle = ext(q), ext(af), ext(wl)
    awe = qrot(qe, afe); awe = awe - awe[ve].mean(0); binf = _binf(te, ve, CFG["gap_ref"])
    guard = False
    if rotor:
        tk, pk, vk = smooth_rotor(te, qe, afe, awe, wle, ve, ze, binf, CFG["dec_rotor"], k_drag); p_r = hermite(tk, pk, vk, te)
        tkh, pkh, vkh = smooth_hand(te, awe, ve, ze, binf, CFG["dec_rotor"]); p_h = hermite(tkh, pkh, vkh, te)
        if radius(p_r) > CFG["guard"] * max(radius(p_h), 1e-6): guard = True
        else: p = p_r
    if not rotor or guard:
        tk, pk, vk = smooth_hand(te, awe, ve, ze, binf, CFG["dec_hand"]); p = hermite(tk, pk, vk, te)
    is_rotor = rotor and not guard; N0 = L + 1
    if TRACE is not None: TRACE.update(t=t0, valid=v0, qa=q, bg=bg, zmask=zmask, wl=wl, aw=aw, af=af, tk=tk, pk=pk, vk=vk, p_slots=p, rows=rows)
    qq = qe.copy()
    if nt > 0:
        vel = np.stack([np.interp(te, tk, vk[:, j]) for j in range(3)], 1); tau = np.arange(1, nt + 1) / fs; pT = p[N0 - 1]; vT = vel[N0 - 1]
        dp = (0.5 * (1.0 - np.exp(-tau / 0.5)))[:, None] * vT if is_rotor else p[N0:N0 + nt] - pT
        p = p.copy(); p[N0:] = p[N0 - 1] + dp; qq[N0:] = q[-1]
    info = dict(rotor_gate=bool(rotor), rotor=bool(is_rotor), guard=guard, lock=lock, mode=mode, D=D, h=h, coarse=coarse, n_tail=nt, L=L,
                implausible=ri["implausible"], spikes_legacy=ri["spikes"], slots=N)
    return p[rows], qq[rows], info

def motionless(tn, X):
    ok = np.isfinite(X).all(1) & ~(X == 0).all(1); q = np.array([0, 0, 0, 1.0])
    if ok.any():
        u = X[ok, 3:].mean(0); nu = np.linalg.norm(u)
        if nu > 1e-6: q = shortest_arc(u[None] / nu, EZ[None])[0]
    return np.zeros((len(tn), 3)), np.repeat(q[None], len(tn), 0)

def estimate_safe(tn, X, k_drag=K_DRAG):
    """Never fails: v5 -> hand-held branch only -> motionless. Non-increasing timestamps: estimate on a stably sorted copy."""
    n = len(tn)
    if n == 0: return np.zeros((0, 3)), np.zeros((0, 4)), dict(fallback="empty")
    order = None
    if n > 1 and (np.diff(tn) < 0).any():
        order = np.argsort(tn, kind="stable"); tn, X = tn[order], X[order]; log("timestamps not monotonic: estimated on a sorted copy")
    res = None
    for force in (None, "hand"):
        try:
            ok = np.isfinite(X).all(1) & ~(X == 0).all(1)
            if ok.sum() < 10 or len(np.unique(tn)) < 10: break
            p, q, info = estimate(tn, X, k_drag, force)
            if np.isfinite(p).all() and np.isfinite(q).all(): info["fallback"] = force or ""; res = (p, q, info); break
            log("non-finite output (%s branch)" % (force or "v5"))
        except Exception as ex:                                         # noqa: BLE001 - the output file must always be written
            log("estimation failed (%s branch): %r" % (force or "v5", ex))
    if res is None: p, q = motionless(tn, X); res = (p, q, dict(fallback="motionless"))
    p, q, info = res; q = q / np.linalg.norm(q, axis=1, keepdims=True)
    if order is not None:
        inv = np.empty_like(order); inv[order] = np.arange(n); p, q = p[inv], q[inv]
    return p, q, info

def resolve_output(arg):
    if arg is None: return os.path.join(os.getcwd(), "estimated_trajectory.txt")
    if os.path.isdir(arg) or arg.endswith(("/", os.sep)): os.makedirs(arg, exist_ok=True); return os.path.join(arg, "estimated_trajectory.txt")
    d = os.path.dirname(os.path.abspath(arg)); os.makedirs(d, exist_ok=True); return arg
def run(inp=None, out=None, k_drag=K_DRAG):
    with open_input(inp) as s: tn, X = read_imu(s)
    tn, X = unit_guard(tn, X); p, q, info = estimate_safe(tn, X, k_drag); path = resolve_output(out)
    write_trajectory(path, tn, p, q); return path, info
def main(argv):
    inp = out = None; pos = []; it = iter(argv)
    for a in it:
        if a in ("-i", "--input"): inp = next(it, None)
        elif a in ("-o", "--output"): out = next(it, None)
        elif a in ("-h", "--help"): print(__doc__); return 0
        elif a == "--version": print(__version__); return 0
        else: pos.append(a)
    if pos and inp is None: inp = pos.pop(0)
    if pos and out is None: out = pos.pop(0)
    try:
        path, info = run(inp, out)
    except Exception as ex:                                             # unreadable input: still write a valid (empty) file
        log("could not read the input: %r" % (ex,)); path = resolve_output(out); open(path, "wb").close(); info = dict(fallback="no input")
    log("rows written, platform %s%s, mode %s, lock %.4f%s" % ("rotor" if info.get("rotor") else "hand-held", " (guard)" if info.get("guard") else "",
        info.get("mode", "-"), info.get("lock", float("nan")), (", fallback " + info["fallback"]) if info.get("fallback") else ""))
    print(path); return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
