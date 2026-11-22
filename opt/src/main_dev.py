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
rotor = damped velocity (0.5 s), hand-held = the smoother's own prediction, orientation held.

Memory: the program streams. Raw samples are kept as int32 multiples of 1e-6 (lossless for the official 6-decimal format) on
the slot grid; every full-rate quantity (gap-filled and bridged signals, strapdown attitude, rotated specific force, filtered
gyro) is regenerated chunk by chunk when needed; smoothers keep sqrt(K) checkpoints; drag rows are stored for flight blocks
only; on long inputs the block times are recomputed on demand and the smoother re-derives its predictions (same values); the
input is read again to write the output, so no per-row array outlives the parse (except the row -> slot map when the input's
timestamps are not monotonic)."""
import os, sys
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"          # single CPU thread, set before NumPy loads its BLAS
import math, zlib
import numpy as np
import numpy.fft                                  # loaded with NumPy, not lazily inside the estimation

__version__ = "1.0.0"
G = 9.81
CFG = dict(k2=0.25, gate=0.035, thr_hand=2.0e-3, health=0.9, lam2=3.0, knot=20.0, gap_ref=2.0, dec_rotor=8, dec_hand=2, guard=12.0, gs_dec=16,
           rest_win=0.5, rest_frac=0.05, rest_gate=0.035, rest_minw=3,    # Phase 5 development flags (defaults = v1.0.0)
           acc_comp=0, acc_comp_hand=1, acc_comp_rotor=0, acc_comp_gain=1.0, drag_off_rest=0, tau_g_rotor=None, tau_g_hand=None,
           drag_adapt_r=None, drag_tight=(0.05, 300.0, 0.15),     # X043: drag-bias prior (sc, tauc, sc0) used when the guard radius >= drag_adapt_r
           het_acc=0, het_lo=0.2, het_hi=5.0, blind_tau_r=0.5,    # X005: local accelerometer noise per chunk; X030: rotor blind damping time
           drag_switch_r=None,                                     # X044: re-smooth with drag_tight when the rotor solution's radius >= this (m)
           tau_g_final_hand=None, final_const_only=1,              # X045: hand-held final attitude re-computed with this gravity time constant (bias and mode from the 3 s front end)
           final_mu_max=None, final_sw_max=None,                   # X046: ... only if the residual constant bias |mu| [rad/s] and the gyro sample noise [rad/s] are below these
           vel_aid_dir=None, vel_aid_scale=1.0, vel_aid_floor=0.05,  # X060 (research only): learned body-velocity predictions fused as velocity rows
           guard_ref="final", switch_blend=None, switch_ratio=None)  # X110 guard on the loose radius ("loose"); X111 soft switch [lo, hi] m; X112 switch only if rr0 >= ratio * rh
K_DRAG = 0.1873                                   # rotor-drag coefficient [1/s]: mean of 11 per-flight fits (Phase 2)
ROTOR = (20.0, 2.0, 0.3, 1.0, 20.0, 0.1)          # spring-damper prior: T_p [s], tau_v [s], tau_a [s], sigma_a, tau_b [s], sigma_b
MATERN = (1.0, 2.0, 20.0, 0.1)                    # Matern-5/2 prior: ell [s], sigma_p [m], tau_b [s], sigma_b
SD_DRAG, SZ, BIAS_JUMP, TAU_C, SC, LEVER_PRIOR, TAU_G = 1.4, 0.02, 0.5, 100.0, 0.15, 0.15, 3.0
SC0 = None                                        # Phase 5: initial drag-bias prior SD (None = SC, the stationary SD)
EZ = np.array([0.0, 0.0, 1.0])
CH = 2048                                         # slots per chunk of every streaming pass (a multiple of all block sizes)
SCALE = 1e6                                       # raw values are int32 multiples of 1 / SCALE
SENT = -2 ** 31                                   # raw-value sentinel: missing slot, NaN / inf or |value| >= 2147
TRACE = None                                      # tests set a dict here to receive intermediate results
PROF = None                                       # tools/measure.py sets a list here to receive per-stage time and heap

def log(msg):
    sys.stderr.write("[main] %s\n" % msg)
def _mark(name):
    if PROF is None: return
    import time, tracemalloc
    tr = tracemalloc.is_tracing(); PROF.append((name, time.perf_counter(), tracemalloc.get_traced_memory()[1] if tr else 0))
    if tr: tracemalloc.reset_peak()

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
    """Inclusive prefix product (Hillis-Steele, renormalised at every level)."""
    out = dq.copy(); s = 1
    while s < len(out):
        out[s:] = qmul(out[:-s], out[s:]); out /= np.linalg.norm(out, axis=1, keepdims=True); s *= 2
    return out
def shortest_arc(u, v):
    c = np.cross(u, v); d = np.sum(u * v, -1, keepdims=True); q = np.concatenate([c, 1.0 + d], -1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)
def rot_rows01(q):
    """Rows 0 and 1 of the rotation matrix of q (scalar last), normalised first; the formulas of scipy's Rotation.as_matrix."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True); x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w; xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    return np.stack([x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw), 2 * (xy + zw), -x2 + y2 - z2 + w2, 2 * (yz - xw)], 1).reshape(-1, 2, 3)
def _median(x, axis=None, inplace=False):
    """= np.median for NaN-free input (np.median would import numpy.ma, ~1.9 MB of heap, on its first call); inplace=True
    partitions a 1-D scratch array in place instead of copying it."""
    x = np.asarray(x)
    if axis is None: x = x.ravel(); axis = 0
    n = x.shape[axis]; k = n // 2
    if inplace and x.ndim == 1:
        x.partition([k - 1, k] if n % 2 == 0 and n > 1 else k); return x[k] if n % 2 else (x[k - 1] + x[k]) / 2
    if n % 2: return np.take(np.partition(x, k, axis=axis), k, axis=axis)
    p = np.partition(x, [k - 1, k], axis=axis); return (np.take(p, k - 1, axis=axis) + np.take(p, k, axis=axis)) / 2
def _percentile(x, q, inplace=False):
    """= np.percentile(x, q) (method 'linear') for a NaN-free 1-D array, without np.percentile's import of numpy.ma."""
    x = np.asarray(x, float); x = x if inplace else x.copy(); n = x.size; qq = np.true_divide(q, 100); v = (n - 1) * qq; lo = int(np.floor(v)); hi = lo + 1
    if v >= n - 1: lo = hi = -1
    if v < 0: lo = hi = 0
    x.partition(sorted(set([0, n - 1, lo % n, hi % n]))); a = x[lo]; b = x[hi]; g = v - np.floor(v); d = b - a
    return (b - d * (1 - g)) if g >= 0.5 else (a + d * g)
def _med5(a, b, c, d, e):
    """Elementwise median of five arrays (= np.median of the five, which is the middle value)."""
    L = np.maximum(np.minimum(a, b), np.minimum(c, d)); H = np.minimum(np.maximum(a, b), np.maximum(c, d))
    return np.maximum(np.minimum(e, L), np.minimum(np.maximum(e, L), H))
def _percentile_small(small, n, q):
    """np.percentile(x, q) (linear) of an n-element array x from its len(small) smallest values (enough for the two order
    statistics the interpolation uses)."""
    v = (n - 1) * np.true_divide(q, 100); lo = int(np.floor(v)); hi = lo + 1
    if v >= n - 1: lo = hi = n - 1
    if v < 0: lo = hi = 0
    x = np.sort(small); a = x[lo]; b = x[hi]; g = v - np.floor(v); d = b - a
    return (b - d * (1 - g)) if g >= 0.5 else (a + d * g)
def _uniq(a):
    """= np.unique for a 1-D array (np.unique imports numpy.ma)."""
    s = np.sort(np.asarray(a).ravel())
    return s[np.concatenate([[True], s[1:] != s[:-1]])] if len(s) else s
def _seqsum(acc, rows):
    """acc + rows[0] + rows[1] + ... in that order (= the order of an axis-0 NumPy sum over the whole array)."""
    return np.concatenate([acc[None], rows], 0).sum(0)

# =============================================================================================== zero-phase Butterworth low-pass
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
_CO = {}
def _coeffs(fc, fs):
    """Biquad, steady-state initial state, and (if its impulse response decays to 1e-18 within 4096 samples) the truncated
    impulse response with its spectrum for overlap-save filtering."""
    key = (float(fc), float(fs))
    if key in _CO: return _CO[key]
    b0, b1, b2, a1, a2 = _butter2(fc, fs)
    zi = np.linalg.solve(np.array([[1 + a1, -1.0], [a2, 1.0]]), np.array([b1 - a1 * b0, b2 - a2 * b0]))
    rho = math.sqrt(abs(a2)) if a1 * a1 < 4 * a2 else max(abs((-a1 + s * math.sqrt(a1 * a1 - 4 * a2)) / 2) for s in (-1, 1))
    L = int(math.ceil(math.log(1e-18) / math.log(rho))) + 16 if 0 < rho < 1 else None
    co = dict(b=(b0, b1, b2, a1, a2), zi=(float(zi[0]), float(zi[1])), L=L)
    if L is not None and L <= 4096:
        u0, u1 = b1 - a1 * b0, b2 - a2 * b0; h = np.concatenate([[b0], _homog(a1, a2, u0, u1, L - 1)]); nf = 4096 if L <= 1024 else 8192
        co.update(h=h, c1=_homog(a1, a2, 1.0, 0.0, L), c2=_homog(a1, a2, 0.0, 1.0, L), nf=nf, H=np.fft.rfft(h, nf))
    _CO[key] = co; return co
def _filt(x, co, z1, z2):
    """Causal biquad over the 1-D float64 array x IN PLACE (x may be a reversed view), initial DF2T state (z1, z2)."""
    n = len(x)
    if "h" in co:                                                       # overlap-save with the truncated impulse response
        h = co["h"]; L = len(h); nf = co["nf"]; H = co["H"]; B = nf - L + 1; hist = np.zeros(L - 1); c1, c2 = co["c1"], co["c2"]
        for s in range(0, n, B):
            e = min(n, s + B); seg = np.concatenate([hist, x[s:e]]); y = np.fft.irfft(np.fft.rfft(seg, nf) * H, nf)[L - 1:L - 1 + e - s]
            if s < L:
                k = min(e - s, L - s); y[:k] += c1[s:s + k] * z1 + c2[s:s + k] * z2
            hist = seg[len(seg) - (L - 1):]; x[s:e] = y
        return
    b0, b1, b2, a1, a2 = co["b"]                                       # slowly decaying filters: the recursion itself
    for s in range(0, n, 4096):
        v = x[s:s + 4096].tolist()
        for i, xi in enumerate(v):
            y = b0 * xi + z1; z1 = b1 * xi - a1 * y + z2; z2 = b2 * xi - a2 * y; v[i] = y
        x[s:s + 4096] = v
def zerophase(x, fs, fc):
    """= scipy.signal.sosfiltfilt(butter(2, fc/(fs/2), output='sos'), x) for a 1-D signal (odd extension of 9 samples,
    steady-state initial conditions scaled by the first sample of each pass). Returns a new array."""
    x = np.asarray(x, float); n = len(x); co = _coeffs(fc, fs); edge = min(9, n - 1); ext = np.empty(n + 2 * edge); ext[edge:edge + n] = x
    if edge: ext[:edge] = 2 * x[0] - x[edge:0:-1]; ext[edge + n:] = 2 * x[-1] - x[-2:-edge - 2:-1]
    z = co["zi"]; _filt(ext, co, z[0] * ext[0], z[1] * ext[0]); r = ext[::-1]; _filt(r, co, z[0] * r[0], z[1] * r[0])
    return ext[edge:edge + n]
def zerophase_inplace(buf, n, fs, fc):
    """zerophase of the n samples in buf[e:e + n] (e = min(9, n - 1); buf has 2e spare samples for the extensions), in place."""
    co = _coeffs(fc, fs); edge = min(9, n - 1); x = buf[edge:edge + n]
    if edge: buf[:edge] = 2 * x[0] - x[edge:0:-1]; buf[edge + n:] = 2 * x[-1] - x[-2:-edge - 2:-1]
    z = co["zi"]; _filt(buf, co, z[0] * buf[0], z[1] * buf[0]); r = buf[::-1]; _filt(r, co, z[0] * r[0], z[1] * r[0]); return x
def zp_range(get, n, fs, fc, a, b):
    """Rows a..b-1 of the zero-phase low-pass of the n-row signal get(0, n), computed from rows a-M .. b+M only (M = the
    impulse-response length to 1e-18; at a true end the exact end treatment of zerophase is used)."""
    co = _coeffs(fc, fs); M = co["L"] if co["L"] is not None else n; w0 = max(0, a - M); w1 = min(n, b + M); X = get(w0, w1)
    out = np.empty((b - a, X.shape[1])); edge = min(9, n - 1); z = co["zi"]
    for j in range(X.shape[1]):
        x = X[:, j]; le = edge if w0 == 0 else 0; re = edge if w1 == n else 0; m = len(x); ext = np.empty(m + le + re); ext[le:le + m] = x
        if le: ext[:le] = 2 * x[0] - x[le:0:-1]
        if re: ext[le + m:] = 2 * x[-1] - x[-2:-re - 2:-1]
        _filt(ext, co, z[0] * ext[0], z[1] * ext[0]); r = ext[::-1]; _filt(r, co, z[0] * r[0], z[1] * r[0])
        out[:, j] = ext[le + a - w0:le + b - w0]
    return out

# =============================================================================================== small numerics and models
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
def model_matern52(ell, sp, tau_b, sb):
    lam = math.sqrt(5.0) / ell; F = np.zeros((4, 4)); F[0, 1] = 1; F[1, 2] = 1; F[2, :3] = [-lam ** 3, -3 * lam ** 2, -3 * lam]; F[3, 3] = -1.0 / tau_b
    Qc = np.zeros((4, 4)); Qc[2, 2] = 16.0 / 3.0 * sp ** 2 * lam ** 5; Qc[3, 3] = 2 * sb ** 2 / tau_b
    return F, Qc, lyap(F, Qc), np.array([0, 0, 1.0, 1.0])
def model_spring(T_p, tau_v, tau_a, sa, tau_b, sb):
    w0 = 2 * math.pi / T_p; c = 1.0 / tau_v; F = np.zeros((4, 4)); F[0, 1] = 1; F[1, 0] = -w0 ** 2; F[1, 1] = -c; F[1, 2] = 1
    F[2, 2] = -1.0 / tau_a; F[3, 3] = -1.0 / tau_b; Qc = np.zeros((4, 4)); Qc[2, 2] = 2 * sa ** 2 / tau_a; Qc[3, 3] = 2 * sb ** 2 / tau_b
    return F, Qc, lyap(F, Qc), np.array([-w0 ** 2, -c, 1.0, 1.0])
def model_rotor(sc=None, tauc=None, sc0=None):
    sc = SC if sc is None else sc; tauc = TAU_C if tauc is None else tauc; sc0 = (SC if SC0 is None else SC0) if sc0 is None else sc0
    F4, Qc4, P04, H4 = model_spring(*ROTOR); n = 17; F = np.zeros((n, n)); Qc = np.zeros((n, n)); P0 = np.zeros((n, n))
    for j in range(3):
        sl = slice(4 * j, 4 * j + 4); F[sl, sl] = F4; Qc[sl, sl] = Qc4; P0[sl, sl] = P04
    for j in (12, 13): F[j, j] = -1.0 / tauc; Qc[j, j] = 2 * sc ** 2 / tauc; P0[j, j] = sc0 ** 2
    for j in (14, 15, 16): P0[j, j] = LEVER_PRIOR ** 2
    return F, Qc, P0, H4

# =============================================================================================== input
def _find_member(names):
    names = [n for n in names if not n.startswith("__MACOSX") and not n.endswith("/")]; low = [n.replace("\\", "/").lower() for n in names]
    for want in ("imu0/data.csv", "data.csv"):
        hits = sorted((len(l), n) for n, l in zip(names, low) if l == want or l.endswith("/" + want))
        if hits: return hits[0][1]
    csv = [n for n, l in zip(names, low) if l.endswith(".csv")]
    return csv[0] if len(csv) == 1 else (sorted(csv, key=len)[0] if csv else None)
class _ZipStream:
    """Streaming read of one stored / deflated zip member with zlib (the zipfile module alone costs ~0.9 MB of heap)."""
    def __init__(self, path, off, csize, meth, crc):
        self.f = open(path, "rb"); self.f.seek(off); h = self.f.read(30)
        if h[:4] != b"PK\x03\x04": self.f.close(); raise ValueError("bad local header")
        self.f.seek(off + 30 + int.from_bytes(h[26:28], "little") + int.from_bytes(h[28:30], "little"))
        self.left = csize; self.z = zlib.decompressobj(-15) if meth == 8 else None; self.buf = b""; self.crc = 0; self.want = crc; self.eof = False
    def read(self, n):
        """Up to n bytes; decompression is bounded so that no buffer exceeds n bytes."""
        out = []; got = 0; z = self.z
        while got < n and not self.eof:
            if z is not None and z.unconsumed_tail: raw = z.unconsumed_tail
            else:
                raw = self.f.read(min(self.left, 1 << 14)); self.left -= len(raw)
            if not raw:
                self.eof = True; d = z.flush() if z is not None else b""
            else: d = z.decompress(raw, n - got) if z is not None else raw
            if d: self.crc = zlib.crc32(d, self.crc); out.append(d); got += len(d)
        if self.eof and self.crc != self.want: raise ValueError("zip member CRC mismatch")
        return b"".join(out)
    def close(self): self.f.close()
def _zip_dir(path):
    """Central directory of a plain (non-zip64, unencrypted) archive: [(name, method, compressed size, offset, crc)], or None."""
    with open(path, "rb") as f:
        f.seek(0, 2); size = f.tell(); k = min(size, 65536 + 22); f.seek(size - k); tail = f.read(k); i = tail.rfind(b"PK\x05\x06")
        if i < 0 or len(tail) - i < 22: return None
        u = lambda b, a, n: int.from_bytes(b[a:a + n], "little"); n_tot = u(tail, i + 10, 2); cd_size = u(tail, i + 12, 4); cd_off = u(tail, i + 16, 4)
        if n_tot == 0xFFFF or cd_off == 0xFFFFFFFF: return None
        f.seek(cd_off); cd = f.read(cd_size); out = []; p = 0
        for _ in range(n_tot):
            if cd[p:p + 4] != b"PK\x01\x02": return None
            flag, meth, crc, csize = u(cd, p + 8, 2), u(cd, p + 10, 2), u(cd, p + 16, 4), u(cd, p + 20, 4); nl, el, cl, off = u(cd, p + 28, 2), u(cd, p + 30, 2), u(cd, p + 32, 2), u(cd, p + 42, 4)
            name = cd[p + 46:p + 46 + nl].decode("utf-8" if flag & 0x800 else "latin-1", "replace")
            out.append((name, meth if not flag & 1 and meth in (0, 8) and 0xFFFFFFFF not in (csize, off) else -1, csize, off, crc)); p += 46 + nl + el + cl
        return out
def _is_zip(path):
    try:
        with open(path, "rb") as f: return f.read(4) in (b"PK\x03\x04", b"PK\x05\x06")
    except OSError: return False
class Source:
    """The IMU CSV inside INPUT (zip, folder or file); open() can be called repeatedly (the input is read more than once)."""
    def __init__(self, path):
        path = path or "."; self.zip = None; self.member = None
        if os.path.isdir(path):
            files = []
            for root, _d, fs in os.walk(path):
                for f in fs: files.append(os.path.relpath(os.path.join(root, f), path))
            m = _find_member([f for f in files if f.lower().endswith(".csv")])
            if m is None:
                zips = [f for f in files if f.lower().endswith(".zip")]
                if not zips: raise FileNotFoundError("no IMU CSV found under %s" % path)
                path = os.path.join(path, sorted(zips, key=len)[0])
            else: path = os.path.join(path, m)
        if not os.path.isfile(path): raise FileNotFoundError(path)
        if _is_zip(path):
            self.zip = path; ents = _zip_dir(path)
            if ents is not None:
                m = _find_member([e[0] for e in ents]); self.entry = next((e for e in ents if e[0] == m), None) if m else None
                if self.entry is None: raise FileNotFoundError("no CSV inside %s" % path)
                self.member = m
                if self.entry[1] < 0: ents = None
            if ents is None:                                            # zip64, encryption, other compression: the zipfile module
                import zipfile
                with zipfile.ZipFile(path) as z: self.member = _find_member(z.namelist())
                if self.member is None: raise FileNotFoundError("no CSV inside %s" % path)
                self.entry = None
        self.path = path
    def open(self):
        if self.zip is None: return open(self.path, "rb"), None
        if self.entry is not None: _n, meth, csize, off, crc = self.entry; return _ZipStream(self.zip, off, csize, meth, crc), None
        import zipfile
        z = zipfile.ZipFile(self.zip); return z.open(self.member), z

def _ts_any(tok):
    try: return int(tok)
    except ValueError: v = float(tok)
    s = 1e9 if abs(v) < 1e11 else (1e6 if abs(v) < 1e14 else (1e3 if abs(v) < 1e17 else 1.0)); return int(round(v * s))
def _parse_fast(data, cut, want):
    """Vectorised parse of data[:cut] (complete lines): integer timestamps and six comma-separated values per data line;
    lines that are empty or start with '#' or a letter are skipped. None if the block is irregular (-> line-by-line parse)."""
    if cut == 0: return np.zeros(0, np.int64), (np.zeros((0, 6)) if want else None)
    b = np.frombuffer(data, np.uint8, count=cut); nl = np.flatnonzero(b == 10)
    st = np.concatenate([[0], nl + 1]); en = np.concatenate([nl, [cut]]); ne = en > st
    c = np.zeros(len(st), np.uint8); c[ne] = b[st[ne]]
    dat = ne & (((c >= 48) & (c <= 57)) | (c == 45) | (c == 43) | (c == 46))
    skip = ~ne | (c == 35) | (c == 13) | ((c >= 65) & (c <= 90)) | ((c >= 97) & (c <= 122))
    if not (dat | skip).all(): return None
    ds = st[dat]; de = en[dat]; nd = len(ds)
    if nd == 0: return np.zeros(0, np.int64), (np.zeros((0, 6)) if want else None)
    cm = np.flatnonzero(b == 44); lo = np.searchsorted(cm, ds); hi = np.searchsorted(cm, de)
    if not (hi - lo == 6).all(): return None
    c0 = cm[lo]; tl = c0 - ds
    if tl.min() < 1 or tl.max() > 19: return None
    if (tl == 19).any() and (b[ds[tl == 19]] >= 57).any(): return None   # could exceed int64
    ts = np.zeros(nd, np.int64)
    for j in range(int(tl.max())):                                      # digits, most significant first
        m = tl > j; p = ds[m] + j; dgt = b[p].astype(np.int64) - 48
        if not ((dgt >= 0) & (dgt <= 9)).all(): return None
        ts[m] = ts[m] * 10 + dgt
    if not want: return ts, None
    w = b.copy(); bl = np.zeros(cut + 1, np.int8); np.add.at(bl, ds, 1); np.add.at(bl, c0 + 1, -1); w[np.cumsum(bl[:cut], dtype=np.int8) > 0] = 32; del bl
    for i in np.flatnonzero(skip & ne).tolist(): w[st[i]:en[i]] = 32
    if len(nl): w[nl] = np.where(dat[:-1], 44, 32)
    try: v = np.fromstring(w.tobytes(), dtype=np.float64, sep=",")
    except ValueError: return None
    if v.size != 6 * nd: return None
    return ts, v.reshape(nd, 6)
def _parse_slow(block, want):
    ts = []; va = []
    for ln in block.split(b"\n"):
        s = ln.strip()
        if not s: continue
        c = s[0]
        if c == 35 or 65 <= c <= 90 or 97 <= c <= 122: continue          # comment or header line
        f = [x.strip() for x in s.split(b",")] if b"," in s else s.split()
        try: tv = _ts_any(f[0])
        except (ValueError, OverflowError): continue
        if not -2 ** 63 <= tv < 2 ** 63: continue
        ts.append(tv)
        if want:
            row = []
            for j in range(1, 7):
                try: row.append(float(f[j]) if j < len(f) and f[j] else float("nan"))
                except ValueError: row.append(float("nan"))
            va.append(row)
    return np.array(ts, np.int64), (np.array(va, np.float64).reshape(-1, 6) if want else None)
def iter_rows(src, want, chunk=1 << 16):
    """Yields (timestamps int64, values float64 (n, 6) or None) per block of complete lines, in file order."""
    s, z = src.open()
    try:
        rest = b""
        while True:
            buf = s.read(chunk)
            if not buf: break
            data = rest + buf if rest else buf; cut = data.rfind(b"\n")
            if cut < 0: rest = data; continue
            r = _parse_fast(data, cut, want)
            if r is None: r = _parse_slow(data[:cut], want)
            rest = data[cut + 1:]
            if len(r[0]): yield r
        if rest.strip():
            r = _parse_fast(rest, len(rest), want)
            if r is None: r = _parse_slow(rest, want)
            if len(r[0]): yield r
    finally:
        s.close()
        if z is not None: z.close()

# =============================================================================================== M1-M2: slot grid, raw storage
class Data:
    """Persistent per-run state: slot times t (float64), raw gyro / accelerometer W, A (int32 micro-units, SENT = invalid
    value), validity mask, row -> slot map (int32), spike repairs, and the gap-filled signal provider."""
    def __init__(self, src):
        self.src = src; TS = []
        for ts, _ in iter_rows(src, False): TS.append(ts)
        tn = np.concatenate(TS) if TS else np.zeros(0, np.int64); del TS; n = len(tn); self.n_rows = n
        if n < 2: raise ValueError("fewer than two rows")
        d = np.diff(tn); pos = d[d > 0]; med_in = float(_median(pos, inplace=True)) if len(pos) else 1e9; del pos
        self.tscale = 1
        if med_in < 2e4:                                                # > 50 kHz in ns: the stamps are in us, ms or s
            for sc in (10 ** 3, 10 ** 6, 10 ** 9):
                if 1e9 / (med_in * sc) <= 5000: self.tscale = sc; log("timestamps rescaled by %g" % sc); break
        if self.tscale != 1: tn = tn * self.tscale; d = d * self.tscale
        self.mono = not (d < 0).any(); order = None
        if not self.mono:
            log("timestamps not monotonic: estimated on a sorted copy"); order = np.argsort(tn, kind="stable"); tn = tn[order]; d = np.diff(tn)
        pos = d[d > 0]; med = float(_median(pos, inplace=True)) if len(pos) else 1e9; self.n_distinct = len(pos) + 1; del pos
        dt_ = np.int64 if n > 2 ** 31 - 2 else np.int32; slots = np.empty(n, np.int64 if order is not None else dt_); slots[0] = 0; carry = 0
        for c0 in range(0, n - 1, 1 << 16):                             # slot step = max(round(d / med), 0), cumulated
            x = d[c0:c0 + (1 << 16)] / med; np.round(x, out=x); st = x.astype(np.int64); del x; np.maximum(st, 0, out=st)
            cs = np.cumsum(st); cs += carry; slots[c0 + 1:c0 + 1 + len(cs)] = cs; carry = int(cs[-1]); del st, cs
        del d; N = carry + 1; self.N = N; self.med = med
        if N > 50_000_000: raise MemoryError("implausible time line (%d slots)" % N)
        t = np.empty(N); isocc = np.zeros(N, bool); t0n = int(tn[0]); prev = -1
        for c0 in range(0, n, 1 << 16):                                 # slot time = time of the first row in the slot
            sl = slots[c0:c0 + (1 << 16)].astype(np.int64); f = np.ones(len(sl), bool); f[1:] = sl[1:] != sl[:-1]; f[0] = sl[0] != prev; prev = int(sl[-1])
            t[sl[f]] = (tn[c0:c0 + (1 << 16)][f] - t0n) * 1e-9; isocc[sl[f]] = True; del sl, f
        del tn
        if not isocc.all():                                             # = np.interp(arange(N), occupied, times) at the empty slots
            for c0 in range(0, N, CH):
                c1 = min(N, c0 + CH); m = ~isocc[c0:c1]
                if not m.any(): continue
                lo = c0
                while lo > 0 and not isocc[lo]: lo -= 1
                hi = c1
                while hi < N - 1 and not isocc[hi]: hi += 1
                occ = np.flatnonzero(isocc[lo:hi + 1]) + lo; t[c0:c1][m] = np.interp(np.arange(c0, c1)[m], occ, t[occ])
        del isocc; self.t = t
        if order is None: self.row_slot = None; self.rank = None; del slots                   # recomputed on every later read
        else:
            rs = np.empty(n, np.int64); rs[order] = slots; self.row_slot = rs.astype(dt_); self.rank = np.empty(n, np.int64); self.rank[order] = np.arange(n); del rs, slots
        del order
        self.fac_w = 1.0; self.fac_a = 1.0; self.ov = None; self.zrun = (np.zeros(0, np.int64), np.zeros(0, np.int64)); self._read_values()
    def rows(self, want):
        """Second and later reads: (timestamps in ns, slot of each row, values or None) per block, in input order."""
        r0 = 0; prev = None; carry = 0
        for ts, v in iter_rows(self.src, want):
            k = len(ts)
            if self.tscale != 1: ts = ts * self.tscale
            if self.row_slot is not None: sl = self.row_slot[r0:r0 + k].astype(np.int64)
            else:                                                       # = the cumulated max(round(d / med), 0) of the first read
                d = np.diff(ts) if prev is None else np.diff(np.concatenate([[prev], ts])); x = d / self.med; np.round(x, out=x); st = x.astype(np.int64); np.maximum(st, 0, out=st)
                sl = np.empty(k, np.int64)
                if prev is None: sl[0] = 0; sl[1:] = np.cumsum(st)
                else: sl[:] = carry + np.cumsum(st)
                prev = int(ts[-1]); carry = int(sl[-1]); del d, x, st
            r0 += k; yield ts, sl, v
        if r0 != self.n_rows: raise ValueError("the input changed between passes")
    def _read_values(self, accel_only=False):
        """Values pass: raw samples -> slots (rows sharing a slot: the last row in time order wins, as in the reference)."""
        N = self.N; A = np.full((N, 3), SENT, np.int32); W = None if accel_only else np.full((N, 3), SENT, np.int32)
        writer = None if self.rank is None else np.full(N, -1, np.int64); r0 = 0; n_ok = 0
        for _ts, sl, v in self.rows(True):
            k = len(sl)
            fin = np.isfinite(v) & (np.abs(v) < 2147.48); q = np.round(np.where(fin, v, 0.0) * SCALE).astype(np.int32); q[~fin] = SENT
            n_ok += int((np.isfinite(v).all(1) & ~(v == 0).all(1)).sum())
            if writer is not None:                                      # time order decides which duplicate wins
                rk = self.rank[r0:r0 + k]; o = np.argsort(rk, kind="stable"); sl = sl[o]; q = q[o]; rk = rk[o]
                keep = rk > writer[sl]; sl = sl[keep]; q = q[keep]; writer[sl] = rk[keep]
            A[sl] = q[:, 3:]; r0 += k
            if W is not None: W[sl] = q[:, :3]
        del writer; self.A = A
        if accel_only: return
        self.W = W; self.n_ok = n_ok
        valid = np.empty(N, bool); prev = None                          # M1: zero / NaN rows and exact repeats are invalid
        for c0 in range(0, N, CH):
            c1 = min(N, c0 + CH); X = np.hstack([W[c0:c1], A[c0:c1]]); bad = (X == SENT).any(1) | (X == 0).all(1); same = np.zeros(c1 - c0, bool)
            if prev is None: same[1:] = (X[1:] == X[:-1]).all(1)
            else: same[:] = (X == np.vstack([prev, X[:-1]])).all(1)
            same &= ~bad; valid[c0:c1] = ~(bad | same); prev = X[-1:]
        self.valid = valid
        if not valid.any(): raise ValueError("no valid sample")
        self.L = int(N - 1 - np.argmax(valid[::-1])); self.N0 = self.L + 1; self.nt = N - 1 - self.L
    def accel(self):
        """The accelerometer array is released after the back-end measurements; read it again if it is needed later."""
        if self.A is None: self._read_values(accel_only=True)
        return self.A
    def unit_check(self):
        """Defensive unit checks (never triggered by the official format): accelerometer in g (median |a| < 2), gyro in deg/s
        (median |w| > 20); the median is compared with the threshold exactly from counts and the two neighbouring values."""
        ok = self.valid; n = int(ok.sum())
        if n <= 10: return
        for which, thr in (("a", 2.0), ("w", 20.0)):
            below = 0; mx_b = -np.inf; mn_a = np.inf
            for c0 in range(0, self.N, CH):
                c1 = min(self.N, c0 + CH); m = ok[c0:c1]
                if not m.any(): continue
                x = np.linalg.norm(self.decode(c0, c1, which)[m], axis=1); lo = x < thr; below += int(lo.sum())
                if lo.any(): mx_b = max(mx_b, float(x[lo].max()))
                if not lo.all(): mn_a = min(mn_a, float(x[~lo].min()))
            if n % 2: med_lt = below >= (n + 1) // 2                  # is median < thr ?
            else: med_lt = below >= n // 2 + 1 or (below == n // 2 and (mx_b + mn_a) / 2 < thr)
            med_gt = (n - below) >= (n + 1) // 2 if n % 2 else ((n - below) >= n // 2 + 1 and mn_a > thr) or (below == n // 2 and (mx_b + mn_a) / 2 > thr)
            if which == "a" and med_lt: self.fac_a = 9.80665; log("accelerometer in g -> m/s^2")
            if which == "w" and med_gt: self.fac_w = math.pi / 180; log("gyro in deg/s -> rad/s")
    def medians(self):
        """Median slot step on the estimation span (m0) and on the span extended over the blind window (me); the slot times
        of the blind window become the reference's extrapolated time line te. One scratch buffer of N - 1 steps."""
        N0 = self.N0; N = self.N; t = self.t; nt = self.nt; buf = np.empty(max(N - 1, 1)); np.subtract(t[1:N0], t[:N0 - 1], out=buf[:N0 - 1])
        self.m0 = float(_median(buf[:N0 - 1], inplace=True)); self.fs = 1.0 / self.m0
        if nt > 0:
            te = np.concatenate([t[N0 - 1:N0], t[N0 - 1] + (1.0 / self.fs) * np.arange(1, nt + 1)]); buf[N0 - 1:N - 1] = np.diff(te)
            self.me = float(_median(buf[:N - 1], inplace=True)); t[N0:] = te[1:]; del te
        else: self.me = self.m0
        del buf; self.fs_e = 1.0 / self.me
    # ------------------------------------------------------------------------------------------- providers
    def decode(self, s0, s1, which):
        """Raw float64 values of slots s0..s1-1 (NaN where invalid-valued), unit factors and spike repairs applied."""
        k = (self.W if which == "w" else self.accel())[s0:s1]; x = k / SCALE; x[k == SENT] = np.nan; f = self.fac_w if which == "w" else self.fac_a
        if f != 1.0: x *= f
        if self.ov is not None:
            i0 = int(np.searchsorted(self.ov[0], s0)); i1 = int(np.searchsorted(self.ov[0], s1))
            if i1 > i0: x[self.ov[0][i0:i1] - s0] = self.ov[1 if which == "w" else 2][i0:i1]
        return x
    def set_zmask(self, m):
        """Zero-velocity (genuine rest) slots, kept as runs."""
        dv = np.diff(np.concatenate([[0], m.astype(np.int8), [0]])); self.zrun = (np.flatnonzero(dv == 1), np.flatnonzero(dv == -1))
    def zmask(self, s0, s1):
        out = np.zeros(s1 - s0, bool); a, b = self.zrun; j0 = int(np.searchsorted(b, s0, "right")); j1 = int(np.searchsorted(a, s1))
        for i in range(j0, j1): out[max(a[i], s0) - s0:min(b[i], s1) - s0] = True
        return out
    def decode_at(self, idx, which):
        """decode() at scattered slots."""
        idx = np.asarray(idx, np.int64); k = (self.W if which == "w" else self.accel())[idx]; x = k / SCALE; x[k == SENT] = np.nan; f = self.fac_w if which == "w" else self.fac_a
        if f != 1.0: x *= f
        if self.ov is not None and len(idx):
            j = np.searchsorted(self.ov[0], idx); j = np.minimum(j, len(self.ov[0]) - 1); hit = self.ov[0][j] == idx if len(self.ov[0]) else np.zeros(len(idx), bool)
            if hit.any(): x[hit] = self.ov[1 if which == "w" else 2][j[hit]]
        return x
    def prev_valid(self, s):
        if s <= 0: return -1
        seg = self.valid[:s][::-1]; i = int(np.argmax(seg)); return s - 1 - i if seg[i] else -1
    def next_valid(self, s):
        if s >= self.N0: return -1
        seg = self.valid[s:self.N0]; i = int(np.argmax(seg)); return s + i if seg[i] else -1
    def filled(self, s0, s1, which):
        """fill_gaps of the reference on slots s0..s1-1 (s1 <= N0): linear interpolation in time across invalid slots,
        constant beyond the first / last valid sample."""
        X = self.decode(s0, s1, which); v = self.valid[s0:s1]
        if v.all(): return X
        p = self.prev_valid(s0); q = self.next_valid(s1); tv = [self.t[s0:s1][v]]; xv = [X[v]]
        if p >= 0: tv.insert(0, self.t[p:p + 1]); xv.insert(0, self.decode(p, p + 1, which))
        if q >= 0: tv.append(self.t[q:q + 1]); xv.append(self.decode(q, q + 1, which))
        tv = np.concatenate(tv); xv = np.concatenate(xv); inv = ~v; ti = self.t[s0:s1][inv]
        for j in range(3): X[inv, j] = np.interp(ti, tv, xv[:, j])
        return X

# =============================================================================================== M3: repair
def repair(D, spike_k=10.0, floor_a=15.0, floor_w=1.5, g_lo=5.0, g_hi=15.5, win=1.0):
    """Isolated outliers far above the local robust spread are interpolated (stay valid); windows whose 1 s median |a| is
    outside [g_lo, g_hi] m/s^2 become invalid (fbs6.repair; despike floors as in the code: 1.5 rad/s, 15 m/s^2)."""
    N = D.N0; t = D.t; fs = D.fs; valid = D.valid; n1 = int(fs); m = N // n1 if n1 > 0 else 0; floor = np.array([floor_w] * 3 + [floor_a] * 3)
    B = n1 * max(1, CH // n1) if n1 > 0 else CH; spikes = []; last = [np.zeros(6)]
    def spike_chunk(c0):                                                # 5-sample median residual against k x the 1 s robust spread
        c1 = min(N, c0 + B); lo = max(0, c0 - 2); hi = min(N, c1 + 2); n = c1 - c0
        X = np.hstack([D.filled(lo, hi, "w"), D.filled(lo, hi, "a")]); P = np.concatenate([np.repeat(X[:1], 2 - (c0 - lo), 0), X, np.repeat(X[-1:], 2 - (hi - c1), 0)], 0)
        r = np.abs(X[c0 - lo:c0 - lo + n] - _med5(P[0:n], P[1:n + 1], P[2:n + 2], P[3:n + 3], P[4:n + 4])); del X, P
        sig = np.empty((n, 6)); nb = max(0, min(m, c1 // n1) - c0 // n1) if n1 > 0 else 0
        if nb:
            sb = _median(r[:nb * n1].reshape(nb, n1, 6), 1) * 1.4826; sig[:nb * n1] = np.repeat(sb, n1, 0); last[0] = sb[-1]
        sig[nb * n1:] = last[0]; return np.flatnonzero((r > np.maximum(floor, spike_k * sig)).any(1) & valid[c0:c1]) + c0
    for c0 in range(0, N, B):
        sp = spike_chunk(c0)
        if len(sp): spikes.append(sp)
    sp_idx = np.concatenate(spikes) if spikes else np.zeros(0, np.int64); ns = len(sp_idx); del spikes
    if ns:                                                              # spike values: interpolated from good neighbours
        spm = np.zeros(N, bool); spm[sp_idx] = True; ow = np.empty((ns, 3)); oa = np.empty((ns, 3))
        for k, s in enumerate(sp_idx.tolist()):
            p = s - 1
            while p >= 0 and not (valid[p] and not spm[p]): p -= 1
            q = s + 1
            while q < N and not (valid[q] and not spm[q]): q += 1
            ix = [i for i in (p, q) if 0 <= i < N]
            if not ix: ow[k] = D.decode(s, s + 1, "w")[0]; oa[k] = D.decode(s, s + 1, "a")[0]; continue
            tv = t[ix]; xw = np.vstack([D.decode(i, i + 1, "w") for i in ix]); xa = np.vstack([D.decode(i, i + 1, "a") for i in ix])
            for j in range(3): ow[k, j] = np.interp(t[s], tv, xw[:, j]); oa[k, j] = np.interp(t[s], tv, xa[:, j])
        del spm; D.ov = (sp_idx, ow, oa)
    h = max(1, int(win * fs) // 2); k = max(1, int(fs * 0.1)); idx = np.arange(0, N, k); flags = np.zeros(len(idx), bool); CB = k * max(1, CH // k)
    def flag_chunk(c0):                                                 # 1 s running median of |a| every 0.1 s
        c1 = min(N, c0 + CB); lo = max(0, c0 - h); hi = min(N, c1 + h); an = np.linalg.norm(D.filled(lo, hi, "a"), axis=1)
        for j in range(c0 // k, (c1 + k - 1) // k):
            i = int(idx[j]); med = _median(an[max(0, i - h) - lo:i + h + 1 - lo]); flags[j] = med < g_lo or med > g_hi
    for c0 in range(0, N, CB): flag_chunk(c0)
    imp = 0
    if flags.any():                                                     # interp of the flags > 0, dilated by +-0.5 s (np.convolve > 0)
        d = int(0.5 * fs); bad0 = np.zeros(N, bool); fl = flags.astype(float)
        for c0 in range(0, N, CH):
            c1 = min(N, c0 + CH); j0 = max(0, c0 // k - 1); j1 = min(len(idx), c1 // k + 2); bad0[c0:c1] = np.interp(np.arange(c0, c1), idx[j0:j1], fl[j0:j1]) > 0.0
        badd = np.zeros(N, bool)
        for c0 in range(0, N, CH):
            c1 = min(N, c0 + CH); lo = max(0, c0 - d); hi = min(N, c1 + d); cs = np.concatenate([[0], np.cumsum(bad0[lo:hi], dtype=np.int32)]); i = np.arange(c0, c1)
            badd[c0:c1] = (cs[np.minimum(hi, i + d + 1) - lo] - cs[np.maximum(lo, i - d) - lo]) > 0
        del bad0; badd &= valid[:N]; imp = int(badd.sum()); D.valid[:N] &= ~badd; del badd
    return dict(spikes=ns, implausible=imp)

# =============================================================================================== M4-M5: platform statistics and gate
def thrust_lock_fill(D, win=10.0, span=3):
    """Median over 10 s windows (5 s hop) of RMS(f_perp)/g of the 1 Hz low-passed, gap-filled specific force."""
    N = D.N0; fs = D.fs; n = int(win * fs); hop = n // 2; out = []; valid = D.valid; get = lambda a, b: D.filled(a, b, "a")
    if hop < 1: return 1.0
    starts = list(range(0, N - n + 1, hop))
    def group(ss):
        a = ss[0]; al = zp_range(get, N, fs, 1.0, a, ss[-1] + n); res = []
        for i in ss:
            v = valid[i:i + n]
            if v.sum() < n // 2: continue
            x = al[i - a:i - a + n][v]; zb = x.mean(0); zb = zb / np.linalg.norm(zb); perp = x - np.outer(x @ zb, zb); res.append(np.sqrt((perp ** 2).sum(1).mean()) / G)
        return res
    for g in range(0, len(starts), span): out += group(starts[g:g + span])
    return float(_median(out)) if out else 1.0
def rest_info(D, win=0.5, frac=0.05, gate=0.035, min_windows=3):
    """Genuine rest: clusters of >= 3 quiet 0.5 s windows whose gyro means agree. Returns (anchors [(time, bias)], mask)."""
    N = D.N0; fs = D.fs; mask = np.zeros(N, bool); n = max(8, int(win * fs)); m = N // n
    okw = D.valid[:m * n].reshape(m, n).all(1)
    if okw.sum() < 10: return [], mask
    gs = np.full(m, np.inf); as_ = np.full(m, np.inf); WB = max(1, CH // n)
    for w0 in range(0, m, WB):
        w1 = min(m, w0 + WB); ok = okw[w0:w1]
        if not ok.any(): continue
        Wn = np.nan_to_num(D.decode(w0 * n, w1 * n, "w")).reshape(-1, n, 3); An = np.nan_to_num(D.decode(w0 * n, w1 * n, "a")).reshape(-1, n, 3)
        gs[w0:w1][ok] = Wn.std(1).max(1)[ok]; as_[w0:w1][ok] = An.std(1).max(1)[ok]
    sc = gs / _median(gs[okw]) + as_ / _median(as_[okw]); k = max(3, int(frac * okw.sum())); cand = np.argsort(sc)[:k]; cand = np.sort(cand[gs[cand] < gate]); out = []
    if len(cand) == 0: return out, mask
    t = D.t
    for g in np.split(cand, np.where(np.diff(cand) > 4)[0] + 1):
        if len(g) < min_windows: continue
        Wg = np.stack([D.decode(int(i) * n, int(i) * n + n, "w") for i in g]); means = Wg.mean(1); spread = means.std(0).max(); allow = 1.0e-3 + 3.0 * gs[g].mean() / np.sqrt(n)
        if spread < allow:
            ti = g[:, None] * n + np.arange(n)[None, :]; out.append((float(t[ti].mean()), _median(means, 0))); mask[ti.ravel()] = True
    return out, mask

# =============================================================================================== M6: gyro bridge
class Bridge:
    """Ornstein-Uhlenbeck bridge across each interior gap of the gyro (fbs5.bridge_gyro): conditional mean pinned to the two
    end samples, reverting to the local mean (+-5 s); outside interior gaps the gap-filled signal."""
    def __init__(self, D, tau=0.4, mwin=5.0):
        self.D = D; self.tau = tau; N = D.N0; v = D.valid[:N]; k = int(mwin * D.fs)
        dv = np.diff(np.concatenate([[0], (~v).astype(np.int8), [0]])); i0 = np.flatnonzero(dv == 1); i1 = np.flatnonzero(dv == -1); del dv
        keep = (i0 > 0) & (i1 < N); self.i0, self.i1 = i0[keep], i1[keep]; self.ng = len(self.i0)
        if not self.ng: return
        lo = np.maximum(0, self.i0 - k); hi = np.minimum(N, self.i1 + k); need = _uniq(np.concatenate([lo, self.i0, self.i1, hi]))
        cs = np.empty((len(need), 3)); cv = np.empty(len(need), np.int64); acc = np.zeros(3); cnt = 0; p = 0
        while p < len(need) and need[p] == 0: cs[p] = 0.0; cv[p] = 0; p += 1
        for c0 in range(0, N, CH):                                      # cumulative sums in the reference's order
            c1 = min(N, c0 + CH); vv = v[c0:c1]; wv = np.where(vv[:, None], np.nan_to_num(D.decode(c0, c1, "w")), 0.0)
            csl = np.concatenate([acc[None], wv], 0).cumsum(0)[1:]; cvl = cnt + np.cumsum(vv)
            while p < len(need) and need[p] <= c1:
                j = int(need[p]) - c0 - 1; cs[p] = csl[j]; cv[p] = cvl[j]; p += 1
            acc = csl[-1]; cnt = int(cvl[-1])
        g = lambda ix: np.searchsorted(need, ix)
        self.mu = 0.5 * ((cs[g(self.i0)] - cs[g(lo)]) / (cv[g(self.i0)] - cv[g(lo)])[:, None] + (cs[g(hi)] - cs[g(self.i1)]) / (cv[g(hi)] - cv[g(self.i1)])[:, None])

    def __call__(self, s0, s1):
        D = self.D; out = D.filled(s0, s1, "w")
        if not self.ng: return out
        a = int(np.searchsorted(self.i1, s0, "right")); b = int(np.searchsorted(self.i0, s1))
        if b <= a: return out
        i0, i1, t = self.i0[a:b], self.i1[a:b], D.t; j0 = np.maximum(i0, s0); j1 = np.minimum(i1, s1); lens = j1 - j0
        g = np.repeat(np.arange(len(i0)), lens); j = j0[g] + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)); tau = self.tau
        T = (t[i1] - t[i0 - 1])[g]; tt = t[j] - t[i0 - 1][g]; den = 1 - np.exp(-2 * T / tau)
        s0w = np.exp(-tt / tau) * (1 - np.exp(-2 * (T - tt) / tau)) / den; s1w = np.exp(-(T - tt) / tau) * (1 - np.exp(-2 * tt / tau)) / den
        wl = D.decode_at(i0 - 1, "w"); wr = D.decode_at(i1, "w")                                    # the two end samples of each gap
        m_ = self.mu[a:b][g]; out[j - s0] = m_ + s0w[:, None] * (wl[g] - m_) + s1w[:, None] * (wr[g] - m_); return out

# =============================================================================================== M8: gravity smoother
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
    the mask and the jumps, so it is computed once per noise setting; every call runs two affine scans (in chunks)."""
    SB = 1024
    def __init__(self, D, d=16, tau=TAU_G, k2=0.25):
        N = D.N0; valid = D.valid[:N]; t = D.t; self.N, self.d = N, d; dt = D.m0; self.dt = dt; K = (N + d - 1) // d; self.K = K
        st = np.flatnonzero(valid[1:] & ~valid[:-1]) + 1; en = np.flatnonzero(~valid[1:] & valid[:-1]) + 1; j = np.searchsorted(en, st) - 1
        T = np.where(j >= 0, t[st] - t[en[np.maximum(j, 0)] - 1], 0.0) if len(st) else np.zeros(0)
        self.jblk = st // d; self.jth = np.minimum(0.5 * k2 * T * T, 1.0); cnt = np.zeros(K)
        for c0 in range(0, N, CH):
            vb = valid[c0:min(N, c0 + CH)].astype(float); pad = (-len(vb)) % d
            if pad: vb = np.concatenate([vb, np.zeros(pad)])
            cnt[c0 // d:c0 // d + len(vb) // d] = vb.reshape(-1, d).sum(1)
        self.cnt = cnt; self.okb = cnt > 0; self.q = dt / tau ** 4; self.dtb = d * dt; self.cache = {}
        self.tb = (np.arange(K) * d + (d - 1) / 2.0) * dt
    def _gains(self, sig_m):
        if sig_m in self.cache: return self.cache[sig_m]
        K = self.K; dtb = self.dtb; q = self.q; q11, q12, q22 = q * dtb ** 3 / 3, q * dtb ** 2 / 2, q * dtb
        jb = np.zeros(K)
        if len(self.jblk): np.add.at(jb, self.jblk, (9.81 * self.jth / sig_m) ** 2)
        g0 = np.zeros(K); g1 = np.zeros(K); PF = np.empty((K, 3)); PP = np.empty((K, 3)); p00, p01, p11 = 1e6, 0.0, 1e2; first = -1
        for a in range(0, K, self.SB):                                  # pure-Python recursion, its inputs converted a block at a time
            i = a
            for ji, oki, ci in zip(jb[a:a + self.SB].tolist(), self.okb[a:a + self.SB].tolist(), self.cnt[a:a + self.SB].tolist()):
                if first >= 0: p00, p01, p11 = p00 + 2 * dtb * p01 + dtb * dtb * p11 + q11, p01 + dtb * p11 + q12, p11 + q22
                p00 += ji; p11 += 0.01 * ji; PP[i] = (p00, p01, p11)
                if oki:
                    if first < 0: first = i
                    s = p00 + 1.0 / ci; k0, k1 = p00 / s, p01 / s; p00, p01, p11 = p00 - k0 * p00, p01 - k0 * p01, p11 - k1 * p01; g0[i] = k0; g1[i] = k1
                PF[i] = (p00, p01, p11); i += 1
        del jb; C = np.empty((max(K - 1, 0), 4))
        for a in range(0, K - 1, self.SB):                              # RTS coefficients, elementwise, a block at a time
            b = min(K - 1, a + self.SB); f00, f01, f11 = PF[a:b, 0], PF[a:b, 1], PF[a:b, 2]; b00, b01, b10, b11 = f00 + dtb * f01, f01, f01 + dtb * f11, f11
            d00, d01, d11 = PP[a + 1:b + 1, 0], PP[a + 1:b + 1, 1], PP[a + 1:b + 1, 2]; det = d00 * d11 - d01 * d01
            C[a:b, 0] = (b00 * d11 - b01 * d01) / det; C[a:b, 1] = (-b00 * d01 + b01 * d00) / det; C[a:b, 2] = (b10 * d11 - b11 * d01) / det; C[a:b, 3] = (-b10 * d01 + b11 * d00) / det
        del PF, PP; r = dict(first=first, g0=g0, g1=g1, C=C)
        if sig_m == 1.5: self.cache[sig_m] = r                         # the gravity setting is used by every pass; the regressor one once
        return r
    def accum(self, xs, c0, x, valid):
        """Add the block sums over valid slots of x (rows c0..c0+len(x)-1, c0 a multiple of d) into xs."""
        d = self.d; xv = np.nan_to_num(x) * valid[:, None]; pad = (-len(xv)) % d
        if pad: xv = np.concatenate([xv, np.zeros((pad, xv.shape[1]))])
        xs[c0 // d:c0 // d + len(xv) // d] += xv.reshape(-1, d, xv.shape[1]).sum(1)
    def smooth(self, xb, sig_m=1.5, inplace=False, r=None):
        """Smoothed level of each column at the block centres (K, C); inplace=True overwrites xb with the result."""
        r = self._gains(sig_m) if r is None else r; K = self.K; f = r["first"]; C = xb.shape[1]; dtb = self.dtb; SB = self.SB
        if f < 0:
            if inplace: xb[:] = 0.0; return xb
            return np.zeros((K, C))
        if inplace and C > 3:                                           # columns are independent: three at a time
            for c in range(0, C, 3):
                sub = np.ascontiguousarray(xb[:, c:c + 3]); self.smooth(sub, sig_m, True, r); xb[:, c:c + 3] = sub; del sub
            return xb
        F0 = xb if inplace else np.zeros((K, C)); F1 = np.zeros((K, C)); x0 = xb[f].copy(); x1 = np.zeros(C)
        if not inplace: F0[f] = xb[f]
        F0[:f] = 0.0
        for a in range(f + 1, K, SB):
            b = min(K, a + SB); g0 = r["g0"][a:b]; g1 = r["g1"][a:b]
            F0[a:b], F1[a:b] = _scan(1 - g0, (1 - g0) * dtb, -g1, 1 - g1 * dtb, g0[:, None] * xb[a:b], g1[:, None] * xb[a:b], x0, x1); x0 = F0[b - 1].copy(); x1 = F1[b - 1].copy()
        if K == 1: return F0
        Cm = r["C"]; s0 = F0[K - 1].copy(); s1 = F1[K - 1].copy()
        for b in range(K - 1, 0, -SB):
            a = max(0, b - SB); c = Cm[a:b][::-1]; c00, c01, c10, c11 = c[:, 0], c[:, 1], c[:, 2], c[:, 3]; R0 = F0[a:b][::-1]; R1 = F1[a:b][::-1]
            v0 = (1 - c00)[:, None] * R0 + (-(c00 * dtb + c01))[:, None] * R1; v1 = (-c10)[:, None] * R0 + (1 - (c10 * dtb + c11))[:, None] * R1
            S0, S1 = _scan(c00, c01, c10, c11, v0, v1, s0, s1); F0[a:b] = S0[::-1]; s0 = S0[-1].copy(); s1 = S1[-1].copy()
        return F0
    def to_slots(self, sb, c0, c1):
        tf = np.arange(c0, c1) * self.dt; return np.stack([np.interp(tf, self.tb, sb[:, c]) for c in range(sb.shape[1])], 1)

# =============================================================================================== M9-M10: bias tracks and attitude passes
class Bias:
    """Gyro-bias track as a base (zero, rest-anchor interpolation, or a full-rate array) plus a sequence of increments
    (constant, knot interpolation, linear spline), evaluated chunk-wise in the reference's order of additions."""
    def __init__(self, base=None, arr=None):
        self.base = base; self.arr = arr; self.terms = []
    def plus(self, term):
        b = Bias(self.base, self.arr); b.terms = self.terms + [term]; return b
    def __call__(self, t, s0, s1):
        tt = t[s0:s1]
        if self.arr is not None: bg = self.arr[s0:s1].copy()
        elif self.base is not None: ta, B = self.base; bg = np.stack([np.interp(tt, ta, B[:, i]) for i in range(3)], 1)
        else: bg = np.zeros((s1 - s0, 3))
        bas = None
        for term in self.terms:
            if term[0] == "const": bg = bg + term[1]
            elif term[0] == "knots": bg = bg + np.stack([np.interp(tt, term[1], term[2][:, i]) for i in range(3)], 1)
            else:
                if bas is None: bas = _basis_idx(tt, term[1], term[2], term[3])
                _K, i0, w0, i1, w1 = bas; beta = term[4]; bg = bg + (w0[:, None] * beta[i0] + w1[:, None] * beta[i1])
        return bg
def _basis_idx(tm, t0, t1, knot):
    """Linear-spline basis of the reference (fbs5.attitude6) as two (index, weight) pairs per sample; K functions."""
    if t1 - t0 < 1.5 * knot:
        s = (tm - 0.5 * (t0 + t1)) / max(t1 - t0, 1e-9); return 2, np.zeros(len(tm), np.int64), np.ones_like(tm), np.ones(len(tm), np.int64), s
    K = int(np.ceil((t1 - t0) / knot)) + 1; u = np.clip((tm - t0) / knot, 0, K - 1 - 1e-9); i = np.floor(u).astype(np.int64); fr = u - i
    return K, i, 1 - fr, i + 1, fr
def _basis_eval(tt, t0, t1, knot, beta):
    K, i0, w0, i1, w1 = _basis_idx(tt, t0, t1, knot); return w0[:, None] * beta[i0] + w1[:, None] * beta[i1]
class Att:
    """One attitude solution: bias track, strapdown carries per chunk and the smoothed gravity block series; q is regenerated
    chunk by chunk (strapdown of the bridged gyro minus bias, gravity-tilt correction)."""
    def __init__(self, F, bias, carries, sb):
        self.F, self.bias, self.carries, self.sb = F, bias, carries, sb
    def chunk(self, ci):
        """(c0, c1, q, qc, |g|) of chunk ci."""
        F = self.F; c0 = ci * CH; c1 = min(F.N, c0 + CH); qg = F.strap(c0, c1, self.bias, self.carries[ci])
        gI = F.gs.to_slots(self.sb, c0, c1); gn = np.linalg.norm(gI, axis=1, keepdims=True); gI /= gn
        qc = shortest_arc(gI, np.broadcast_to(EZ, gI.shape)); del gI; q = qmul(qc, qg); del qg; q /= np.linalg.norm(q, axis=1, keepdims=True)
        return c0, c1, q, qc, gn[:, 0]
    def q(self, ci):
        """Attitude of chunk ci only."""
        r = self.chunk(ci); return r[2]
class Front:
    def __init__(self, D, tau_g=None):
        self.D = D; N = D.N0; self.N = N; self.fs = D.fs; self.t = D.t; self.valid = D.valid[:N]; self.nch = (N + CH - 1) // CH
        self.bridge = Bridge(D); self.gs = GravitySmoother(D, CFG["gs_dec"], TAU_G if tau_g is None else tau_g, CFG["k2"]); self.step = st = max(1, int(self.fs * 1.0))
        parts = []; cnt = 0; L = max(0, N - st)                         # every 10th i with valid[i .. i + st] (fbs5: same segment)
        for c0 in range(0, L, CH):
            c1 = min(L, c0 + CH); cs = np.concatenate([[0], np.cumsum(~self.valid[c0:c1 + st], dtype=np.int32)])
            idx = np.flatnonzero((cs[st + 1:st + 1 + c1 - c0] - cs[:c1 - c0]) == 0); parts.append(idx[(np.arange(len(idx)) + cnt) % 10 == 0] + c0); cnt += len(idx)
        self.ii = np.concatenate(parts) if parts else np.zeros(0, np.int64); del parts
    def wf(self, s0, s1): return self.bridge(s0, s1)
    def af(self, s0, s1): return self.D.filled(s0, s1, "a")
    def strap(self, c0, c1, bias, carry):
        """Strapdown attitude of slots c0..c1-1 (fbs.integrate_gyro of wf - bias) from the attitude at c0 - 1 (carry)."""
        t = self.t
        s0 = max(0, c0 - 1); w = self.wf(s0, c1); w -= bias(t, s0, c1); h = w[:-1] + w[1:]; del w
        h *= 0.5; h *= np.diff(t[s0:c1])[:, None]; dq = qexp(h); del h                            # = qexp(0.5 (w_k-1 + w_k) dt_k)
        out = qcumprod(np.concatenate([np.array([[0, 0, 0, 1.0]]) if c0 == 0 else carry[None], dq], 0)); del dq
        return out if c0 == 0 else out[1:]
    def attitude(self, bias, ii=False, rf=False, gn=False, aq=True, sub=None):
        """One pass of the reference's attitude(): strapdown, gravity in the gyro-integrated frame, tilt correction. Returns the
        Att solution and the requested collections: tilt drift y and A = rows 0-1 of R(q) at the regression points (ii),
        block means of rows 0-1 of R(q) (rf), the 1st percentile of |g|/9.81 over valid slots (gn)."""
        gsm = self.gs; N = self.N; carries = np.empty((self.nch, 4)); xs = np.zeros((gsm.K, 3)); prev = np.array([0, 0, 0, 1.0])
        def s1(ci, carry):                                              # strapdown + block sums of the rotated specific force
            c0 = ci * CH; c1 = min(N, c0 + CH); qg = self.strap(c0, c1, bias, carry); gsm.accum(xs, c0, qrot(qg, self.af(c0, c1)), self.valid[c0:c1]); return qg[-1].copy()
        for ci in range(self.nch): carries[ci] = prev; prev = s1(ci, prev)
        xs /= np.maximum(gsm.cnt, 1)[:, None]
        if sub is not None: xs -= sub                                   # X001: motion acceleration removed before the gravity smoother
        sb = gsm.smooth(xs, 1.5, inplace=True); att = Att(self, bias, carries, sb); out = {}
        if not (ii or rf or gn): return att, out
        I = self.ii; J = I + self.step; nI = len(I); t = self.t
        if ii: Y = np.empty((nI, 2)); A5 = np.empty((nI, 2, 3)) if aq else None
        if rf: rs = np.zeros((gsm.K, 6))
        if gn: nv = int(self.valid.sum()); vq = (nv - 1) * np.true_divide(1, 100); keep = min(nv, int(np.floor(vq)) + 2)
        st = dict(p0=0, pend=np.zeros((0, 4)), small=np.zeros(0))
        def s2(ci):                                                     # tilt correction; collections at ii, block means, |g|
            c0, c1, q, qc, g = att.chunk(ci)
            if ii:                                                      # y_k = tilt drift between I_k and J_k = I_k + 1 s
                a, b = np.searchsorted(I, [c0, c1]); avail = np.concatenate([st["pend"], qc[I[a:b] - c0]], 0); p0 = st["p0"]
                if aq: A5[a:b] = rot_rows01(q[I[a:b] - c0])
                aJ, bJ = np.searchsorted(J, [c0, c1])
                if bJ > aJ: Y[aJ:bJ] = -(qlog(qmul(qc[J[aJ:bJ] - c0], qinv(avail[aJ - p0:bJ - p0]))) / (t[J[aJ:bJ]] - t[I[aJ:bJ]])[:, None])[:, :2]
                st["pend"] = avail[bJ - p0:].copy(); st["p0"] = bJ
            if rf: gsm.accum(rs, c0, rot_rows01(q).reshape(-1, 6), self.valid[c0:c1])
            if gn:
                sm = np.concatenate([st["small"], g[self.valid[c0:c1]]])
                if len(sm) > keep: sm = np.partition(sm, keep - 1)[:keep]
                st["small"] = sm.copy()
        for ci in range(self.nch): s2(ci)
        if ii:
            out["y"] = Y
            if aq: out["A"] = A5
        if rf: rs /= np.maximum(gsm.cnt, 1)[:, None]; out["rf"] = rs
        if gn: out["h"] = float(_percentile_small(st["small"], nv, 1) / 9.81)
        return att, out
    def attitude5b(self, bias=None, fit="const", ridge=1e-1, win=60.0, knot=20.0, iters=2):
        t = self.t; bias = Bias() if bias is None else bias; n_it = iters if fit else 0; hmin = 1.0
        for it in range(n_it + 1):
            last = it == n_it; att, o = self.attitude(bias, ii=not last and len(self.ii) >= 50, gn=last)
            if last: hmin = o["h"]; break
            ii = self.ii
            if len(ii) < 50: break
            A = o["A"]; y = o["y"]
            if fit == "const":
                A2 = A.reshape(-1, 3); bias = bias.plus(("const", np.linalg.solve(A2.T @ A2 + len(A2) * ridge * np.eye(3), A2.T @ y.reshape(-1))))
            else:
                tm = t[ii] + 0.5 * self.step / self.fs; knots = np.arange(t[0], t[self.N - 1] + knot, knot); db = np.zeros((len(knots), 3))
                for j, tk in enumerate(knots):
                    wg = np.clip(1 - np.abs(tm - tk) / win, 0, None); sel = wg > 0
                    if sel.sum() < 20: continue
                    Aw = A[sel] * wg[sel][:, None, None]; db[j] = np.linalg.solve(np.einsum("nij,nik->jk", Aw, A[sel]) + 2 * wg[sel].sum() * ridge * np.eye(3), np.einsum("nij,ni->j", Aw, y[sel]))
                bias = bias.plus(("knots", knots, db))
        return att, bias, hmin
    def attitude6(self, bias, knot=20.0, lam2=3.0, lam0=0.05, iters=6, tol=3e-4):
        """Gyro-bias increments as a linear spline in time, penalised least squares on the drift of the tilt correction with the
        regressors passed through the same gravity filter and 1 s difference as the response (fbs5.attitude6)."""
        t = self.t; N = self.N; st = self.step; done = False; gsm = self.gs; t0, t1 = float(t[0]), float(t[N - 1])
        for it in range(iters + 1):
            fin = it == iters or done; att, o = self.attitude(bias, ii=not fin and len(self.ii) >= 50, rf=not fin and len(self.ii) >= 50, aq=False)
            if fin: break
            ii = self.ii
            if len(ii) < 50: break
            y = o["y"]; tm = t[ii] + 0.5 * st / self.fs; n = len(ii); Rs = gsm.smooth(o["rf"], 1.5 / 9.81, inplace=True); del o
            K = _basis_idx(tm[:1], t0, t1, knot)[0]; J = ii + st; B = {}         # normal equations M'M, M'y of the reference's design
            def seqbin(key, bins, w, nb):                               # matrix, accumulated chunk by chunk in the order of a
                prev = B.get(key); bins = bins if prev is None else np.concatenate([np.arange(nb), bins])   # single np.bincount
                B[key] = np.bincount(bins, weights=w if prev is None else np.concatenate([prev, w]), minlength=nb)
            def creg(c0, acc, pend):                                    # A = (cs[ii + st] - cs[ii]) / st, cs = cumulative sums of the
                c1 = min(N, c0 + CH); cs = np.concatenate([acc[None], gsm.to_slots(Rs, c0, c1)], 0).cumsum(0)   # smoothed regressors
                a, b = np.searchsorted(ii, [c0, c1]); p0, P = pend; P = np.concatenate([P, cs[ii[a:b] - c0]], 0)  # cs[j] = sum < c0 + j
                aJ, bJ = np.searchsorted(J, [c0, c1])
                if bJ > aJ:
                    A = ((cs[J[aJ:bJ] - c0] - P[aJ - p0:bJ - p0]) / st).reshape(-1, 2, 3); yy = y[aJ:bJ]; _K, i0, w0, i1, w1 = _basis_idx(tm[aJ:bJ], t0, t1, knot)
                    for j in range(3):
                        rj = A[:, 0, j] * yy[:, 0] + A[:, 1, j] * yy[:, 1]
                        for q_, (ia, wa) in enumerate(((i0, w0), (i1, w1))): seqbin(("r", j, q_), ia, wa * rj, K)
                        for l in range(3):
                            g = A[:, 0, j] * A[:, 0, l] + A[:, 1, j] * A[:, 1, l]
                            for q_, (ia, wa) in enumerate(((i0, w0), (i1, w1))):
                                for r_, (ib, wb) in enumerate(((i0, w0), (i1, w1))): seqbin(("n", j, l, q_, r_), ia * K + ib, wa * wb * g, K * K)
                return cs[-1].copy(), (bJ, P[bJ - p0:].copy())
            acc = np.zeros(6); pend = (0, np.zeros((0, 6)))
            for c0 in range(0, N, CH): acc, pend = creg(c0, acc, pend)
            del Rs, pend; Nm = np.zeros((3 * K, 3 * K)); rhs = np.zeros(3 * K)
            for j in range(3):
                for q_ in range(2): rhs[j::3] += B[("r", j, q_)]
                for l in range(3):
                    for q_ in range(2):
                        for r_ in range(2): Nm[j::3, l::3] += B[("n", j, l, q_, r_)].reshape(K, K)
            del B
            wbar = n / K; Pn = lam0 * wbar * np.eye(3 * K)
            if K > 2: D2 = np.diff(np.eye(K), 2, axis=0); Pn = Pn + lam2 * wbar * np.kron(D2.T @ D2, np.eye(3))
            beta = np.linalg.solve(Nm + Pn, rhs).reshape(K, 3); bias = bias.plus(("spline", t0, t1, knot, beta)); mx = 0.0
            for c0 in range(0, N, CH): mx = max(mx, float(np.abs(_basis_eval(t[c0:min(N, c0 + CH)], t0, t1, knot, beta)).max()))
            done = mx < tol
        return att, bias

def coarse_bias(D, F, kp=1.0, ki=0.3, smooth=5.0):
    """Average of a causal and an anti-causal PI complementary filter's bias states, low-passed at `smooth` s (fbs5.coarse_bias)."""
    N = D.N0; dt = D.m0; B = np.empty((N, 3)); v = D.valid[:N]; chunks = [(c0, min(N, c0 + CH)) for c0 in range(0, N, CH)]
    st = dict(q=(0.0, 0.0, 0.0, 1.0), b=[0.0, 0.0, 0.0], init=False)
    for c0, c1 in chunks: B[c0:c1] = _mahony(np.nan_to_num(F.wf(c0, c1)), np.nan_to_num(F.af(c0, c1)), v[c0:c1], dt, kp, ki, st)
    st = dict(q=(0.0, 0.0, 0.0, 1.0), b=[-0.0, -0.0, -0.0], init=False)
    for c0, c1 in chunks[::-1]:
        bb = _mahony(-np.nan_to_num(F.wf(c0, c1))[::-1], np.nan_to_num(F.af(c0, c1))[::-1], v[c0:c1][::-1], dt, kp, ki, st); B[c0:c1] = 0.5 * (B[c0:c1] - bb[::-1])
    for j in range(3): B[:, j] = zerophase(B[:, j], 1.0 / dt, 1.0 / (2 * np.pi * smooth))
    return B
def _mahony(w, a, valid, dt, kp, ki, st):
    """Causal PI complementary filter with integral (bias) feedback; returns the bias state at every sample (state carried in st)."""
    N = len(w); qx, qy, qz, qs = st["q"]; b = st["b"]; init = st["init"]; B = np.empty((N, 3))
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
    st["q"] = (qx, qy, qz, qs); st["init"] = init; return B

# =============================================================================================== M11-M13: back-end measurements
def _bs1(x, d):
    pad = (-len(x)) % d
    if pad: x = np.concatenate([x, np.zeros(pad)])
    return x.reshape(-1, d).sum(1)
def _bs2(x, d):
    pad = (-len(x)) % d
    if pad: x = np.concatenate([x, np.zeros((pad,) + x.shape[1:])])
    return x.reshape((-1, d) + x.shape[1:]).sum(1)
class Meas:
    """Block-mean measurements of the position smoothers on the extended time line (blocks of d slots; fbs4.build_joint /
    block_average as ported in the reference): world specific force (accelerometer rows), zero-velocity rows, accelerometer
    bias variance at gap ends, and for the rotor model the drag rows (thrust frame, flight mask, lever-arm terms)."""
    def __init__(self, D, F, att, d, rotor, k_drag):
        N0 = D.N0; Ne = D.N; t = D.t; v = D.valid[:N0]; self.d = d; K = (Ne + d - 1) // d; self.K = K; self.rotor = rotor
        bt = BlockTimes(t, d); self.bt = bt; us = []                    # block times (fbs4._grid, on demand) and discretisation classes
        dcl = lambda b0: np.round(np.diff(bt.get(b0, min(K, b0 + CH + 1))) / 1e-5).astype(np.int64)
        for b0 in range(0, K - 1, CH): us.append(_uniq(dcl(b0)))
        self.u = _uniq(np.concatenate(us)) if us else np.zeros(0, np.int64); del us; self.ix = np.zeros(K, np.int16 if len(self.u) < 32000 else np.int64)
        for b0 in range(0, K - 1, CH): self.ix[b0 + 1:b0 + 1 + min(CH, K - 1 - b0)] = np.searchsorted(self.u, dcl(b0))
        nv = int(v.sum()); acc = dict(aw=np.zeros(3), z=np.zeros(3), last=None)   # sweep 1: means and flight vibration
        if rotor: eg = min(9, Ne - 1); vbuf = np.zeros(Ne + 2 * eg); vib = vbuf[eg:eg + Ne]
        def sweep1(ci):
            c0 = ci * CH; c1 = min(N0, c0 + CH); q = att.q(ci); af = F.af(c0, c1); vv = v[c0:c1]; acc["aw"] = _seqsum(acc["aw"], qrot(q, af)[vv])
            if rotor:
                acc["z"] = _seqsum(acc["z"], af[vv]); vib[c0:c1] = np.sum(np.diff(np.concatenate([af[:1] if acc["last"] is None else acc["last"], af], 0), axis=0) ** 2, 1); acc["last"] = af[-1:].copy()
        for ci in range(F.nch): sweep1(ci)
        self.mean_aw = acc["aw"] / nv
        if rotor:
            vib = zerophase_inplace(vbuf, Ne, D.fs_e, 1.0); np.clip(vib, 0, None, out=vib); np.sqrt(vib, out=vib); fl = vib > 0.5 * _median(vib); del vib, vbuf
            zb, e1, e2 = _thrust_frame(acc["z"] / nv); self.e = (e1, e2); md = v & fl[:N0]; del fl; cd = np.zeros(K, np.int8)
            if CFG["drag_off_rest"]: md &= ~D.zmask(0, N0)                 # no drag rows where the platform is detected at rest
            for c0 in range(0, N0, CH):                                 # flight samples per block; drag rows are kept for flight blocks only
                m = min(N0, c0 + CH) - c0; cd[c0 // d:c0 // d + (m + d - 1) // d] = _bs1(md[c0:c0 + m].astype(float), d)
            fb = np.flatnonzero(cd); nf = len(fb); fidx = np.full(K, -1, np.int32); fidx[fb] = np.arange(nf, dtype=np.int32); del fb
            HV = np.zeros((nf, 2, 3)); YD = np.zeros((nf, 2))
        ca = np.zeros(K, np.int8); cz = np.zeros(K, np.int8); ya = np.zeros((K, 3)); st2 = dict(s1=0.0, s2=0.0, nd=0, last=None, loc=[])
        def sweep2(c0):                                                 # block sums (blocks never straddle chunks)
            c1 = min(Ne, c0 + CH); n = c1 - c0; b0 = c0 // d; nb = (n + d - 1) // d; m = max(0, min(N0, c1) - c0)
            vv = np.zeros(n, bool); vv[:m] = v[c0:c0 + m]; zz = np.zeros(n, bool); zz[:m] = D.zmask(c0, c0 + m)
            ca[b0:b0 + nb] = _bs1(vv.astype(float), d); cz[b0:b0 + nb] = _bs1(zz.astype(float), d)
            if not m: return
            q = att.q(c0 // CH)[:m]; af = F.af(c0, c0 + m); aw = np.zeros((n, 3)); aw[:m] = qrot(q, af) - self.mean_aw
            ya[b0:b0 + nb] = _bs2(aw * vv[:, None], d); av = aw[:m][vv[:m]]
            if len(av):
                lastv = st2["last"]; dv = np.diff(av if lastv is None else np.concatenate([lastv, av], 0), axis=0)
                st2["s1"] += float(dv.sum()); st2["s2"] += float((dv * dv).sum()); st2["nd"] += dv.size; st2["last"] = av[-1:].copy()
                if CFG["het_acc"] and dv.size > 30: st2["loc"].append((c0 + 0.5 * m, (float((dv * dv).mean()) - float(dv.mean()) ** 2) / 2.0))
            del aw, av
            if not rotor: return
            fi = fidx[b0:b0 + nb]; sel = fi >= 0
            if not sel.any(): return
            fj = fi[sel]; mf = md[c0:c0 + m]                            # the drag terms that need q or the accelerometer
            for i, e in enumerate(self.e):
                h = np.zeros((n, 3)); h[:m] = -k_drag * qrot(q, np.broadcast_to(e, (m, 3))) * mf[:, None]; HV[fj, i] = _bs2(h, d)[sel]; del h
                h1 = np.zeros(n); h1[:m] = (af @ e) * mf; YD[fj, i] = _bs1(h1, d)[sel]
        for c0 in range(0, Ne, CH): sweep2(c0)
        if not getattr(D, "keepA", False): D.A = None                 # accelerometer done (reloaded only if the guard fires)
        if rotor:                                                       # lever-arm terms of the drag rows (gyro only)
            HL = np.zeros((nf, 2, 3)); U = np.eye(3); wcf = lambda a, b: F.wf(a, b) - att.bias(t, a, b)
            def sweep3(c0):
                c1 = min(Ne, c0 + CH); n = c1 - c0; b0 = c0 // d; nb = (n + d - 1) // d; m = max(0, min(N0, c1) - c0)
                fi = fidx[b0:b0 + nb]; sel = fi >= 0
                if not m or not sel.any(): return
                fj = fi[sel]; mf = md[c0:c0 + m]
                lo = max(0, c0 - 1); hi = min(N0, c0 + m + 1); wl = zp_range(wcf, N0, D.fs, 15.0, lo, hi)
                if c0 + m + 1 > N0: wl = np.concatenate([wl, wl[-1:]], 0)    # the blind-window extension repeats the last row
                wd = np.gradient(wl, axis=0) * D.fs_e; o = c0 - lo; wl, wd = wl[o:o + m], wd[o:o + m]
                for i, e in enumerate(self.e):
                    for j in range(3):
                        u_ = U[j]; h1 = np.zeros(n); h1[:m] = ((np.cross(wd, u_) + np.cross(wl, np.cross(wl, u_))) @ e) * mf; HL[fj, i, j] = _bs1(h1, d)[sel]
            for c0 in range(0, Ne, CH): sweep3(c0)
            del md
        s1, s2, nd = st2["s1"], st2["s2"], st2["nd"]
        for b0 in range(0, K, CH): ya[b0:b0 + CH] /= np.maximum(ca[b0:b0 + CH], 1).astype(float)[:, None]
        self.ya = ya; self.ca = ca; self.cz = cz; del ya
        self.sn2 = ((s2 / nd - (s1 / nd) ** 2) / 2.0) if nd else 0.0     # hf_std(aw[valid]) ** 2
        self.tabA = _bs_table(self.sn2, d); self.tabZ = _bs_table(SZ ** 2, d)    # block noise variances follow from the counts
        self.ra = None; self.va = {}
        if CFG["vel_aid_dir"]:                                          # X060: velocity rows from a side file of learned predictions
            fn = os.path.join(CFG["vel_aid_dir"], os.path.basename(D.src.path).rsplit(".", 1)[0] + ".npz")
            if os.path.exists(fn):
                Z = np.load(fn); sl = Z["slot"].astype(np.int64); vb = Z["v_body"]; var = np.maximum(np.exp(Z["logvar"]).mean(1) * CFG["vel_aid_scale"] ** 2, CFG["vel_aid_floor"] ** 2)
                for i in range(len(sl)):
                    s_ = int(sl[i])
                    if 0 <= s_ < N0: q_ = att.q(s_ // CH)[s_ - (s_ // CH) * CH]; self.va[s_ // d] = (qrot(q_[None], vb[i][None])[0], float(var[i]))
        if CFG["het_acc"] and len(st2["loc"]) >= 2 and self.sn2 > 0:            # X005: per-block scale of the accelerometer variance
            cs_ = np.array([x[0] for x in st2["loc"]]); vs_ = np.array([x[1] for x in st2["loc"]]) / self.sn2
            self.ra = np.clip(np.interp(np.arange(K) * d + 0.5 * d, cs_, vs_), CFG["het_lo"], CFG["het_hi"]).astype(np.float32)
        ve_end = np.flatnonzero(~v[1:] & v[:-1]) + 1; ve_st = np.flatnonzero(v[1:] & ~v[:-1]) + 1; self.bk = {}
        if len(ve_st):                                                  # 0.5^2 min(1, T / 2 s) at the end of every gap, block maximum
            j = np.searchsorted(ve_end, ve_st) - 1; T = np.where(j >= 0, t[ve_st] - t[ve_end[np.maximum(j, 0)] - 1], CFG["gap_ref"])
            for blk, b in zip((ve_st // d).tolist(), (BIAS_JUMP ** 2 * np.minimum(1.0, T / CFG["gap_ref"])).tolist()): self.bk[blk] = max(self.bk.get(blk, 0.0), b)
        if rotor:
            fb = np.flatnonzero(cd)                                     # the flight blocks (count >= 1), in the order of the compact rows
            for a in range(0, nf, CH):
                c1d = cd[fb[a:a + CH]].astype(float); HV[a:a + CH] /= c1d[:, None, None]; HL[a:a + CH] /= c1d[:, None, None]; YD[a:a + CH] /= c1d[:, None]
            del fb; self.HV = HV; self.HL = HL; self.YD = YD; self.cd = cd; self.fidx = fidx; self.tabD = _bs_table(SD_DRAG ** 2, d)
    @staticmethod
    def R(tab, c):
        """Block noise variance: (block sum of the per-sample variance) / count / count, as the reference divides."""
        c1 = float(max(c, 1)); return tab[c] / c1 / c1
def _thrust_frame(zm):
    zb = zm / np.linalg.norm(zm); h = np.array([0, 1.0, 0]) if abs(zb[1]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(zb, h); e1 /= np.linalg.norm(e1); return zb, e1, np.cross(zb, e1)
def _bs_table(val, d):
    """Block sum of where(mask, val, 0) over d slots for 0..d masked slots (NumPy's block sum of equal terms)."""
    return [float(np.array([val] * k + [0.0] * (d - k)).reshape(1, d).sum(1)[0]) for k in range(d + 1)]

# =============================================================================================== M14-M17: smoothers, guard, blind
def _rts(K, x0, P0, fwd, take):
    """Kalman filter + RTS smoother with sqrt(K) checkpoints: the forward pass keeps one (x, P) per chunk; each chunk is re-run
    forward and smoothed backward (xs_k = xf_k + Pf_k A' Pp_{k+1}^-1 (xs_{k+1} - xp_{k+1})); take(k, xs) stores the output.
    On long inputs (K > 8192) and if fwd.pred exists (the prediction step that fwd itself uses), the re-run keeps only the
    filtered states and the backward pass recomputes the predictions from them (identical values, half the memory per step)."""
    ck = max(16, int(math.sqrt(K))); cps = []; x, P = x0, P0; pred = getattr(fwd, "pred", None) if K > 8192 else None
    for k in range(K):
        if k % ck == 0: cps.append((x, P))
        _, _, x, P = fwd(x, P, k)
    xs = None; nxt = None
    for c in range(len(cps) - 1, -1, -1):
        c0 = c * ck; c1 = min(K, c0 + ck); x, P = cps[c]; XP = []; PP = []; XF = []; PF = []
        for k in range(c0, c1):
            xp, Pp, x, P = fwd(x, P, k); XF.append(x); PF.append(P)
            if pred is None: XP.append(xp); PP.append(Pp)
        del xp, Pp
        for j in range(c1 - c0 - 1, -1, -1):
            k = c0 + j
            if k == K - 1: xs = XF[j]
            else:
                if pred is not None: xpn, Ppn = pred(XF[j], PF[j], k + 1)
                else: xpn, Ppn = (XP[j + 1], PP[j + 1]) if j < c1 - c0 - 1 else nxt
                xs = XF[j] + PF[j] @ (fwd.A(k + 1).T @ np.linalg.solve(Ppn, xs - xpn))
            take(k, xs)
        nxt = (XP[0], PP[0]) if pred is None else None; del XP, PP, XF, PF
def _disc(F, Qc, u):
    A = []; Q = []
    for dd in u.tolist():
        a, q = van_loan(F, Qc, max(int(dd), 0) * 1e-5); A.append(a); Q.append(q)
    return A, Q
def smooth_hand(M, inplace=False):
    """Matern-5/2 prior per world axis (the three axes share one 4 x 4 covariance), accelerometer + zero-velocity rows.
    Returns p_k (K, 3), v_k (K, 3) at the block times M.bt."""
    F4, Qc4, P04, H4 = model_matern52(*MATERN); As, Qs = _disc(F4, Qc4, M.u); ix = M.ix; K = M.K; R = Meas.R
    ya, ca, cz, bk, tA, tZ = M.ya, M.ca, M.cz, M.bk, M.tabA, M.tabZ; h = H4; hz = np.array([0, 1.0, 0, 0]); Hb = np.vstack([h, hz]); z3 = np.zeros(3); ra = M.ra; va = M.va
    def fwd(x, P, k):
        if k:
            A = As[ix[k]]; xp = A @ x; Pp = A @ P @ A.T + Qs[ix[k]]; b = bk.get(k, 0.0)
            if b > 0: Pp[3, 3] += b
        else: xp, Pp = x, P
        c_a = int(ca[k]); c_z = int(cz[k]); a_, z_ = c_a > 0, c_z > 0; zv = z3; rz = R(tZ, c_z) if z_ else 0.0
        if not z_ and k in va: zv, rz = va[k]; z_ = True
        if not (a_ or z_): return xp, Pp, xp, 0.5 * (Pp + Pp.T)
        if a_ and not z_:
            PH = Pp @ h; g = PH / (h @ PH + (R(tA, c_a) if ra is None else R(tA, c_a) * float(ra[k]))); xn = xp + np.outer(g, ya[k] - h @ xp); Pn = Pp - np.outer(g, PH)
        elif z_ and not a_:
            PH = Pp @ hz; g = PH / (hz @ PH + rz); xn = xp + np.outer(g, zv - hz @ xp); Pn = Pp - np.outer(g, PH)
        else:
            PH = Pp @ Hb.T; S = Hb @ PH; S[0, 0] += R(tA, c_a) if ra is None else R(tA, c_a) * float(ra[k]); S[1, 1] += rz; Gt = np.linalg.solve(S, PH.T); Pn = Pp - PH @ Gt
            xn = xp + Gt.T @ (np.vstack([ya[k], zv]) - Hb @ xp)
        return xp, Pp, xn, 0.5 * (Pn + Pn.T)
    fwd.A = lambda k: As[ix[k]]; pk = ya if inplace else np.empty((K, 3)); vk = np.empty((K, 3))   # ya[k] is read before pk[k] is written
    def take(k, xs): pk[k] = xs[0]; vk[k] = xs[1]
    _rts(K, np.zeros((4, 3)), P04.copy(), fwd, take); return pk, vk
def smooth_rotor(M, drag=None, ll_only=False):
    """17-state rotor model: spring-damper prior + OU bias per world axis, 2 OU drag biases, 3 lever-arm states; accelerometer,
    rotor-drag (in flight) and zero-velocity rows. Returns p_k, v_k. drag = (sc, tauc, sc0) overrides the drag-bias prior;
    ll_only = True runs the forward filter only and returns its innovation log-likelihood (Phase 5, X025)."""
    F, Qc, P0, H4 = model_rotor(*(drag or (None, None, None))); n = 17; As, Qs = _disc(F, Qc, M.u); ix = M.ix; K = M.K; R = Meas.R
    ya, HV, HL, YD, fidx, bk, ca, cd, cz, tA, tD, tZ = M.ya, M.HV, M.HL, M.YD, M.fidx, M.bk, M.ca, M.cd, M.cz, M.tabA, M.tabD, M.tabZ
    H = np.zeros((8, n))
    for j in range(3): H[j, 4 * j:4 * j + 4] = H4; H[5 + j, 4 * j + 1] = 1.0
    H[3, 12] = 1.0; H[4, 13] = 1.0; rows3 = (np.array([3, 3, 3]), np.array([1, 5, 9])); rows4 = (np.array([4, 4, 4]), np.array([1, 5, 9]))
    def pred(x, P, k):
        A = As[ix[k]]; xp = A @ x; Pp = A @ P @ A.T + Qs[ix[k]]; b = bk.get(k, 0.0)
        if b > 0: Pp[3, 3] += b; Pp[7, 7] += b; Pp[11, 11] += b
        return xp, Pp
    def fwd(x, P, k):
        xp, Pp = pred(x, P, k) if k else (x, P)
        c_a = int(ca[k]); c_d = int(cd[k]); c_z = int(cz[k]); aid = (not c_z) and k in M.va
        if not (c_a or c_d or c_z or aid): return xp, Pp, xp, 0.5 * (Pp + Pp.T)
        m = 8 if (c_z or aid) else 5; Hk = H[:m].copy(); z = np.zeros(m); r = np.ones(m)
        if aid: z[5:8] = M.va[k][0]
        if c_a: z[:3] = ya[k]; r[:3] = R(tA, c_a) if M.ra is None else R(tA, c_a) * float(M.ra[k])
        else: Hk[:3] = 0.0
        if c_d:
            j = int(fidx[k]); Hk[rows3] = HV[j, 0]; Hk[rows4] = HV[j, 1]; Hk[3, 14:] = HL[j, 0]; Hk[4, 14:] = HL[j, 1]; z[3:5] = YD[j]; r[3:5] = R(tD, c_d)
        else: Hk[3:5] = 0.0
        if m == 8: r[5:] = R(tZ, c_z) if c_z else M.va[k][1]
        PH = Pp @ Hk.T; S = Hk @ PH; S.flat[::m + 1] += r; Gt = np.linalg.solve(S, PH.T); Pn = Pp - PH @ Gt; xn = xp + (z - Hk @ xp) @ Gt
        return xp, Pp, xn, 0.5 * (Pn + Pn.T)
    fwd.A = lambda k: As[ix[k]]; fwd.pred = pred
    if ll_only:
        x, P = np.zeros(n), P0; ll = 0.0
        for k in range(K):
            xp, Pp = pred(x, P, k) if k else (x, P)
            c_a = int(ca[k]); c_d = int(cd[k]); c_z = int(cz[k])
            if not (c_a or c_d or c_z): x, P = xp, 0.5 * (Pp + Pp.T); continue
            m = 8 if c_z else 5; Hk = H[:m].copy(); z = np.zeros(m); r = np.ones(m)
            if c_a: z[:3] = ya[k]; r[:3] = R(tA, c_a)
            else: Hk[:3] = 0.0
            if c_d:
                j = int(fidx[k]); Hk[rows3] = HV[j, 0]; Hk[rows4] = HV[j, 1]; Hk[3, 14:] = HL[j, 0]; Hk[4, 14:] = HL[j, 1]; z[3:5] = YD[j]; r[3:5] = R(tD, c_d)
            else: Hk[3:5] = 0.0
            if m == 8: r[5:] = R(tZ, c_z)
            use = np.flatnonzero(Hk.any(1)); Hk = Hk[use]; z = z[use]; r = r[use]
            PH = Pp @ Hk.T; S = Hk @ PH; S.flat[::len(use) + 1] += r; nu = z - Hk @ xp; L = np.linalg.cholesky(S); w_ = np.linalg.solve(L, nu)
            ll += -0.5 * float(w_ @ w_) - float(np.log(np.diag(L)).sum()) - 0.9189385332046727 * len(use)
            Gt = np.linalg.solve(S, PH.T); Pn = Pp - PH @ Gt; x = xp + nu @ Gt; P = 0.5 * (Pn + Pn.T)
        return ll
    pk = np.empty((K, 3)); vk = np.empty((K, 3)); TS = np.empty((K, n)) if TRACE is not None else None
    def take(k, xs):
        pk[k] = xs[[0, 4, 8]]; vk[k] = xs[[1, 5, 9]]
        if TS is not None: TS[k] = xs
    _rts(K, np.zeros(n), P0, fwd, take)
    if TS is not None: TRACE["rotor_states"] = TS
    return pk, vk
class BlockTimes:
    """Times of the smoother grid (fbs4._grid): the mean slot time of each block of d slots, the last block padded with
    extrapolated times. Kept as an array up to 2^15 blocks, beyond that computed on demand for any block range (same values)."""
    def __init__(self, t, d):
        self.t = t; self.d = d; self.N = len(t); self.K = (self.N + d - 1) // d; self.arr = None
        if self.K <= 1 << 15: self.arr = self.get(0, self.K)            # small: kept (the same values either way)
        self.first = self.get(0, 1)[0]; self.last = self.get(self.K - 1, self.K)[0]
    def get(self, b0, b1):
        if self.arr is not None: return self.arr[b0:b1]
        d = self.d; t = self.t; s1 = min(self.N, b1 * d); tt = t[b0 * d:s1]; pad = b1 * d - s1
        if pad: tt = np.concatenate([tt, t[-1] + (t[-1] - t[-2]) * np.arange(1, pad + 1)])
        return tt.reshape(-1, d).mean(1)
def hermite(bt, pk, vk, s):
    """fbs4.hermite: cubic Hermite of the block-rate (p, v) at the times of slots s; the block times around s are computed
    locally (each slot lies between the centres of its own block's neighbours, so the bracketing pair is always among them)."""
    s = np.asarray(s, np.int64); K = bt.K
    if len(s) == 0: return np.zeros((0, 3))
    t = bt.t[s]; b0 = max(0, int(s.min()) // bt.d - 1); b1 = min(K, int(s.max()) // bt.d + 2); tk = bt.get(b0, b1)
    i = np.clip(np.searchsorted(tk, t) + b0 - 1, 0, K - 2); j = i - b0; h = (tk[j + 1] - tk[j])[:, None]; u = ((t - tk[j]) / (tk[j + 1] - tk[j]))[:, None]
    u2, u3 = u * u, u * u * u; out = (2 * u3 - 3 * u2 + 1) * pk[i] + (u3 - 2 * u2 + u) * h * vk[i] + (-2 * u3 + 3 * u2) * pk[i + 1] + (u3 - u2) * h * vk[i + 1]
    lo = t < bt.first; hi = t > bt.last; out[lo] = pk[0] + (t[lo] - bt.first)[:, None] * vk[0]; out[hi] = pk[-1] + (t[hi] - bt.last)[:, None] * vk[-1]; return out
def radius(bt, pk, vk):
    """RMS distance from the mean position of the full-rate Hermite trajectory (fbs6.radius), streamed."""
    n = bt.N; s = np.zeros(3)
    for c0 in range(0, n, CH): s = _seqsum(s, hermite(bt, pk, vk, np.arange(c0, min(n, c0 + CH))))
    mu = s / n; acc = 0.0
    for c0 in range(0, n, CH): acc += float((((hermite(bt, pk, vk, np.arange(c0, min(n, c0 + CH))) - mu) ** 2).sum(1)).sum())
    return math.sqrt(acc / n)

RAD = {}                                                            # guard radii of the last estimate (diagnostics, info)
def accel_sub(F, att, bt, pk, vk, gain):
    """X001: the smoother's world acceleration (derivative of the block velocity) at the gravity-smoother block centres, rotated
    into the gyro-integrated frame by the inverse tilt correction of the current gravity solution."""
    gsm = F.gs; tk = bt.get(0, bt.K); a = np.gradient(vk, tk, axis=0); tq = gsm.tb
    aW = np.stack([np.interp(tq, tk, a[:, j]) for j in range(3)], 1); del a
    g = att.sb / np.linalg.norm(att.sb, axis=1, keepdims=True); qc = shortest_arc(g, np.broadcast_to(EZ, g.shape))
    return gain * qrot(qinv(qc), aW)
class Solution:
    """Final trajectory: smoother positions (Hermite between blocks), attitude regenerated chunk-wise, blind-window rule."""
    def __init__(self, D, att, bt, pk, vk, is_rotor):
        self.D, self.att, self.bt, self.pk, self.vk, self.rot = D, att, bt, pk, vk, is_rotor; N0 = D.N0; t = D.t; self.cache = (-1, None)
        self.pT = hermite(bt, pk, vk, [N0 - 1])[0]; b = (N0 - 1) // bt.d; b0 = max(0, b - 1); b1 = min(bt.K, b + 2); tl = bt.get(b0, b1)
        self.vT = np.array([np.interp(t[N0 - 1], tl, vk[b0:b1, j]) for j in range(3)])   # = np.interp over all block times
        c = (N0 - 1) // CH; self.qL = self._qchunk(c)[N0 - 1 - c * CH].copy()
    def _qchunk(self, c):
        if self.cache[0] != c: self.cache = (None, None); self.cache = (c, self.att.q(c))
        return self.cache[1]
    def poses(self, s):
        """Positions and unit quaternions of slots s (int array, any order)."""
        D = self.D; N0 = D.N0; t = D.t; n = len(s); p = np.empty((n, 3)); q = np.empty((n, 4)); bl = s >= N0; inn = ~bl
        if inn.any():
            si = s[inn]; p[inn] = hermite(self.bt, self.pk, self.vk, si); qq = np.empty((len(si), 4)); ch = si // CH
            for c in _uniq(ch).tolist():
                m = ch == c; qq[m] = self._qchunk(c)[si[m] - c * CH]
            q[inn] = qq
        if bl.any():
            sb = s[bl]
            if self.rot: tb_ = CFG["blind_tau_r"]; tau = (sb - N0 + 1) / D.fs; p[bl] = self.pT + (tb_ * (1.0 - np.exp(-tau / tb_)))[:, None] * self.vT
            else: p[bl] = self.pT + (hermite(self.bt, self.pk, self.vk, sb) - self.pT)
            q[bl] = self.qL
        return p, q / np.linalg.norm(q, axis=1, keepdims=True)

# =============================================================================================== end to end
def estimate(src, k_drag=K_DRAG, force=None):
    """v5 / V5P4 on one input. Returns (Data, Solution, info)."""
    _mark("start"); RAD.clear(); D = Data(src); D.unit_check()
    if D.n_ok < 10 or D.n_distinct < 10: raise ValueError("fewer than 10 usable rows")
    D.medians(); _mark("parse+grid")
    ri = repair(D); _mark("repair")
    lock = thrust_lock_fill(D); anchors, zm = rest_info(D, win=CFG["rest_win"], frac=CFG["rest_frac"], gate=CFG["rest_gate"], min_windows=CFG["rest_minw"]); D.set_zmask(zm); del zm; _mark("lock+rest")
    rotor = lock < CFG["gate"] if force is None else force == "rotor"
    F = Front(D, CFG["tau_g_rotor"] if rotor else CFG["tau_g_hand"]); base = Bias(base=(np.array([x[0] for x in anchors]), np.array([x[1] for x in anchors]))) if anchors else None
    if base is None: att3, base, h = F.attitude5b(fit="const")
    else: att3, _b, h = F.attitude5b(bias=base, fit=None)
    Dst = float("nan"); coarse = False; _mark("front-const")
    def spline():
        if h < CFG["health"]: return F.attitude6(Bias(arr=coarse_bias(D, F)), knot=CFG["knot"], lam2=CFG["lam2"], iters=10), True
        return F.attitude6(base, knot=CFG["knot"], lam2=CFG["lam2"]), False
    if rotor: (att, bias), coarse = spline(); mode = "spline"
    else:
        if h < CFG["health"]: Dst = float("inf")
        else:
            _a4, b4, _ = F.attitude5b(bias=base, fit="knots", win=60.0, knot=20.0); del _a4; N = F.N; t = D.t; s = np.zeros(3)
            for c0 in range(0, N, CH): s = _seqsum(s, b4(t, c0, min(N, c0 + CH)) - base(t, c0, min(N, c0 + CH)))
            mu = s / N; acc = 0.0
            for c0 in range(0, N, CH): acc += float((((b4(t, c0, min(N, c0 + CH)) - base(t, c0, min(N, c0 + CH)) - mu) ** 2).sum(1)).sum())
            Dst = math.sqrt(acc / N); RAD["mu"] = float(np.linalg.norm(mu))
        if Dst > CFG["thr_hand"]: (att, bias), coarse = spline(); mode = "spline"
        else: att, bias, mode = att3, base, "const"
    if not rotor:                                                       # X046 statistics: gyro sample noise (first differences of valid raw samples)
        s1_ = 0.0; s2_ = 0.0; n_ = 0; Nf = F.N
        for c0 in range(0, Nf, CH):
            c1 = min(Nf, c0 + CH + 1); vv = D.valid[c0:c1]; ww = D.decode(c0, c1, "w"); ok_ = vv[1:] & vv[:-1]; dd = np.diff(ww, axis=0)[ok_]
            if len(dd): s1_ += float(dd.sum()); s2_ += float((dd * dd).sum()); n_ += dd.size
        RAD["sw"] = math.sqrt(max(s2_ / n_ - (s1_ / n_) ** 2, 0.0) / 2.0) if n_ else float("nan")
    safe_ = (CFG["final_mu_max"] is None or RAD.get("mu", 0.0) <= CFG["final_mu_max"]) and (CFG["final_sw_max"] is None or RAD.get("sw", 0.0) <= CFG["final_sw_max"])
    if (not rotor) and CFG["tau_g_final_hand"] is not None and (mode == "const" or not CFG["final_const_only"]) and safe_:
        RAD["final_long"] = True
        F.gs = GravitySmoother(D, CFG["gs_dec"], CFG["tau_g_final_hand"], CFG["k2"]); att, _o = F.attitude(bias); del _o   # X045
    del att3; F.ii = None; _mark("front-final")
    AC = CFG["acc_comp"] > 0; D.keepA = AC
    if not AC: F.gs.cache = {}; F.gs.cnt = F.gs.okb = F.gs.jblk = F.gs.jth = None
    _CO.clear(); Mz = Meas(D, F, att, CFG["dec_rotor"] if rotor else CFG["dec_hand"], rotor, k_drag)
    if not AC: D.A = None
    _mark("measurements"); guard = False; t = D.t
    if rotor:
        pkh, vkh = smooth_hand(Mz); rh = radius(Mz.bt, pkh, vkh); del pkh, vkh; _mark("guard-hand")
        dsel = tuple(CFG["drag_tight"]) if CFG["drag_adapt_r"] is not None and rh >= CFG["drag_adapt_r"] else None
        pk, vk = smooth_rotor(Mz, drag=dsel); rr = radius(Mz.bt, pk, vk); bt = Mz.bt; RAD.update(rh=rh, rr0=rr, tight=dsel is not None)
        loose_guard = CFG["guard_ref"] in ("loose", "revert") and rr > CFG["guard"] * max(rh, 1e-6)   # X110: the guard checks the loose solution first
        ratio_ok = CFG["switch_ratio"] is None or rr >= CFG["switch_ratio"] * max(rh, 1e-6)       # X112: the drag model must explain most of the workspace
        if CFG["switch_blend"] is not None and dsel is None and not loose_guard and ratio_ok:       # X111: soft switch between the loose and the steady solution
            lo_, hi_ = CFG["switch_blend"]; wb = min(1.0, max(0.0, (rr - lo_) / (hi_ - lo_))); RAD["blend"] = wb
            if wb > 0.0:
                pk2, vk2 = smooth_rotor(Mz, drag=tuple(CFG["drag_tight"])); RAD["tight"] = wb >= 1.0
                if wb >= 1.0: pk, vk = pk2, vk2
                else: pk = (1.0 - wb) * pk + wb * pk2; vk = (1.0 - wb) * vk + wb * vk2
                del pk2, vk2; rr = radius(Mz.bt, pk, vk)
        elif CFG["drag_switch_r"] is not None and dsel is None and not loose_guard and ratio_ok and rr >= CFG["drag_switch_r"]:     # X044: large workspace -> steady drag offset
            del pk, vk; pk, vk = smooth_rotor(Mz, drag=tuple(CFG["drag_tight"])); rr = radius(Mz.bt, pk, vk); RAD["tight"] = True
            if CFG["guard_ref"] == "revert" and rr > CFG["guard"] * max(rh, 1e-6):             # X110d: the steady solution fails the guard -> loose solution again
                del pk, vk; pk, vk = smooth_rotor(Mz, drag=None); rr = radius(Mz.bt, pk, vk); RAD["tight"] = False; RAD["reverted"] = True
        RAD["rr"] = rr; _mark("smoother-rotor")
        if loose_guard or (CFG["guard_ref"] == "final" and rr > CFG["guard"] * max(rh, 1e-6)):
            guard = True; del pk, vk; Mz = Meas(D, F, att, CFG["dec_hand"], False, k_drag)
            if not AC: D.A = None
    if not rotor or guard: pk, vk = smooth_hand(Mz, inplace=True); bt = Mz.bt; _mark("smoother-hand")
    is_r0 = rotor and not guard
    if AC and (CFG["acc_comp_rotor"] if is_r0 else CFG["acc_comp_hand"]):          # X001: acceleration-compensated gravity
        for _it in range(int(CFG["acc_comp"])):
            sub = accel_sub(F, att, bt, pk, vk, CFG["acc_comp_gain"]); del pk, vk
            att, _o = F.attitude(bias, sub=sub); del sub, _o
            Mz = Meas(D, F, att, CFG["dec_rotor"] if is_r0 else CFG["dec_hand"], is_r0, k_drag)
            if is_r0: pk, vk = smooth_rotor(Mz)
            else: pk, vk = smooth_hand(Mz, inplace=True)
            bt = Mz.bt
        _mark("acc-comp")
    if AC: F.gs.cache = {}; F.gs.cnt = F.gs.okb = F.gs.jblk = F.gs.jth = None; D.A = None
    if TRACE is None: Mz = None
    _CO.clear()
    if not (np.isfinite(pk).all() and np.isfinite(vk).all() and np.isfinite(att.sb).all()): raise FloatingPointError("non-finite smoother output")
    is_rotor = rotor and not guard; sol = Solution(D, att, bt, pk, vk, is_rotor)
    info = dict(rotor_gate=bool(rotor), rotor=bool(is_rotor), guard=guard, lock=lock, mode=mode, D=Dst, h=h, coarse=coarse, n_tail=D.nt, L=D.L,
                implausible=ri["implausible"], spikes_legacy=ri["spikes"], slots=D.N, **RAD)
    if TRACE is not None: TRACE.update(D=D, F=F, att=att, bias=bias, Mz=Mz, tk=bt.get(0, bt.K), pk=pk, vk=vk, anchors=anchors)
    return D, sol, info

def write_output(path, src, D, sol):
    """Second read of the input: one pose per row, in input order."""
    piece = 1024 if D.N <= 1 << 16 else 512                             # rows per renderer call (smaller on long inputs: less heap)
    with open(path, "wb") as f:
        for ts, s, _ in D.rows(False):
            for a in range(0, len(ts), piece): f.write(_rows(sol, s[a:a + piece], ts[a:a + piece]))
def _rows(sol, s, ts):
    p, q = sol.poses(s)
    if not (np.isfinite(p).all() and np.isfinite(q).all()): raise FloatingPointError("non-finite pose")
    return render(ts, p, q)
def write_motionless(path, src):
    """Fallback: zero positions, orientation from the mean specific force (levelled) or identity, one row per input row."""
    s = np.zeros(3); n = 0; tsc = 1; T = []
    try:
        for ts, v in iter_rows(src, True):
            ok = np.isfinite(v).all(1) & ~(v == 0).all(1)
            if ok.any(): s = _seqsum(s, v[ok, 3:]); n += int(ok.sum())
            if len(T) < 2: T.append(ts)
    except Exception: n = 0                                             # noqa: BLE001
    q = np.array([0, 0, 0, 1.0])
    if n:
        u = s / n; nu = np.linalg.norm(u)
        if nu > 1e-6: q = shortest_arc(u[None] / nu, EZ[None])[0]
    q = q / np.linalg.norm(q)
    with open(path, "wb") as f:
        try:
            for ts, _ in iter_rows(src, False): f.write(render(ts, np.zeros((len(ts), 3)), np.repeat(q[None], len(ts), 0)))
        except Exception as ex: log("could not re-read the input: %r" % (ex,))   # noqa: BLE001

# =============================================================================================== output
def _fmt_py(tn, P, Q):
    return "".join("%d.%09d %.6f %.6f %.6f %.7f %.7f %.7f %.7f\n" % ((a // 10 ** 9, a % 10 ** 9) + tuple(p) + tuple(q))
                   for a, p, q in zip(np.asarray(tn).tolist(), np.asarray(P).tolist(), np.asarray(Q).tolist())).encode()
def render(tn, P, Q, dp=6, dq=7):
    """'%d.%09d %.6f %.6f %.6f %.7f %.7f %.7f %.7f\\n' for every row, vectorised (fixed-point digits from integer arithmetic,
    pad bytes removed). Values whose scaled magnitude lies within 1e-6 of a rounding tie are rounded by Python's formatter,
    so the bytes equal %-formatting; blocks with |value| >= 999999999 or negative times use %-formatting directly."""
    N = len(tn)
    if N == 0: return b""
    tn = np.asarray(tn, np.int64)
    if tn.min() < 0 or not (np.abs(P).max() < 999999999.0 and np.abs(Q).max() < 999999999.0): return _fmt_py(tn, P, Q)   # 9 integer digits
    sec, ns = np.divmod(tn, 10 ** 9)
    def digits(v, width):
        out = np.zeros((N, width), np.uint8); x = v.copy()
        for j in range(width - 1, -1, -1): out[:, j] = 48 + (x % 10).astype(np.uint8); x //= 10
        nd = np.maximum(1, np.floor(np.log10(np.maximum(v, 1))).astype(int) + 1); out[np.arange(width)[None, :] < (width - nd)[:, None]] = 0; return out
    def fixed(x, dec):
        neg = np.signbit(x); pr = np.abs(x) * 10 ** dec; s = np.round(pr).astype(np.int64); fr = pr - np.floor(pr)
        for i in np.flatnonzero(np.abs(fr - 0.5) < 1e-6).tolist(): s[i] = int(("%.*f" % (dec, abs(float(x[i])))).replace(".", ""))
        ip, fp = np.divmod(s, 10 ** dec); F_ = np.zeros((N, dec), np.uint8); y = fp.copy()
        for j in range(dec - 1, -1, -1): F_[:, j] = 48 + (y % 10); y //= 10
        return np.hstack([np.where(neg, 45, 0).astype(np.uint8)[:, None], digits(ip, 9), np.full((N, 1), 46, np.uint8), F_])
    nsd = np.zeros((N, 9), np.uint8); y = ns.copy()
    for j in range(8, -1, -1): nsd[:, j] = 48 + (y % 10); y //= 10
    sp = np.full((N, 1), 32, np.uint8); parts = [digits(sec, 12), np.full((N, 1), 46, np.uint8), nsd]
    for j in range(3): parts += [sp, fixed(P[:, j], dp)]
    for j in range(4): parts += [sp, fixed(Q[:, j], dq)]
    parts.append(np.full((N, 1), 10, np.uint8)); buf = np.hstack(parts).ravel(); return buf[buf != 0].tobytes()

def resolve_output(arg):
    if arg is None: return os.path.join(os.getcwd(), "estimated_trajectory.txt")
    if os.path.isdir(arg) or arg.endswith(("/", os.sep)): os.makedirs(arg, exist_ok=True); return os.path.join(arg, "estimated_trajectory.txt")
    d = os.path.dirname(os.path.abspath(arg)); os.makedirs(d, exist_ok=True); return arg
def run(inp=None, out=None, k_drag=K_DRAG):
    """v5 -> hand-held branch only -> motionless: the output always has exactly one row per input row."""
    src = Source(inp); path = resolve_output(out)
    for force in (None, "hand"):
        try:
            D, sol, info = estimate(src, k_drag, force); write_output(path, src, D, sol); info["fallback"] = force or ""; _mark("output"); return path, info
        except Exception as ex:                                         # noqa: BLE001 - the output file must always be written
            log("estimation failed (%s branch): %r" % (force or "v5", ex))
    write_motionless(path, src); return path, dict(fallback="motionless")
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
