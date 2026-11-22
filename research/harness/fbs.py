"""fbs.py - Five Blind Seconds research harness.

SCIENTIFIC INSTRUMENT, NOT THE SUBMISSION. Ground truth is loaded only for evaluation and is never
passed to an estimator. Quaternions are xyzw (scipy convention) and rotate sensor -> world.
"""
import os, json, numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot, Slerp
from scipy.signal import butter, sosfiltfilt, savgol_filter

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
G = 9.81
EUROC = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult", "V1_01_easy",
         "V1_02_medium", "V1_03_difficult", "V2_01_easy", "V2_02_medium", "V2_03_difficult"]
TUMVI = ["room%d" % i for i in range(1, 7)]
ALL = EUROC + TUMVI
def group(n): return "TUM" if n.startswith("room") else n[:2]
def platform(n): return "handheld" if n.startswith("room") else "mav"
def seq_dir(n):
    return os.path.join(ROOT, "data/tumvi/dataset-%s_512_16/mav0" % n) if n.startswith("room") else os.path.join(ROOT, "data/euroc", n, "mav0")
def available(): return [n for n in ALL if os.path.exists(os.path.join(seq_dir(n), "imu0/data.csv"))]

class Seq: pass

def load(name):
    """Real IMU stream plus ground truth interpolated to the IMU stamps (evaluation only)."""
    d = seq_dir(name); s = Seq(); s.name = name
    imu = pd.read_csv(os.path.join(d, "imu0/data.csv"), comment="#", header=None)
    s.tn = imu[0].values.astype(np.int64); s.t = (s.tn - s.tn[0]) * 1e-9
    s.w = imu.iloc[:, 1:4].values.astype(float); s.a = imu.iloc[:, 4:7].values.astype(float)
    gfile = "mocap0/data.csv" if name.startswith("room") else "state_groundtruth_estimate0/data.csv"
    gt = pd.read_csv(os.path.join(d, gfile), comment="#", header=None, skipinitialspace=True)
    gtn = gt[0].values.astype(np.int64); g = gt.iloc[:, 1:].values.astype(float)
    keep = np.concatenate([[True], np.diff(gtn) > 0]); gtn = gtn[keep]; g = g[keep]
    s.gt_raw_t = (gtn - s.tn[0]) * 1e-9; s.gt_raw_p = g[:, 0:3]; s.gt_raw_q = g[:, [4, 5, 6, 3]]
    s.gt_raw = g
    tt = s.t; gtt = s.gt_raw_t
    idx = np.clip(np.searchsorted(gtt, tt), 1, len(gtt) - 1)
    s.ok = (tt >= gtt[0]) & (tt <= gtt[-1]) & ((gtt[idx] - gtt[idx - 1]) < 0.05)
    s.gt_p = np.stack([np.interp(tt, gtt, g[:, i]) for i in range(3)], 1)
    tc = np.clip(tt, gtt[0], gtt[-1])
    s.gt_q = Slerp(gtt, Rot.from_quat(s.gt_raw_q))(tc).as_quat()
    if g.shape[1] >= 16:
        s.gt_v = np.stack([np.interp(tt, gtt, g[:, 7 + i]) for i in range(3)], 1)
        s.gt_bg = g[:, 10:13]; s.gt_ba = g[:, 13:16]
    else:
        dtg = np.median(np.diff(gtt)); win = int(round(0.25 / dtg)) | 1
        vraw = np.stack([savgol_filter(g[:, i], win, 3, deriv=1, delta=dtg) for i in range(3)], 1)
        s.gt_v = np.stack([np.interp(tt, gtt, vraw[:, i]) for i in range(3)], 1); s.gt_bg = None; s.gt_ba = None
    return s

# ---------------------------------------------------------------- quaternions (xyzw)
def qmul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]; bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)
def qinv(q): return q * np.array([-1.0, -1.0, -1.0, 1.0])
def qrot(q, v):
    u = q[..., :3]; w = q[..., 3:4]; c = np.cross(u, v)
    return v + 2 * w * c + 2 * np.cross(u, c)
def qexp(rv):
    th = np.linalg.norm(rv, axis=-1, keepdims=True); k = np.where(th > 1e-9, np.sin(th / 2) / np.maximum(th, 1e-12), 0.5)
    return np.concatenate([rv * k, np.cos(th / 2)], -1)
def qlog(q):
    q = q * np.where(q[..., 3:4] < 0, -1.0, 1.0); n = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    return q[..., :3] * np.where(n > 1e-9, 2 * np.arctan2(n, q[..., 3:4]) / np.maximum(n, 1e-12), 2.0)
def qcumprod(dq):
    """out[k] = dq[0] * dq[1] * ... * dq[k] (prefix product, log-depth)."""
    out = dq.copy(); s = 1
    while s < len(out):
        out[s:] = qmul(out[:-s], out[s:]); out /= np.linalg.norm(out, axis=1, keepdims=True); s *= 2
    return out
def integrate_gyro(t, w):
    dt = np.diff(t)[:, None]; dq = qexp(0.5 * (w[:-1] + w[1:]) * dt)
    return qcumprod(np.vstack([[0, 0, 0, 1.0], dq]))
def shortest_arc(u, v):
    """Quaternion rotating unit vectors u onto unit vectors v along the shortest arc."""
    c = np.cross(u, v); d = np.sum(u * v, -1, keepdims=True)
    q = np.concatenate([c, 1.0 + d], -1); return q / np.linalg.norm(q, axis=-1, keepdims=True)
def ang_deg(q1, q2):
    d = np.abs(np.sum(q1 * q2, -1)).clip(0, 1); return np.degrees(2 * np.arccos(d))
def tilt_deg(q_est, q_gt):
    """Angle between the estimated and true gravity directions expressed in the sensor frame (yaw-free)."""
    z = np.array([0, 0, 1.0]); a = qrot(qinv(q_est), z); b = qrot(qinv(q_gt), z)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, -1), -1, 1)))

def lowpass(x, fs, fc, order=2):
    sos = butter(order, fc / (fs / 2), output="sos"); return sosfiltfilt(sos, x, axis=0)

def rest_mask(t, w, a, win=1.0, gyro_std=0.01, acc_std=0.08):
    """Samples inside windows where both gyro and accelerometer are quiet (thresholds in rad/s and m/s^2)."""
    fs = 1.0 / np.median(np.diff(t)); n = max(3, int(win * fs)); m = np.zeros(len(t), bool)
    for i in range(0, len(t) - n + 1, n // 2):
        if w[i:i + n].std(0).max() < gyro_std and a[i:i + n].std(0).max() < acc_std: m[i:i + n] = True
    return m

def attitude(t, w, a, tau=3.0, iters=2, bg0=None, ridge=1e-4, return_all=False):
    """Offline VQF-style attitude: gyro strapdown, then zero-phase low-pass of specific force in the
    gyro-integrated frame gives the gravity direction (bounded velocity => mean linear acceleration -> 0).
    Constant gyro bias is refined from the drift of that gravity direction: d(tilt error)/dt = -(R b)_xy.
    Heading is gyro-only (unobservable from IMU); the initial heading is arbitrary."""
    fs = 1.0 / np.median(np.diff(t)); bg = np.zeros(3) if bg0 is None else np.array(bg0, float).copy(); ez = np.array([0, 0, 1.0])
    for it in range(iters + 1):
        qg = integrate_gyro(t, w - bg)
        aI = qrot(qg, a); gI = lowpass(aI, fs, 1.0 / (2 * np.pi * tau)); gI /= np.linalg.norm(gI, axis=1, keepdims=True)
        qc = shortest_arc(gI, np.broadcast_to(ez, gI.shape)); q = qmul(qc, qg); q /= np.linalg.norm(q, axis=1, keepdims=True)
        if it == iters: break
        step = max(1, int(fs * 1.0)); wc = qlog(qmul(qc[step:], qinv(qc[:-step]))) / (t[step:] - t[:-step])[:, None]
        Rm = Rot.from_quat(q[:-step]).as_matrix()
        A = Rm[::10, :2, :].reshape(-1, 3); y = -wc[::10, :2].reshape(-1)
        db = np.linalg.solve(A.T @ A + ridge * len(y) * np.eye(3), A.T @ y); bg = bg + db
    return (q, bg, qg, qc) if return_all else (q, bg)

# ---------------------------------------------------------------- metrics (ground truth used here only)
def umeyama(est, gt):
    """Rigid SE(3) alignment (no scale) est -> gt. A rank-deficient estimate (e.g. constant) gets R = I,
    which is what makes the motionless reference score the RMS radius about the GT centroid."""
    me, mg = est.mean(0), gt.mean(0); H = (est - me).T @ (gt - mg)
    if np.linalg.matrix_rank(H, tol=1e-9) < 2: return np.eye(3), mg - me
    U, S, Vt = np.linalg.svd(H); D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]); Rm = Vt.T @ D @ U.T
    return Rm, mg - Rm @ me
def yaw_align(est, gt):
    """4-DoF alignment (yaw + translation) for gravity-aligned estimates."""
    me, mg = est.mean(0), gt.mean(0); e, g = est - me, gt - mg
    th = np.arctan2(np.sum(e[:, 0] * g[:, 1] - e[:, 1] * g[:, 0]), np.sum(e[:, 0] * g[:, 0] + e[:, 1] * g[:, 1]))
    c, s_ = np.cos(th), np.sin(th); Rm = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]]); return Rm, mg - Rm @ me
def rmse(e): return float(np.sqrt(np.mean(np.sum(e ** 2, -1))))
def ate(est, gt, mode="se3"):
    if mode == "se3": Rm, tt = umeyama(est, gt)
    elif mode == "yaw": Rm, tt = yaw_align(est, gt)
    elif mode == "origin": Rm, tt = np.eye(3), gt[0] - est[0]
    else: Rm, tt = np.eye(3), np.zeros(3)
    return rmse(est @ Rm.T + tt - gt)
def rpe_trans(t, p, q, gp, gq, delta=1.0):
    """evo-style translational RPE: relative motion over delta expressed in each trajectory's own body frame."""
    fs = 1.0 / np.median(np.diff(t)); n = int(round(delta * fs))
    de = qrot(qinv(q[:-n]), p[n:] - p[:-n]); dg = qrot(qinv(gq[:-n]), gp[n:] - gp[:-n])
    return rmse(de - dg)
def rpe_disp(t, p, gp, Rm, delta=1.0):
    """Displacement RPE after rotating the estimate with a global alignment rotation (orientation-free)."""
    fs = 1.0 / np.median(np.diff(t)); n = int(round(delta * fs))
    return rmse((p[n:] - p[:-n]) @ Rm.T - (gp[n:] - gp[:-n]))

# ---------------------------------------------------------------- linear-Gaussian models, Kalman filter and RTS smoother
from scipy.linalg import expm, solve_continuous_lyapunov
from numba import njit

def van_loan(F, Qc, dt):
    n = F.shape[0]; M = np.zeros((2 * n, 2 * n)); M[:n, :n] = -F; M[:n, n:] = Qc; M[n:, n:] = F.T
    E = expm(M * dt); A = E[n:, n:].T; return A, A @ E[:n, n:]
def discretise(F, Qc, t, round_s=1e-5):
    dts = np.round(np.diff(t) / round_s).astype(np.int64); u, inv = np.unique(dts, return_inverse=True)
    A = np.zeros((len(u),) + F.shape); Q = np.zeros_like(A)
    for i, d in enumerate(u):
        A[i], Q[i] = van_loan(F, Qc, max(d, 0) * round_s); Q[i] = 0.5 * (Q[i] + Q[i].T)
    return A, Q, np.concatenate([[0], inv]).astype(np.int64)

def model_matern52(ell, sp, tau_b, sb):
    """Stationary position prior (Matern nu=5/2, length scale ell [s], std sp [m]) + OU bias (tau_b [s], std sb [m/s^2]).
    State [p, v, a, b]; the measured world-frame acceleration is a + b + noise."""
    lam = np.sqrt(5.0) / ell; F = np.zeros((4, 4)); F[0, 1] = 1; F[1, 2] = 1; F[2, :3] = [-lam ** 3, -3 * lam ** 2, -3 * lam]; F[3, 3] = -1.0 / tau_b
    Qc = np.zeros((4, 4)); Qc[2, 2] = 16.0 / 3.0 * sp ** 2 * lam ** 5; Qc[3, 3] = 2 * sb ** 2 / tau_b
    P0 = solve_continuous_lyapunov(F, -Qc); return F, Qc, 0.5 * (P0 + P0.T), np.array([0, 0, 1.0, 1.0])
def model_singer(tau_a, sa, tau_b, sb, p0=1e-6, v0=1.0):
    """Non-stationary reference: Singer acceleration (OU, tau_a, std sa), free velocity and position."""
    F = np.zeros((4, 4)); F[0, 1] = 1; F[1, 2] = 1; F[2, 2] = -1.0 / tau_a; F[3, 3] = -1.0 / tau_b
    Qc = np.zeros((4, 4)); Qc[2, 2] = 2 * sa ** 2 / tau_a; Qc[3, 3] = 2 * sb ** 2 / tau_b
    return F, Qc, np.diag([p0, v0 ** 2, sa ** 2, sb ** 2]), np.array([0, 0, 1.0, 1.0])
def model_ou_vel(tau_v, sv, tau_a, sa, tau_b, sb, p0=1e-6):
    """Velocity-stationary reference: Singer acceleration with velocity damping 1/tau_v; position is free (random-walk-like)."""
    F = np.zeros((4, 4)); F[0, 1] = 1; F[1, 1] = -1.0 / tau_v; F[1, 2] = 1; F[2, 2] = -1.0 / tau_a; F[3, 3] = -1.0 / tau_b
    Qc = np.zeros((4, 4)); Qc[2, 2] = 2 * sa ** 2 / tau_a; Qc[3, 3] = 2 * sb ** 2 / tau_b
    return F, Qc, np.diag([p0, sv ** 2, sa ** 2, sb ** 2]), np.array([0, 0, 1.0, 1.0])

@njit(cache=True)
def kf_rts(A, Q, idx, H, y, Rk, mask, P0, smooth):
    """Per-axis independent scalar-measurement Kalman filter + RTS smoother sharing one covariance recursion.
    y: (N, 3) measurements, Rk: (N,) measurement variance, mask: (N,) valid flag. Returns smoothed and filtered
    means (N, 3, n), covariances (N, n, n) and the log-likelihood."""
    N = y.shape[0]; n = A.shape[1]
    xf = np.zeros((N, 3, n)); Pf = np.zeros((N, n, n)); xp = np.zeros((N, 3, n)); Pp = np.zeros((N, n, n))
    x = np.zeros((3, n)); P = P0.copy(); ll = 0.0
    for k in range(N):
        if k > 0:
            Ak = A[idx[k]]; x = x @ Ak.T; P = Ak @ P @ Ak.T + Q[idx[k]]
        xp[k] = x; Pp[k] = P
        if mask[k]:
            PH = P @ H; S = H @ PH + Rk[k]; K = PH / S
            for j in range(3):
                r = y[k, j] - H @ x[j]; x[j] = x[j] + K * r; ll += -0.5 * (np.log(2 * np.pi * S) + r * r / S)
            P = P - np.outer(K, PH); P = 0.5 * (P + P.T)
        xf[k] = x; Pf[k] = P
    xs = xf.copy(); Ps = Pf.copy()
    if smooth:
        for k in range(N - 2, -1, -1):
            Ak = A[idx[k + 1]]; C = Pf[k] @ Ak.T @ np.linalg.inv(Pp[k + 1])
            for j in range(3): xs[k, j] = xf[k, j] + C @ (xs[k + 1, j] - xp[k + 1, j])
            Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return xs, Ps, xf, Pf, ll

def world_acc(q, a, g=G):
    aw = qrot(q, a); aw[:, 2] -= g; return aw
def run_linear(t, aw, model, sn, mask=None, Rk=None, smooth=True):
    F, Qc, P0, H = model; A, Q, idx = discretise(F, Qc, t)
    mask = np.ones(len(t), np.bool_) if mask is None else mask.astype(np.bool_)
    Rk = np.full(len(t), sn ** 2) if Rk is None else Rk
    return kf_rts(A, Q, idx, H, np.ascontiguousarray(np.nan_to_num(aw)), Rk, mask, P0, smooth)
def strapdown_pos(t, aw):
    dt = np.diff(t, prepend=t[0])[:, None]; v = np.cumsum(aw * dt, 0); return np.cumsum(v * dt, 0)
def highpass_pos(t, aw, fc):
    """Classic 'heave filter' reference: zero-phase high-pass at fc after each of the two integrations."""
    fs = 1.0 / np.median(np.diff(t)); sos = butter(2, fc / (fs / 2), btype="high", output="sos"); dt = 1.0 / fs
    v = sosfiltfilt(sos, np.cumsum(sosfiltfilt(sos, aw, axis=0), 0) * dt, axis=0); return sosfiltfilt(sos, np.cumsum(v, 0) * dt, axis=0)

# ---------------------------------------------------------------- bias + attitude front end (no ground truth)
def gyro_bias(t, w, a, win=0.5, frac=0.05, gate=0.035):
    """Constant gyro bias from the stream alone. If the quietest `frac` of windows are genuinely quiet (max-axis gyro
    std < gate rad/s; true for the EuRoC MAV before take-off / after landing, false for hand-held TUM-VI), their mean
    gyro reading is the bias on all three axes. Otherwise returns zeros and quiet=False (see estimate_attitude)."""
    fs = 1.0 / np.median(np.diff(t)); n = max(8, int(win * fs)); m = len(t) // n
    W = w[:m * n].reshape(m, n, 3); A = a[:m * n].reshape(m, n, 3); gs = W.std(1).max(1); as_ = A.std(1).max(1)
    sc = gs / np.median(gs) + as_ / np.median(as_); k = max(3, int(frac * m)); o = np.argsort(sc)[:k]; o = o[gs[o] < gate]
    if len(o) >= 3: return W[o].mean((0, 1)), True
    return np.zeros(3), False
def estimate_attitude(t, w, a, tau=3.0):
    bg, quiet = gyro_bias(t, w, a)
    if quiet: q, _ = attitude(t, w, a, tau=tau, iters=0, bg0=bg)
    else: q, bg = attitude(t, w, a, tau=tau, iters=2, ridge=1e-1)
    return q, bg, quiet
def world_acc_centered(q, a):
    """World-frame specific force with its sequence mean removed (bounded velocity => mean acceleration ~ 0; this also
    removes the unknown local gravity magnitude and the mean accelerometer bias)."""
    aw = qrot(q, a); return aw - aw.mean(0)
def hf_std(x):
    return float(np.diff(x, axis=0).std() / np.sqrt(2))

# ---------------------------------------------------------------- joint smoother with optional rotor-drag velocity information
@njit(cache=True)
def kf_rts_seq(A, Q, idx, H, y, R, mask, P0, smooth):
    """General Kalman filter + RTS smoother with sequential scalar updates. H: (N, m, n), y/R/mask: (N, m)."""
    N = H.shape[0]; m = H.shape[1]; n = H.shape[2]
    xf = np.zeros((N, n)); Pf = np.zeros((N, n, n)); xp = np.zeros((N, n)); Pp = np.zeros((N, n, n))
    x = np.zeros(n); P = P0.copy(); ll = 0.0
    for k in range(N):
        if k > 0:
            Ak = A[idx[k]]; x = Ak @ x; P = Ak @ P @ Ak.T + Q[idx[k]]
        xp[k] = x; Pp[k] = P
        for i in range(m):
            if mask[k, i]:
                h = H[k, i]; PH = P @ h; S = h @ PH + R[k, i]; r = y[k, i] - h @ x
                K = PH / S; x = x + K * r; P = P - np.outer(K, PH); ll += -0.5 * (np.log(2 * np.pi * S) + r * r / S)
        P = 0.5 * (P + P.T); xf[k] = x; Pf[k] = P
    xs = xf.copy(); Ps = Pf.copy()
    if smooth:
        for k in range(N - 2, -1, -1):
            Ak = A[idx[k + 1]]; C = Pf[k] @ Ak.T @ np.linalg.inv(Pp[k + 1])
            xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1]); Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return xs, Ps, xf, Pf, ll

def flight_mask(t, a, frac=0.5):
    """Rotor-vibration detector (no ground truth): high-frequency accelerometer energy above `frac` of its median."""
    fs = 1.0 / np.median(np.diff(t)); e = np.sum(np.diff(a, axis=0, prepend=a[:1]) ** 2, 1)
    vib = np.sqrt(lowpass(e, fs, 1.0).clip(0)); return vib > frac * np.median(vib)
def thrust_frame(a):
    """Thrust axis = mean specific-force direction in the sensor frame, plus two in-plane unit vectors."""
    zb = a.mean(0); zb = zb / np.linalg.norm(zb); h = np.array([0, 1.0, 0]) if abs(zb[1]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(zb, h); e1 /= np.linalg.norm(e1); return zb, e1, np.cross(zb, e1)
def run_joint(t, q, a, gp, sn, k_drag=0.0, sd=1.0, tau_c=100.0, sc=0.15, valid=None, flying=None, smooth=True, pos_meas=None, vel_meas=None):
    """Joint 14-state smoother: per-axis [p, v, a, b] (stationary Matern-5/2 position prior + OU bias) and two in-plane
    drag-bias states. Measurements: world-frame acceleration (3) and, if k_drag > 0 and flying, the rotor-drag relation
    e_i . f = -k * (R e_i) . v + c_i (2). pos_meas = (index array, positions (M,3), sigma) adds position fixes (waypoints)."""
    ell, sp, tb, sb = gp; F4, Qc4, P04, _ = model_matern52(ell, sp, tb, sb); n = 14; N = len(t)
    F = np.zeros((n, n)); Qc = np.zeros((n, n)); P0 = np.zeros((n, n))
    for j in range(3):
        F[4 * j:4 * j + 4, 4 * j:4 * j + 4] = F4; Qc[4 * j:4 * j + 4, 4 * j:4 * j + 4] = Qc4; P0[4 * j:4 * j + 4, 4 * j:4 * j + 4] = P04
    for j in (12, 13): F[j, j] = -1.0 / tau_c; Qc[j, j] = 2 * sc ** 2 / tau_c; P0[j, j] = sc ** 2
    A, Q, idx = discretise(F, Qc, t); m = 5 + (3 if pos_meas is not None else 0) + (3 if vel_meas is not None else 0)
    H = np.zeros((N, m, n)); y = np.zeros((N, m)); R = np.ones((N, m)); mask = np.zeros((N, m), np.bool_)
    aw = world_acc_centered(q, a) if valid is None else qrot(q, a) - qrot(q[valid], a[valid]).mean(0)
    v_ok = np.ones(N, np.bool_) if valid is None else valid.astype(np.bool_)
    snk = np.full(N, sn ** 2) if np.isscalar(sn) else np.asarray(sn) ** 2
    for j in range(3):
        H[:, j, 4 * j + 2] = 1; H[:, j, 4 * j + 3] = 1; y[:, j] = np.nan_to_num(aw[:, j]); R[:, j] = snk; mask[:, j] = v_ok
    if k_drag > 0:
        zb, e1, e2 = thrust_frame(a[v_ok]); fl = flight_mask(t, np.nan_to_num(a)) if flying is None else flying
        for i, e in enumerate((e1, e2)):
            ew = qrot(q, np.broadcast_to(e, (N, 3)))
            for j in range(3): H[:, 3 + i, 4 * j + 1] = -k_drag * ew[:, j]
            H[:, 3 + i, 12 + i] = 1; y[:, 3 + i] = np.nan_to_num(a @ e); R[:, 3 + i] = sd ** 2 if np.isscalar(sd) else np.asarray(sd) ** 2; mask[:, 3 + i] = v_ok & fl
    if pos_meas is not None:
        ii, pp, sg = pos_meas
        for j in range(3): H[ii, 5 + j, 4 * j] = 1; y[ii, 5 + j] = pp[:, j]; R[ii, 5 + j] = sg ** 2; mask[ii, 5 + j] = True
    if vel_meas is not None:
        o = 8 if pos_meas is not None else 5; ii, vv, sg = vel_meas
        for j in range(3): H[ii, o + j, 4 * j + 1] = 1; y[ii, o + j] = vv[:, j]; R[ii, o + j] = sg ** 2; mask[ii, o + j] = True
    xs, Ps, xf, Pf, ll = kf_rts_seq(A, Q, idx, H, y, R, mask, P0, smooth)
    return {"p": xs[:, [0, 4, 8]], "v": xs[:, [1, 5, 9]], "pf": xf[:, [0, 4, 8]], "vf": xf[:, [1, 5, 9]], "xs": xs, "xf": xf, "Ps": Ps, "Pf": Pf, "ll": float(ll), "A": A, "idx": idx}

# ---------------------------------------------------------------- degradation simulator (OUR assumptions, not the organiser's script)
def degrade(t, w, a, cfg, seed=0):
    """Synthetic corruption of a real stream. cfg keys (all optional):
    gn, an   : final std of added white noise on gyro [rad/s] / accel [m/s^2]; shape: 'linear' | 'exp' | 'step'
    gb, ab   : final magnitude of added bias on gyro / accel; bias_shape: 'ramp' | 'rw'
    gaps     : list of dropout durations [s] (samples become NaN and valid=False); rot: True applies a random
               constant rotation of the sensor frame (unknown gravity alignment).
    Returns w2, a2, valid, meta. The organiser's degradation script was not located; this is NOT a replica of it."""
    rng = np.random.default_rng(seed); N = len(t); u = (t - t[0]) / (t[-1] - t[0]); shape = cfg.get("shape", "linear")
    prof = u if shape == "linear" else (np.exp(4 * u) - 1) / (np.exp(4) - 1) if shape == "exp" else np.minimum(np.floor(u * 4) / 3, 1.0)
    w2 = w + rng.normal(size=(N, 3)) * (cfg.get("gn", 0.0) * prof)[:, None]; a2 = a + rng.normal(size=(N, 3)) * (cfg.get("an", 0.0) * prof)[:, None]
    for arr, key in ((w2, "gb"), (a2, "ab")):
        mag = cfg.get(key, 0.0)
        if mag > 0:
            if cfg.get("bias_shape", "ramp") == "ramp":
                d = rng.normal(size=3); arr += (mag * u)[:, None] * d / np.linalg.norm(d)
            else:
                rw = np.cumsum(rng.normal(size=(N, 3)), 0); arr += rw * mag / np.sqrt(N)
    valid = np.ones(N, bool); gaps = []
    for dur in cfg.get("gaps", []):
        for _ in range(50):
            t0 = rng.uniform(t[0] + 0.2 * (t[-1] - t[0]), t[0] + 0.9 * (t[-1] - t[0]) - dur)
            if all(abs(t0 - g0) > dur + 5.0 for g0, _ in gaps): break
        gaps.append((t0, dur)); m = (t >= t0) & (t < t0 + dur); valid[m] = False
    Rr = np.eye(3)
    if cfg.get("rot", False):
        Rr = Rot.random(random_state=int(rng.integers(1 << 31))).as_matrix(); w2 = w2 @ Rr.T; a2 = a2 @ Rr.T
    w2[~valid] = np.nan; a2[~valid] = np.nan
    return w2, a2, valid, {"gaps": gaps, "Rr": Rr}
def fill_gaps(t, x, valid):
    if valid.all(): return x
    return np.stack([np.interp(t, t[valid], x[valid, i]) for i in range(x.shape[1])], 1)
def rolling_hf_std(t, x, valid, win=1.0):
    """Time-varying white-noise level from the local variance of first differences (cheap adaptive-R estimate)."""
    fs = 1.0 / np.median(np.diff(t)); n = max(8, int(win * fs)); d2 = np.sum(np.diff(x, axis=0, prepend=x[:1]) ** 2, 1) / (2 * x.shape[1])
    d2[~valid] = np.nan; d2 = pd.Series(d2).rolling(n, center=True, min_periods=n // 4).median().interpolate(limit_direction="both").values / 0.789
    return np.sqrt(np.maximum(d2, 1e-8))
def front_end(t, w, a, valid=None, tau=3.0, adaptive_tau=False):
    """Attitude + bias front end that tolerates NaN gaps: gyro/accel are linearly bridged for the attitude filter only."""
    valid = np.ones(len(t), bool) if valid is None else valid; wf = fill_gaps(t, w, valid); af = fill_gaps(t, a, valid)
    if adaptive_tau:
        gs = np.median(rolling_hf_std(t, wf, valid)); tau = float(np.clip(0.02 / max(gs, 1e-4), 0.5, 3.0)) if gs > 0.0067 else tau
    q, bg, quiet = estimate_attitude(t, wf, af, tau=tau); return q, bg, quiet, wf, af, tau

# ---------------------------------------------------------------- gap-aware front end (v2, after E03 showed v1 fails across dropouts)
def segments(valid, min_len=1):
    d = np.diff(np.concatenate([[0], valid.astype(np.int8), [0]])); st = np.where(d == 1)[0]; en = np.where(d == -1)[0]
    return [(i, j) for i, j in zip(st, en) if j - i >= min_len]
def lowpass_seg(x, fs, fc, valid):
    """Zero-phase low-pass applied independently inside each contiguous valid segment (nothing is smeared across a gap);
    gap samples are filled afterwards by linear interpolation between the neighbouring segment ends."""
    y = np.full_like(x, np.nan); sos = butter(2, fc / (fs / 2), output="sos")
    for i, j in segments(valid):
        n = j - i
        y[i:j] = sosfiltfilt(sos, x[i:j], axis=0) if n > 30 else x[i:j].mean(0)
    ok = ~np.isnan(y[:, 0]); tt = np.arange(len(x))
    return np.stack([np.interp(tt, tt[ok], y[ok, k]) for k in range(x.shape[1])], 1)
def attitude_seg(t, w, a, valid, tau=3.0, iters=2, bg0=None, ridge=1e-4, bridge="lin"):
    """Gap-aware version of attitude(): gyro is bridged across gaps ('lin' = linear interpolation of the rate, 'zero' = no
    rotation), the gravity direction is re-estimated inside every segment, and the bias fit never spans a gap."""
    fs = 1.0 / np.median(np.diff(t)); bg = np.zeros(3) if bg0 is None else np.array(bg0, float).copy(); ez = np.array([0, 0, 1.0])
    af = fill_gaps(t, a, valid); seg = np.cumsum(np.concatenate([[0], np.diff(valid.astype(np.int8)) != 0]))
    for it in range(iters + 1):
        wc = fill_gaps(t, w, valid) - bg
        if bridge == "zero": wc = np.where(valid[:, None], wc, 0.0)
        qg = integrate_gyro(t, wc); gI = lowpass_seg(qrot(qg, af), fs, 1.0 / (2 * np.pi * tau), valid); gI /= np.linalg.norm(gI, axis=1, keepdims=True)
        qc = shortest_arc(gI, np.broadcast_to(ez, gI.shape)); q = qmul(qc, qg); q /= np.linalg.norm(q, axis=1, keepdims=True)
        if it == iters: break
        step = max(1, int(fs * 1.0)); wcr = qlog(qmul(qc[step:], qinv(qc[:-step]))) / (t[step:] - t[:-step])[:, None]
        good = valid[:-step] & valid[step:] & (seg[:-step] == seg[step:]); Rm = Rot.from_quat(q[:-step]).as_matrix()
        A = Rm[good][::10, :2, :].reshape(-1, 3); y = -wcr[good][::10, :2].reshape(-1)
        bg = bg + np.linalg.solve(A.T @ A + ridge * len(y) * np.eye(3), A.T @ y)
    return q, bg
def front_end2(t, w, a, valid=None, tau=3.0, bridge="lin"):
    valid = np.ones(len(t), bool) if valid is None else valid
    wf = fill_gaps(t, w, valid); af = fill_gaps(t, a, valid); b0, quiet = gyro_bias(t[valid], w[valid], a[valid])
    if quiet: q, bg = attitude_seg(t, w, a, valid, tau=tau, iters=0, bg0=b0, bridge=bridge)
    else: q, bg = attitude_seg(t, w, a, valid, tau=tau, iters=2, ridge=1e-1, bridge=bridge)
    return q, bg, quiet, wf, af

@njit(cache=True)
def kf_rts2(A, Q, idx, H, y, Rk, mask, P0, binf):
    """kf_rts with an extra per-step variance injection on the bias state (index n-1), used to let the bias re-converge
    after a dropout, where the tilt error (and hence the apparent world-frame bias) can jump."""
    N = y.shape[0]; n = A.shape[1]
    xf = np.zeros((N, 3, n)); Pf = np.zeros((N, n, n)); xp = np.zeros((N, 3, n)); Pp = np.zeros((N, n, n))
    x = np.zeros((3, n)); P = P0.copy(); ll = 0.0
    for k in range(N):
        if k > 0:
            Ak = A[idx[k]]; x = x @ Ak.T; P = Ak @ P @ Ak.T + Q[idx[k]]; P[n - 1, n - 1] += binf[k]
        xp[k] = x; Pp[k] = P
        if mask[k]:
            PH = P @ H; S = H @ PH + Rk[k]; K = PH / S
            for j in range(3):
                r = y[k, j] - H @ x[j]; x[j] = x[j] + K * r; ll += -0.5 * (np.log(2 * np.pi * S) + r * r / S)
            P = P - np.outer(K, PH); P = 0.5 * (P + P.T)
        xf[k] = x; Pf[k] = P
    xs = xf.copy(); Ps = Pf.copy()
    for k in range(N - 2, -1, -1):
        Ak = A[idx[k + 1]]; C = Pf[k] @ Ak.T @ np.linalg.inv(Pp[k + 1])
        for j in range(3): xs[k, j] = xf[k, j] + C @ (xs[k + 1, j] - xp[k + 1, j])
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return xs, Ps, xf, Pf, ll
def gp_smooth(t, q, af, valid, gp=(1, 2, 20, 0.1), bias_jump=0.0, adaptive_R=False):
    """Stationary-GP position smoother on a (possibly gappy) stream. bias_jump [m/s^2]: std injected into the bias state
    at the first valid sample after each gap."""
    aw = qrot(q, af); aw = aw - aw[valid].mean(0); F, Qc, P0, H = model_matern52(*gp); A, Q, idx = discretise(F, Qc, t)
    Rk = rolling_hf_std(t, aw, valid) ** 2 if adaptive_R else np.full(len(t), hf_std(aw[valid]) ** 2)
    binf = np.zeros(len(t)); binf[1:][valid[1:] & ~valid[:-1]] = bias_jump ** 2
    xs, Ps, xf, Pf, ll = kf_rts2(A, Q, idx, H, np.ascontiguousarray(aw), Rk, valid.astype(np.bool_), P0, binf)
    return xs[:, :, 0], xs, Ps, xf, Pf, ll

# ---------------------------------------------------------------- spring-damper prior (v3, after E03b: Matern reverts to the mean inside gaps)
def model_spring(T_p, tau_v, tau_a, sa, tau_b, sb):
    """Position = weakly restored, velocity-damped particle driven by an OU acceleration:
       p' = v;  v' = -w0^2 p - v/tau_v + d;  d' = -d/tau_a + noise (std sa);  bias OU (tau_b, sb).
    w0 = 2*pi/T_p sets a slow return to the centre (keeps position stationary), tau_v is the velocity memory that also
    governs coasting through gaps and the blind window, tau_a/sa give the high-frequency motion power.
    Measured world acceleration = p'' + b + n = -w0^2 p - v/tau_v + d + b + n."""
    w0 = 2 * np.pi / T_p; c = 1.0 / tau_v; F = np.zeros((4, 4)); F[0, 1] = 1; F[1, 0] = -w0 ** 2; F[1, 1] = -c; F[1, 2] = 1; F[2, 2] = -1.0 / tau_a; F[3, 3] = -1.0 / tau_b
    Qc = np.zeros((4, 4)); Qc[2, 2] = 2 * sa ** 2 / tau_a; Qc[3, 3] = 2 * sb ** 2 / tau_b
    P0 = solve_continuous_lyapunov(F, -Qc); return F, Qc, 0.5 * (P0 + P0.T), np.array([-w0 ** 2, -c, 1.0, 1.0])
def lin_smooth(t, q, af, valid, model, bias_jump=0.0):
    """Per-axis linear-Gaussian smoother for any (F, Qc, P0, H) model on a possibly gappy stream."""
    aw = qrot(q, af); aw = aw - aw[valid].mean(0); F, Qc, P0, H = model; A, Q, idx = discretise(F, Qc, t)
    Rk = np.full(len(t), hf_std(aw[valid]) ** 2); binf = np.zeros(len(t)); binf[1:][valid[1:] & ~valid[:-1]] = bias_jump ** 2
    xs, Ps, xf, Pf, ll = kf_rts2(A, Q, idx, H, np.ascontiguousarray(aw), Rk, valid.astype(np.bool_), P0, binf)
    return xs[:, :, 0], xs, Ps, xf, Pf, ll
def model_predict(F, x, taus):
    """Model-consistent blind prediction of position displacement from a per-axis state x (3, n) over horizons taus."""
    out = np.zeros((len(taus), 3))
    for i, tu in enumerate(taus): out[i] = x @ expm(F * tu)[0] - x[:, 0]
    return out
