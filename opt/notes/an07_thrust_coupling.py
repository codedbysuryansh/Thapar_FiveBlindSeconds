"""an07: is part of the drag-row residual a thrust-coupled offset (thrust-axis misalignment: e_i . f picks up delta_i * (fz - mean))?
Clean EuRoC + ground truth (analysis only), flight mask as an06. Per flight: residual of the linear drag model with and without
per-axis thrust-deviation regressors (own fit and leave-one-flight-out shared coefficients), full band and slow band (0.05 Hz)."""
import os, sys, numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, "~/SamsungPRISM_FiveBlindSeconds/opt/harness")
import fbsopt as H
def lp(x, fc, fs=200.0):
    b, a = butter(2, fc / (fs / 2)); return filtfilt(b, a, x, axis=0)
D = {}
for n in H.EUROC:
    imu = np.loadtxt(os.path.join(H.mav0(n), "imu0", "data.csv"), delimiter=",", comments="#"); gt = np.loadtxt(os.path.join(H.mav0(n), "state_groundtruth_estimate0", "data.csv"), delimiter=",", comments="#")
    t = imu[:, 0]; tg = gt[:, 0]; sel = (t >= tg[0]) & (t <= tg[-1]); t = t[sel] * 1e-9; a = imu[sel, 4:7]
    idx = np.clip(np.searchsorted(tg * 1e-9, t), 0, len(tg) - 1); R = Rot.from_quat(gt[idx][:, [5, 6, 7, 4]]); v = gt[idx, 8:11]; ba = gt[idx, 14:17]; z = gt[idx, 3]
    vib = np.sqrt(np.clip(lp(np.r_[0, np.sum(np.diff(a, axis=0) ** 2, 1)], 1.0), 0, None)); fly = (vib > 0.5 * np.median(vib)) & (z > np.quantile(z, 0.05) + 0.3)
    f = a - ba; vS = R.inv().apply(v); zb = f.mean(0); zb /= np.linalg.norm(zb); h = np.array([0, 1.0, 0]) if abs(zb[1]) < 0.9 else np.array([1.0, 0, 0]); e1 = np.cross(zb, h); e1 /= np.linalg.norm(e1); e2 = np.cross(zb, e1)
    fl = lp(f, 2.0); vl = lp(vS, 2.0); F = np.stack([fl @ e1, fl @ e2], 1)[fly]; U = np.stack([vl @ e1, vl @ e2], 1)[fly]; T = (fl @ zb)[fly]; T = T - T.mean()
    D[n] = (F, U, T)
def fit(F, U, T, thrust, k=None, d=None):
    """Per-flight offsets always fitted; k and thrust coefficients fitted unless given."""
    n = len(U); cols = []; y = F.reshape(-1, order="F")
    base = np.kron(np.eye(2), np.ones((n, 1)))
    if k is None: cols.append(-U.reshape(-1, order="F"))
    else: y = y + k * U.reshape(-1, order="F")
    if thrust:
        if d is None: cols += [np.r_[T, np.zeros(n)], np.r_[np.zeros(n), T]]
        else: y = y - np.r_[d[0] * T, d[1] * T]
    X = np.column_stack(cols + [base[:, 0], base[:, 1]]); b = np.linalg.lstsq(X, y, rcond=None)[0]; r = (y - X @ b).reshape(2, -1).T; return b, r
for n in H.EUROC:
    F, U, T = D[n]
    b0, r0 = fit(F, U, T, False); b1, r1 = fit(F, U, T, True)
    # leave-one-flight-out: k and thrust coefficients from the other flights (pooled, offsets per flight)
    ks, ds = [], []
    for m in H.EUROC:
        if m == n: continue
        bb, _ = fit(*D[m], True); ks.append(bb[0]); ds.append(bb[1:3])
    kL, dL = np.mean(ks), np.mean(ds, 0); _bb, rL0 = fit(F, U, T, False, k=kL); _bb, rL1 = fit(F, U, T, True, k=kL, d=dL)
    slow = lambda r: np.sqrt(np.mean(np.sum(lp(r, 0.05, 20.0) ** 2, 1))) if len(r) > 50 else np.nan
    rms = lambda r: np.sqrt(np.mean(np.sum(r ** 2, 1)))
    print("%-16s own: k %.3f d (%+.3f,%+.3f) | resid %.4f -> %.4f, slow %.4f -> %.4f | LOFO: resid %.4f -> %.4f, slow %.4f -> %.4f" % (n, b1[0], b1[1], b1[2], rms(r0), rms(r1), slow(r0[::10]), slow(r1[::10]), rms(rL0), rms(rL1), slow(rL0[::10]), slow(rL1[::10])))
