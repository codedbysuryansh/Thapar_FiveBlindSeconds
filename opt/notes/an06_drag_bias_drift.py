"""an06: how much does the effective drag-row offset wander within a flight? Clean EuRoC + ground truth (analysis only).
Flight mask = rotor vibration on (as main.py) and ground-truth height above the 5 % quantile + 0.3 m. Residual of the linear
drag model with the flight's own k, low-passed at 0.05 / 0.02 Hz; its spread within the flight (velocity-equivalent = / k)."""
import os, sys, numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, "~/SamsungPRISM_FiveBlindSeconds/opt/harness")
import fbsopt as H
def lp(x, fc, fs=200.0):
    b, a = butter(2, fc / (fs / 2)); return filtfilt(b, a, x, axis=0)
for n in H.EUROC:
    imu = np.loadtxt(os.path.join(H.mav0(n), "imu0", "data.csv"), delimiter=",", comments="#"); gt = np.loadtxt(os.path.join(H.mav0(n), "state_groundtruth_estimate0", "data.csv"), delimiter=",", comments="#")
    t = imu[:, 0]; tg = gt[:, 0]; sel = (t >= tg[0]) & (t <= tg[-1]); t = t[sel] * 1e-9; a = imu[sel, 4:7]
    idx = np.clip(np.searchsorted(tg * 1e-9, t), 0, len(tg) - 1); R = Rot.from_quat(gt[idx][:, [5, 6, 7, 4]]); v = gt[idx, 8:11]; ba = gt[idx, 14:17]; z = gt[idx, 3]
    vib = np.sqrt(np.clip(lp(np.r_[0, np.sum(np.diff(a, axis=0) ** 2, 1)], 1.0), 0, None)); fly = (vib > 0.5 * np.median(vib)) & (z > np.quantile(z, 0.05) + 0.3)
    f = a - ba; vS = R.inv().apply(v); zb = f.mean(0); zb /= np.linalg.norm(zb); h = np.array([0, 1.0, 0]) if abs(zb[1]) < 0.9 else np.array([1.0, 0, 0]); e1 = np.cross(zb, h); e1 /= np.linalg.norm(e1); e2 = np.cross(zb, e1)
    fl = lp(f, 2.0); vl = lp(vS, 2.0); F = np.stack([fl @ e1, fl @ e2], 1); U = np.stack([vl @ e1, vl @ e2], 1)
    Fi, Ui = F[fly], U[fly]; X = np.concatenate([-Ui.reshape(-1, order="F")[:, None], np.kron(np.eye(2), np.ones((len(Ui), 1)))], 1); beta = np.linalg.lstsq(X, Fi.reshape(-1, order="F"), rcond=None)[0]
    r = F + beta[0] * U - beta[1:3]; r[~fly] = 0.0
    out = []
    for fc in (0.05, 0.02):
        rl = lp(r, fc)[fly]; out.append(float(np.sqrt(np.mean(np.sum(rl ** 2, 1)))))
    rfull = r[fly]; seg = len(rfull) // 3; thirds = [rfull[i * seg:(i + 1) * seg].mean(0) for i in range(3)]
    print("%-16s k %.3f  fly %.2f  resid rms %.3f | slow part rms (0.05 Hz) %.3f m/s^2 = %.2f m/s, (0.02 Hz) %.3f = %.2f m/s | offset by third (m/s^2): %s" % (
        n, beta[0], fly.mean(), np.sqrt(np.mean(np.sum(rfull ** 2, 1))), out[0], out[0] / beta[0], out[1], out[1] / beta[0], " ".join("(%+.3f,%+.3f)" % tuple(x) for x in thirds)))
