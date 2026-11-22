"""an05: what makes the rotor-drag rows misfit? Clean EuRoC IMU + ground truth (analysis only). Airborne samples (ground-truth
height > first-second height + 0.3 m). Residual of the linear drag model (per-flight k and offsets, 2 Hz low-pass), binned by
|angular rate|, |thrust - mean|, speed and |jerk| proxies; per flight the residual RMS in the bins."""
import os, sys, json, numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, "~/SamsungPRISM_FiveBlindSeconds/opt/harness")
import fbsopt as H
def lp(x, fc, fs=200.0):
    b, a = butter(2, fc / (fs / 2)); return filtfilt(b, a, x, axis=0)
out = {}
for n in H.EUROC:
    imu = np.loadtxt(os.path.join(H.mav0(n), "imu0", "data.csv"), delimiter=",", comments="#"); gt = np.loadtxt(os.path.join(H.mav0(n), "state_groundtruth_estimate0", "data.csv"), delimiter=",", comments="#")
    t = imu[:, 0]; tg = gt[:, 0]; sel = (t >= tg[0]) & (t <= tg[-1]); t = t[sel]; w = imu[sel, 1:4]; a = imu[sel, 4:7]
    idx = np.clip(np.searchsorted(tg, t), 0, len(tg) - 1); R = Rot.from_quat(gt[idx][:, [5, 6, 7, 4]]); v = gt[idx, 8:11]; ba = gt[idx, 14:17]; bw = gt[idx, 11:14]; z = gt[idx, 3]
    f = a - ba; vS = R.inv().apply(v); fl = lp(f, 2.0); vl = lp(vS, 2.0); wl = lp(w - bw, 2.0)
    zb = f.mean(0); zb /= np.linalg.norm(zb); h = np.array([0, 1.0, 0]) if abs(zb[1]) < 0.9 else np.array([1.0, 0, 0]); e1 = np.cross(zb, h); e1 /= np.linalg.norm(e1); e2 = np.cross(zb, e1)
    air = z > np.quantile(z, 0.05) + 0.3
    F = np.stack([fl @ e1, fl @ e2], 1)[air]; U = np.stack([vl @ e1, vl @ e2], 1)[air]
    X = np.concatenate([-U.reshape(-1, order="F")[:, None], np.kron(np.eye(2), np.ones((len(U), 1)))], 1); y = F.reshape(-1, order="F")
    beta = np.linalg.lstsq(X, y, rcond=None)[0]; r = (y - X @ beta).reshape(2, -1).T; rn = np.linalg.norm(r, axis=1)
    wn = np.linalg.norm(wl[air], axis=1); thr = np.abs(fl[air] @ zb - (fl[air] @ zb).mean()); sp = np.linalg.norm(U, axis=1)
    jerk = np.linalg.norm(np.gradient(fl[air], axis=0) * 200.0, axis=1)
    res = dict(k=float(beta[0]), resid_rms=float(np.sqrt(np.mean(rn ** 2))), air_frac=float(air.mean()))
    for name, q in (("w", wn), ("thrust", thr), ("speed", sp), ("jerk", jerk)):
        qs = np.quantile(q, [0, 0.5, 0.8, 0.95, 1.0]); res[name] = [(float(qs[i]), float(qs[i + 1]), float(np.sqrt(np.mean(rn[(q >= qs[i]) & (q <= qs[i + 1])] ** 2)))) for i in range(4)]
        res[name + "_corr"] = float(np.corrcoef(q, rn)[0, 1])
    out[n] = res
    print("%-16s k %.3f resid %.3f m/s^2 (vel-eq %.2f m/s) air %.2f | corr |w| %.2f thrust %.2f speed %.2f jerk %.2f | resid by |w| quantile bins %s" % (
        n, res["k"], res["resid_rms"], res["resid_rms"] / res["k"], res["air_frac"], res["w_corr"], res["thrust_corr"], res["speed_corr"], res["jerk_corr"], " ".join("%.3f" % b[2] for b in res["w"])))
json.dump(out, open("~/SamsungPRISM_FiveBlindSeconds/opt/notes/an05_drag_residuals.json", "w"), indent=1)
