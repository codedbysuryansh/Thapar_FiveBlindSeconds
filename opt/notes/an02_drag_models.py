"""an02: which rotor-drag measurement model explains the EuRoC body-frame specific force best? Clean IMU + ground-truth velocity
and orientation (analysis only: model-structure identification; shared coefficients judged leave-one-flight-out, per-flight
per-axis offsets removed by within-flight centring, as the smoother's drag-bias states would absorb them).
Models (thrust frame e1, e2 perpendicular to the mean specific force; u = body velocity on e1, e2; w = body velocity on the thrust axis):
  M0  f_i = -k u_i                 (champion)        M1  per-axis k_i            M2  -k (fz/mean fz)^0.5 u_i
  M3  -k u_i - c u_i |u|           M4  -k u_i - c |w| u_i                       M5  full 2x2 drag matrix
Signals low-passed (zero-phase Butterworth, 2nd order) at fc; velocity-equivalent noise = residual RMS / k."""
import os, sys, json, numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, "~/SamsungPRISM_FiveBlindSeconds/opt/harness")
import fbsopt as H
def load(n):
    imu = np.loadtxt(os.path.join(H.mav0(n), "imu0", "data.csv"), delimiter=",", comments="#")
    gt = np.loadtxt(os.path.join(H.mav0(n), "state_groundtruth_estimate0", "data.csv"), delimiter=",", comments="#")
    t = imu[:, 0]; a = imu[:, 4:7]; tg = gt[:, 0]; sel = (t >= tg[0]) & (t <= tg[-1]); t, a = t[sel], a[sel]
    R = Rot.from_quat(gt[:, [5, 6, 7, 4]]); v = gt[:, 8:11]; ba = gt[:, 14:17]
    vi = np.stack([np.interp(t, tg, v[:, j]) for j in range(3)], 1); bai = np.stack([np.interp(t, tg, ba[:, j]) for j in range(3)], 1)
    idx = np.clip(np.searchsorted(tg, t), 0, len(tg) - 1); return t * 1e-9, a - bai, R[idx].inv().apply(vi), vi
def lp(x, fc, fs=200.0):
    b, a = butter(2, fc / (fs / 2)); return filtfilt(b, a, x, axis=0)
def frame(f):
    zb = f.mean(0); zb /= np.linalg.norm(zb); h = np.array([0, 1.0, 0]) if abs(zb[1]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(zb, h); e1 /= np.linalg.norm(e1); return zb, e1, np.cross(zb, e1)
def cols(model, U, fz, W):
    n = len(U); z = np.zeros(n); sp = np.linalg.norm(U, axis=1); s = np.sqrt(np.maximum(fz, 1e-3) / fz.mean())
    if model == "M0": c = [(-U[:, 0], -U[:, 1])]
    elif model == "M1": c = [(-U[:, 0], z), (z, -U[:, 1])]
    elif model == "M2": c = [(-U[:, 0] * s, -U[:, 1] * s)]
    elif model == "M3": c = [(-U[:, 0], -U[:, 1]), (-U[:, 0] * sp, -U[:, 1] * sp)]
    elif model == "M4": c = [(-U[:, 0], -U[:, 1]), (-U[:, 0] * np.abs(W), -U[:, 1] * np.abs(W))]
    else: c = [(-U[:, 0], z), (-U[:, 1], z), (z, -U[:, 0]), (z, -U[:, 1])]
    return c
def centred(model, F, U, fz, W):
    """Design and response with each axis block centred (removes per-flight, per-axis offsets)."""
    c = cols(model, U, fz, W); Xa = np.stack([x0 for x0, _ in c], 1); Xb = np.stack([x1 for _, x1 in c], 1)
    X = np.concatenate([Xa - Xa.mean(0), Xb - Xb.mean(0)]); y = np.concatenate([F[:, 0] - F[:, 0].mean(), F[:, 1] - F[:, 1].mean()]); return X, y
out = {}
for fc in (0.5, 2.0):
    data = {}
    for n in H.EUROC:
        t, f, vS, vW = load(n); fl = lp(f, fc); vl = lp(vS, fc); zb, e1, e2 = frame(f); fly = np.linalg.norm(vW, axis=1) > 0.05; dec = 10
        data[n] = tuple(x[fly][::dec] for x in (np.stack([fl @ e1, fl @ e2], 1), np.stack([vl @ e1, vl @ e2], 1), fl @ zb, vl @ zb))
    for model in ("M0", "M1", "M2", "M3", "M4", "M5"):
        res = {}
        for n in H.EUROC:
            XY = [centred(model, *data[m]) for m in H.EUROC if m != n]; beta = np.linalg.lstsq(np.concatenate([x for x, _ in XY]), np.concatenate([y for _, y in XY]), rcond=None)[0]
            X, y = centred(model, *data[n]); r = y - X @ beta; k = beta[0] if model not in ("M1", "M5") else 0.5 * (beta[0] + beta[-1])
            res[n] = dict(resid=float(np.sqrt(np.mean(r ** 2))), vel_noise=float(np.sqrt(np.mean(r ** 2)) / k), beta=[float(x) for x in beta])
        out["%s_fc%g" % (model, fc)] = res
        print("fc %.1f Hz %s  mean velocity-equivalent noise %.4f m/s  beta(first flight) %s | %s" % (fc, model, np.mean([v["vel_noise"] for v in res.values()]), np.round(res["MH_01_easy"]["beta"], 4),
              " ".join("%.3f" % v["vel_noise"] for v in res.values())), flush=True)
json.dump(out, open("~/SamsungPRISM_FiveBlindSeconds/opt/notes/an02_drag_models.json", "w"), indent=1)
