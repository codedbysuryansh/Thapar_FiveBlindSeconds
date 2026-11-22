#!/usr/bin/env python3
"""E01b - how well can a constant gyro bias be recovered from the IMU stream alone? Research instrument.
Estimators (no ground truth): LS tilt-drift (fbs.attitude), per-axis median, per-axis mode, quiet-window mean,
and 'drift+median' (horizontal components from tilt drift, vertical from the median world-vertical rate).
Scored against the EuRoC ground-truth bias and, for every sequence, by heading drift against ground truth."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import fbs
from scipy.spatial.transform import Rotation as Rot
def mode1(x, bw):
    h, e = np.histogram(x, bins=np.arange(x.min(), x.max() + bw, bw)); k = np.convolve(h, np.ones(5) / 5, "same"); i = np.argmax(k); return 0.5 * (e[i] + e[i + 1])
def quiet_mean(t, w, a, frac=0.05, win=0.5):
    fs = 1 / np.median(np.diff(t)); n = int(win * fs); m = len(t) // n
    W = w[:m * n].reshape(m, n, 3); A = a[:m * n].reshape(m, n, 3)
    sc = W.std(1).max(1) / np.median(W.std(1).max(1)) + A.std(1).max(1) / np.median(A.std(1).max(1)); k = max(3, int(frac * m)); o = np.argsort(sc)[:k]
    return W[o].mean((0, 1)), float(W[o].std(1).max(1).mean()), float(A[o].std(1).max(1).mean())
def yaw_drift(s, bg):
    q, _ = fbs.attitude(s.t, s.w, s.a, bg0=bg, iters=0); ok = s.ok
    rel = fbs.qmul(s.gt_q[ok], fbs.qinv(q[ok])); yaw = np.unwrap(Rot.from_quat(rel).as_euler("zyx")[:, 0]); tt = s.t[ok]
    sl = np.polyfit(tt, yaw, 1)[0]; res = yaw - np.polyval(np.polyfit(tt, yaw, 1), tt)
    return float(np.degrees(sl)), float(np.degrees(yaw[-1] - yaw[0])), float(np.degrees(np.abs(yaw - yaw[0]).max()))
out = {}
print("seq       method        bias_est (mrad/s)        |err| vs GT   yaw-rate err (deg/s)  net yaw drift (deg)  max |yaw err| (deg)")
for n in fbs.available():
    s = fbs.load(n); r = {}
    q, b_ls = fbs.attitude(s.t, s.w, s.a)
    b_med = np.median(s.w, 0); b_mode = np.array([mode1(s.w[:, i], 0.002) for i in range(3)]); b_q, qg, qa = quiet_mean(s.t, s.w, s.a)
    # drift + median: start from LS horizontal estimate, replace world-vertical component by median vertical rate
    q0, b_h = fbs.attitude(s.t, s.w, s.a, ridge=1e-1)
    wz = fbs.qrot(q0, s.w - b_h)[:, 2]; gdir = fbs.qrot(fbs.qinv(q0), np.array([0, 0, 1.0])).mean(0); gdir /= np.linalg.norm(gdir)
    b_dm = b_h + np.median(wz) * gdir
    q1, b_h1 = fbs.attitude(s.t, s.w, s.a, bg0=b_q, iters=0)
    cands = {"zero": np.zeros(3), "ls_drift": b_ls, "median": b_med, "mode": b_mode, "quiet5%": b_q, "ls_ridge": b_h, "drift+median": b_dm}
    if s.gt_bg is not None: cands["GT_bias(oracle)"] = s.gt_bg.mean(0)
    for k, b in cands.items():
        yr, net, mx = yaw_drift(s, b); err = float(np.linalg.norm(b - s.gt_bg.mean(0))) if s.gt_bg is not None else float("nan")
        r[k] = {"bias": b.tolist(), "err_norm": err, "yaw_rate_deg_s": yr, "net_yaw_deg": net, "max_yaw_err_deg": mx}
        print("%-8s  %-15s [%6.1f %6.1f %6.1f]   %8.1f      %8.3f          %8.1f           %8.1f" % (n[:8], k, *(1e3 * b), 1e3 * err, yr, net, mx))
    r["quiet_gyro_std"] = qg; r["quiet_acc_std"] = qa; out[n] = r
json.dump(out, open(os.path.join(fbs.ROOT, "research/results/e01b_gyro_bias.json"), "w"), indent=1)
