#!/usr/bin/env python3
"""E05 - does the rotor-drag relation (in-plane specific force = -k * in-plane velocity) hold on the real EuRoC IMU?
Research instrument. Ground truth velocity is used to MEASURE the physical relation (k would be learned on training
sequences only). Thrust axis = mean specific-force direction in the sensor frame (no ground truth needed)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import fbs
out = {}; rows = []
print("seq        thrust_axis(sensor)      flight%   k[1/s]  bias_inplane[m/s2]   R2    resid[m/s2]  => vel noise[m/s]   speed_rms  k(LP0.5Hz) R2(0.5Hz)")
for n in [x for x in fbs.available() if not x.startswith("room")]:
    s = fbs.load(n); ok = s.ok.copy(); fs = 200.0
    zb = s.a.mean(0); zb /= np.linalg.norm(zb)
    vib = np.sqrt(fbs.lowpass(np.sum(np.diff(s.a, axis=0, prepend=s.a[:1]) ** 2, 1), fs, 1.0).clip(0)); flying = vib > 0.5 * np.median(vib)
    vs = fbs.qrot(fbs.qinv(s.gt_q), s.gt_v)                       # GT velocity in the sensor frame
    e1 = np.cross(zb, [0, 1.0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(zb, e1)
    res = {}
    for fc in [2.0, 0.5]:
        al = fbs.lowpass(s.a, fs, fc); vl = fbs.lowpass(vs, fs, fc)
        m = ok & flying & (np.linalg.norm(s.gt_v, axis=1) > 0.05)
        A = np.stack([al[m] @ e1, al[m] @ e2], 1); V = np.stack([vl[m] @ e1, vl[m] @ e2], 1)
        X = np.zeros((2 * m.sum(), 3)); X[0::2, 0] = -V[:, 0]; X[0::2, 1] = 1; X[1::2, 0] = -V[:, 1]; X[1::2, 2] = 1; y = A.reshape(-1)
        th, *_ = np.linalg.lstsq(X[::21], y[::21], rcond=None); r = y - X @ th; r2 = 1 - r.var() / (y - np.repeat(A.mean(0)[None], len(A), 0).reshape(-1)).var()
        res[fc] = {"k": float(th[0]), "bias": th[1:].tolist(), "r2": float(r2), "resid": float(r.std()), "vel_noise": float(r.std() / abs(th[0])), "speed_rms": float(np.sqrt((V ** 2).sum(1).mean()))}
    out[n] = {"thrust_axis": zb.tolist(), "flight_frac": float(flying[ok].mean()), "fits": {str(k): v for k, v in res.items()}}
    a_, b_ = res[2.0], res[0.5]
    print("%-9s [%5.2f %5.2f %5.2f]   %5.1f   %6.3f   [%6.3f %6.3f]   %5.2f   %6.3f       %6.3f          %6.3f    %6.3f    %5.2f" % (n[:9], *zb, 100 * out[n]["flight_frac"], a_["k"], *a_["bias"], a_["r2"], a_["resid"], a_["vel_noise"], a_["speed_rms"], b_["k"], b_["r2"]))
json.dump(out, open(os.path.join(fbs.ROOT, "research/results/e05_drag.json"), "w"), indent=1)
