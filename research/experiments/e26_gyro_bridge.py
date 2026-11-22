#!/usr/bin/env python3
"""E26 - how should the gyro be bridged across a dropout?  Clean gyro, synthetic gaps every 12 s; the true rotation over
each gap is known from the removed samples (diagnostic use of the clean data, not of ground-truth poses).
Methods: lin0 = linear between the two end samples (v4); linM(x) = linear between the means of the last / first x seconds;
hold(x) = constant mean of both sides' x-second means; loc(x) = constant mean of the +-x s neighbourhood;
ar = linear between 0.25 s end means plus pull toward the +-10 s mean (OU-style, tau 1 s)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import fbs4 as f
def rot_err(wt, wb, dt):
    qt = f.integrate_gyro(np.arange(len(wt) + 1) * dt, np.vstack([wt, wt[-1:]]))[-1]; qb = f.integrate_gyro(np.arange(len(wb) + 1) * dt, np.vstack([wb, wb[-1:]]))[-1]
    return float(f.ang_deg(qt[None], qb[None])[0])
res = {}
for n in f.available():
    s = f.load(n); w = s.w; N = len(w); fs = 200.0; dt = 1 / fs
    for T in (0.25, 0.5, 1.0, 2.0, 4.0):
        g = int(T * fs)
        for i0 in range(int(12 * fs), N - int(12 * fs) - g, int(12 * fs)):
            i1 = i0 + g; wt = w[i0:i1]; u = (np.arange(g) + 1) / (g + 1); M = {}
            M["lin0"] = w[i0 - 1] + u[:, None] * (w[i1] - w[i0 - 1])
            for x in (0.1, 0.25, 0.5, 1.0):
                k = int(x * fs); a_, b_ = w[i0 - k:i0].mean(0), w[i1:i1 + k].mean(0); M["linM%.2g" % x] = a_ + u[:, None] * (b_ - a_); M["hold%.2g" % x] = np.broadcast_to(0.5 * (a_ + b_), (g, 3))
            for x in (2.0, 5.0, 10.0):
                k = int(x * fs); M["loc%g" % x] = np.broadcast_to(0.5 * (w[i0 - k:i0].mean(0) + w[i1:i1 + k].mean(0)), (g, 3))
            k = int(0.25 * fs); a_, b_ = w[i0 - k:i0].mean(0), w[i1:i1 + k].mean(0); mu = 0.5 * (w[i0 - 2000:i0].mean(0) + w[i1:i1 + 2000].mean(0)); tt = (np.arange(g) + 1) * dt; wgt = np.minimum(np.exp(-tt / 1.0), np.exp(-(T - tt) / 1.0))[:, None]
            M["ar"] = mu + wgt * ((a_ + u[:, None] * (b_ - a_)) - mu); M["zero"] = np.zeros((g, 3))
            for k_, v in M.items(): res.setdefault((f.fbs.platform(n), T, k_), []).append(rot_err(wt, v, dt))
keys = ["zero", "lin0", "linM0.1", "linM0.25", "linM0.5", "linM1", "hold0.1", "hold0.25", "hold0.5", "hold1", "loc2", "loc5", "loc10", "ar"]
json.dump({"%s|%g|%s" % k: v for k, v in res.items()}, open(os.path.join(f.fbs.ROOT, "research/results/e26_gyro_bridge.json"), "w"))
print("rotation error over the gap [deg], RMS (median):  " + "  ".join("%9s" % k for k in keys))
for plat in ("mav", "handheld"):
    for T in (0.25, 0.5, 1.0, 2.0, 4.0):
        print("%-8s T=%-4g n=%3d | " % (plat, T, len(res[(plat, T, "lin0")])) + "  ".join("%4.1f(%4.1f)" % (np.sqrt(np.mean(np.square(res[(plat, T, k)]))), np.median(res[(plat, T, k)])) for k in keys))
