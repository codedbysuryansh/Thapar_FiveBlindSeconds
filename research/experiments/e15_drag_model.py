#!/usr/bin/env python3
"""E15 - ORACLE DIAGNOSTIC (ground-truth velocity used to MEASURE physics, never for inference):
can a richer in-plane specific-force model shrink the rotor-drag residual on EuRoC?
A: f_i = -k v_i + b_i                      (Phase 1B model)
B: A + accelerometer lever arm r (3)       f += e_i . (wdot x r + w x (w x r))
C: B + thrust leakage eps_i (2)            f += eps_i * (thrust - mean thrust)      [thrust-axis misalignment]
D: A with k scaled by sqrt(thrust / g)     (rotor-speed dependence of the drag coefficient)
E: A with separate k for the two in-plane axes
F: C + D + E
Reported: residual std after 2 Hz and 0.5 Hz zero-phase low-pass, equivalent velocity noise (resid / k), R^2, and the
position random walk that the residual implies over 60 s if it were integrated (the quantity that limits odometry).
Run: .venv/bin/python research/experiments/e15_drag_model.py"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import fbs3 as f
def design(s, fc):
    fs = 200.0; zb = s.a.mean(0); zb /= np.linalg.norm(zb); e1 = np.cross(zb, [0, 1.0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(zb, e1)
    vib = np.sqrt(f.lowpass(np.sum(np.diff(s.a, axis=0, prepend=s.a[:1]) ** 2, 1), fs, 1.0).clip(0)); fly = vib > 0.5 * np.median(vib)
    bg = np.median(s.w[:400], 0) if s.gt_bg is None else np.median(s.gt_bg, 0); w = s.w - bg; wl = f.lowpass(w, fs, 15.0); wd = np.gradient(wl, 1 / fs, axis=0)
    vs = f.qrot(f.qinv(s.gt_q), s.gt_v); thr = s.a @ zb
    def L(j):                                    # lever-arm acceleration for unit r along sensor axis j
        r = np.zeros(3); r[j] = 1.0; return np.cross(wd, r) + np.cross(wl, np.cross(wl, r))
    lev = [L(j) for j in range(3)]; lp = lambda x: f.lowpass(x, fs, fc)
    m = s.ok & fly & (np.linalg.norm(s.gt_v, axis=1) > 0.05)
    cols = {}
    for i, e in enumerate((e1, e2)):
        cols[i] = dict(y=lp(s.a @ e)[m], v=lp(vs @ e)[m], vs=lp((vs @ e) * np.sqrt(np.clip(thr, 1, None) / 9.81))[m], lev=[lp(l @ e)[m] for l in lev], thr=lp(thr - thr[m].mean())[m])
    return cols, int(m.sum())
def fit(cols, n, model):
    Y = np.concatenate([cols[0]["y"], cols[1]["y"]]); z = np.zeros(n); X = []
    vk = "vs" if model in ("D", "F") else "v"
    if model in ("E", "F"): X += [np.concatenate([-cols[0][vk], z]), np.concatenate([z, -cols[1][vk]])]
    else: X += [np.concatenate([-cols[0][vk], -cols[1][vk]])]
    X += [np.concatenate([np.ones(n), z]), np.concatenate([z, np.ones(n)])]
    if model in ("B", "C", "F"): X += [np.concatenate([cols[0]["lev"][j], cols[1]["lev"][j]]) for j in range(3)]
    if model in ("C", "F"): X += [np.concatenate([cols[0]["thr"], z]), np.concatenate([z, cols[1]["thr"]])]
    X = np.stack(X, 1); th, *_ = np.linalg.lstsq(X[::21], Y[::21], rcond=None); r = Y - X @ th
    return th, float(r.std()), float(1 - r.var() / (Y - np.concatenate([np.full(n, cols[0]["y"].mean()), np.full(n, cols[1]["y"].mean())])).var()), r
if __name__ == "__main__":
    names = [x for x in f.available() if not x.startswith("room")]; out = {}
    for fc in (2.0, 0.5):
        print("\nlow-pass %.1f Hz: residual std [m/s^2] (R^2) per model, per-flight fit" % fc); agg = {m: [] for m in "ABCDEF"}
        for n in names:
            s = f.load(n); cols, nn = design(s, fc); row = []
            for model in "ABCDEF":
                th, rs, r2, r = fit(cols, nn, model); agg[model].append((rs, r2, abs(th[0]))); row.append("%s %.3f (%.2f)" % (model, rs, r2))
                if model in ("B", "C") and fc == 2.0: out.setdefault(n, {})[model] = th.tolist()
            print("  %-16s %s" % (n, "  ".join(row)))
        print("  MEAN             " + "  ".join("%s %.3f (%.2f) vnoise %.2f m/s" % (m, np.mean([a[0] for a in agg[m]]), np.mean([a[1] for a in agg[m]]), np.mean([a[0] / a[2] for a in agg[m]])) for m in "ABCDEF"))
    # spectrum of the model-A residual: is it white (integrates to a random walk) or red (integrates worse)?
    print("\nmodel A residual, band-limited std [m/s^2] in octave bands (per-flight mean), and the velocity-equivalent:")
    from scipy.signal import welch
    acc = []
    for n in names:
        s = f.load(n); cols, nn = design(s, 10.0); th, rs, r2, r = fit(cols, nn, "A"); fr, P = welch(r[:nn], fs=200.0, nperseg=8192); acc.append((np.interp([0.03, 0.06, 0.12, 0.25, 0.5, 1, 2, 4], fr, P), abs(th[0])))
    P = np.mean([a[0] for a in acc], 0); k = np.mean([a[1] for a in acc])
    print("  f [Hz]            " + "  ".join("%6.2f" % x for x in [0.03, 0.06, 0.12, 0.25, 0.5, 1, 2, 4]))
    print("  PSD [(m/s^2)^2/Hz] " + "  ".join("%6.4f" % x for x in P))
    print("  white level assumed by sd=2.0 at 200 Hz: %.4f ; by sd=1.0: %.4f ; mean k %.3f" % (2.0 ** 2 / 100, 1.0 / 100, k))
    print("  position random walk after 60 s from integrating a white velocity error at the 0.03-0.12 Hz level: %.2f m per axis" % np.sqrt(P[:3].mean() / k ** 2 * 60 / 2))
    json.dump(out, open(os.path.join(f.fbs.ROOT, "research/results/e15_drag_model.json"), "w"), indent=1)
