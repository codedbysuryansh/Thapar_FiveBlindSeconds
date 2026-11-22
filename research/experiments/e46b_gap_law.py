#!/usr/bin/env python3
"""E46b - the gravity-smoother gap law k2 under LONG gaps (our stress suite; the official script only drops <= 50 samples).
k2 in {0.1, 0.25, 0.5, 1}: theta(T) = min(k2 T^2 / 2, 1 rad). Conditions: 3 x 0.25 s, 3 x 2 s, 3 x 4 s, 6 x 2 s gaps (NaN rows), 3 x 2 s removed
rows, L2 and L3 (gaps + noise + bias + rotation). 17 sequences x 2 seeds, failure at 60 %, blind rows as zeros. Const (|C) and spline (|S)."""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3, fbs6 as f
from e19_stress import COND
CONDS = ["gap_3x0.25", "gap_3x2", "gap_3x4", "gap_6x2", "gap_removed", "L2", "L3", "clean"]; K2S = [0.1, 0.25, 0.5, 1.0]
def run(args):
    c, n, seed = args; kw = dict(COND[c]); m = p3.mission(n, kw.pop("lvl", "clean"), 1, seed, **kw); nb = m.nb; dtn = int(np.median(np.diff(m.s.tn))); tb = m.s.tn[m.iT] + dtn * np.arange(1, nb + 1)
    tn2 = np.concatenate([m.tn, tb]); w2 = np.vstack([m.w, np.zeros((nb, 3))]); a2 = np.vstack([m.a, np.zeros((nb, 3))]); out = {}
    for k2 in K2S:
        f.fbs5.GS_K2 = k2; e = f.estimate_off(tn2, w2, a2, k_drag=m.kd, fe="v5", gs=True, be=dict(gap_ref=2.0), both=True)
        for k in ("C", "S"): out["k2=%g|%s" % (k2, k)] = p3.score(m, e["p" + k], e["q" + k])
    return dict(name=n, lvl=c, seed=seed, m=out)
if __name__ == "__main__":
    names = f.available()
    with Pool(10) as p: res = p.map(run, [(c, n, sd) for c in CONDS for n in names for sd in range(2)], chunksize=1)
    json.dump(res, open(os.path.join(p3.ROOT, "research/results/e46b_gap_law.json"), "w"))
    for grp, sel, kind in (("EuRoC (spline)", lambda n: not n.startswith("room"), "S"), ("TUM (const)", lambda n: n.startswith("room"), "C")):
        print("\n%s mean ATE / tilt / ori        " % grp + "   ".join("k2=%-18g" % k for k in K2S))
        for c in CONDS:
            rr = [r for r in res if r["lvl"] == c and sel(r["name"])]
            print("  %-12s " % c + "   ".join("%.3f / %.2f / %5.1f  " % tuple(np.nanmean([r["m"]["k2=%g|%s" % (k2, kind)][x] for r in rr]) for x in ("ate", "tilt", "ori")) for k2 in K2S))
