#!/usr/bin/env python3
"""E05b - joint stationary-GP smoother with and without the rotor-drag velocity relation, on real IMU. Research instrument.
k (drag coefficient) and the other settings are chosen leave-one-sequence-out, leave-one-environment-out, or by the
marginal likelihood of the test stream itself (no ground truth). Ground truth is used only to score."""
import sys, os, json, itertools, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import fbs
GPS = [(1, 2, 20, 0.1), (2, 2, 20, 0.1), (4, 4, 20, 0.1), (8, 4, 20, 0.1)]
KS = [1e-9, 0.10, 0.15, 0.18, 0.22]; SDS = [0.5, 1.0, 2.0]
CFG = [(g, k, sd) for g in GPS for k in KS for sd in SDS]
def work(n):
    s = fbs.load(n); q, bg, quiet = fbs.estimate_attitude(s.t, s.w, s.a); sn = fbs.hf_std(fbs.world_acc_centered(q, s.a)); ok = s.ok; out = []
    for (g, k, sd) in CFG:
        r = fbs.run_joint(s.t, q, s.a, g, sn, k_drag=k, sd=sd); p = r["p"]; Rm, _ = fbs.umeyama(p[ok], s.gt_p[ok])
        out.append([fbs.ate(p[ok], s.gt_p[ok]), r["ll"], fbs.rpe_disp(s.t[ok], p[ok], s.gt_p[ok], Rm), fbs.ate(r["pf"][ok], s.gt_p[ok])])
    return n, out, fbs.ate(np.zeros((ok.sum(), 3)), s.gt_p[ok])
if __name__ == "__main__":
    names = fbs.available()
    with Pool(min(10, len(names))) as p: res = p.map(work, names)
    names = [r[0] for r in res]; T = np.array([r[1] for r in res]); M = np.array([r[2] for r in res]); G = T[:, :, 0]; L = T[:, :, 1]
    grp = np.array([fbs.group(n) for n in names]); mav = np.array([fbs.platform(n) == "mav" for n in names]); skill = 1 - G / M[:, None]
    isgp = np.array([c[1] < 1e-6 for c in CFG]); out = {}
    print("seq               motionless  GP-only(loso)  +drag(loso)  +drag(loeo)  +drag(ML)  oracle   ML-chosen (gp, k, sd)                 filter-only(loso cfg)")
    for i, n in enumerate(names):
        same = mav == mav[i]; tr = same & (np.arange(len(names)) != i); tre = same & (grp != grp[i])
        c_gp = int(np.argmax(np.where(isgp, skill[tr].mean(0), -9))); c_lo = int(np.argmax(skill[tr].mean(0))); c_le = int(np.argmax(skill[tre].mean(0))) if tre.any() else c_lo
        sdsel = CFG[c_lo][2]; ml_pool = np.array([c[2] == sdsel for c in CFG]); c_ml = int(np.argmax(np.where(ml_pool, L[i], -np.inf)))
        out[n] = {"motionless": M[i], "gp_loso": G[i, c_gp], "drag_loso": G[i, c_lo], "drag_loeo": G[i, c_le], "drag_ml": G[i, c_ml], "oracle": G[i].min(), "cfg_loso": CFG[c_lo], "cfg_ml": CFG[c_ml], "cfg_oracle": CFG[int(np.argmin(G[i]))],
                  "rpe_gp": T[i, c_gp, 2], "rpe_drag": T[i, c_lo, 2], "filter_loso": T[i, c_lo, 3], "table": T[i].tolist()}
        print("%-16s %9.3f %12.3f %12.3f %12.3f %10.3f %8.3f   %-36s %8.3f" % (n, M[i], G[i, c_gp], G[i, c_lo], G[i, c_le], G[i, c_ml], G[i].min(), str(CFG[c_ml]), T[i, c_lo, 3]))
    for gname in ["MH", "V1", "V2", "TUM", "EUROC", "ALL"]:
        m = np.ones(len(names), bool) if gname == "ALL" else (mav if gname == "EUROC" else grp == gname)
        if m.sum():
            f = lambda k: np.mean([out[n][k] for n, mm in zip(names, m) if mm]); sk = lambda k: 100 * np.mean([1 - out[n][k] / out[n]["motionless"] for n, mm in zip(names, m) if mm])
            print("%-5s mean ATE motionless %.3f | GP %.3f (skill %.1f%%) | +drag loso %.3f (%.1f%%) loeo %.3f (%.1f%%) ML %.3f (%.1f%%) | RPE1s GP %.3f drag %.3f" % (gname, f("motionless"), f("gp_loso"), sk("gp_loso"), f("drag_loso"), sk("drag_loso"), f("drag_loeo"), sk("drag_loeo"), f("drag_ml"), sk("drag_ml"), f("rpe_gp"), f("rpe_drag")))
    out["_cfg"] = CFG; json.dump(out, open(os.path.join(fbs.ROOT, "research/results/e05b_drag_smoother.json"), "w"), indent=1, default=float)
