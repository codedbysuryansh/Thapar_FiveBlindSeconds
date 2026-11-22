#!/usr/bin/env python3
"""E11b - Scenario A done properly: two-component fusion. Supplied checkpoints fix the SLOW part of the trajectory
(interpolated residuals), the IMU smoother supplies the FAST part. Tested by withholding each interior checkpoint
(leave-one-out) so the score is always at a point the algorithm was not given. Baseline: spline through the supplied
checkpoints with no IMU at all. K = 6 or 12 checkpoints. Ground truth = supplied checkpoint coordinates + scoring only.
Run: .venv/bin/python research/experiments/e11b_checkpoint_fusion.py"""
import sys, os, json, numpy as np
from multiprocessing import Pool
from scipy.interpolate import CubicSpline
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbs3 as f
from e10_mission import LEVELS, k_loso
def interp(tq, tk, y, kind):
    if kind == "lin": return np.stack([np.interp(tq, tk, y[:, i]) for i in range(3)], 1)
    cs = CubicSpline(tk, y, bc_type="natural"); return cs(np.clip(tq, tk[0], tk[-1]))
def scen(args):
    n, lvl, seed = args; s = f.load(n); fs = 1 / np.median(np.diff(s.t)); oi = np.where(s.ok)[0]; t0, t1 = s.t[oi[0]], s.t[oi[-1]]; iT = int(np.searchsorted(s.t, t1 - 5.05))
    sl = slice(0, iT + 1); w, a, valid, meta = f.degrade(s.t[sl], s.w[sl], s.a[sl], LEVELS[lvl], seed=1000 * seed + sum(n.encode()) % 997); tn = s.tn[sl]; gp = s.gt_p[sl]; ok = s.ok[sl]; t = s.t[sl]
    near = lambda c: int(c if ok[c] else oi[oi <= iT][np.argmin(np.abs(oi[oi <= iT] - c))])
    E = {"spring": f.estimate3(tn, w, a)["p"]}; lock = f.estimate3(tn, w, a)["lock"]
    E["m44_drag"] = f.estimate3(tn, w, a, model=f.model_matern52(4, 4, 20, 0.1), drag=(k_loso(n), 2.0))["p"] if lock < 0.04 else f.estimate3(tn, w, a, model=f.model_matern52(*f.MATERN))["p"]
    out = {"name": n, "lvl": lvl}
    for K in (6, 12):
        cp = np.array([near(c) for c in np.searchsorted(s.t, np.linspace(t0 + 10, s.t[iT], K)).clip(oi[0], iT)])
        for en, p in E.items():
            Rb, tb = f.umeyama(p[ok], gp[ok]); out["B_%s_K%d" % (en, K)] = float(np.linalg.norm(p[cp[1:-1]] @ Rb.T + tb - gp[cp[1:-1]], axis=1).mean())
            for kind in ("lin", "cub"):
                loo = []; none = []
                for j in range(1, K - 1):
                    idx = np.delete(cp, j); Rm, tt = f.umeyama(p[idx], gp[idx]); pa = p @ Rm.T + tt; r = gp[idx] - pa[idx]
                    loo.append(np.linalg.norm(pa[cp[j]] + interp(t[cp[j]:cp[j] + 1], t[idx], r, kind)[0] - gp[cp[j]])); none.append(np.linalg.norm(interp(t[cp[j]:cp[j] + 1], t[idx], gp[idx], kind)[0] - gp[cp[j]]))
                out["A_%s_%s_K%d" % (en, kind, K)] = float(np.mean(loo)); out["noIMU_%s_K%d" % (kind, K)] = float(np.mean(none))
            Rm, tt = f.umeyama(p[cp], gp[cp]); pa = p @ Rm.T + tt; full = pa + interp(t, t[cp], gp[cp] - pa[cp], "cub"); m = ok & (t >= t[cp[0]])
            out["Afull_%s_K%d" % (en, K)] = float(np.sqrt(np.mean(np.sum((full[m] - gp[m]) ** 2, 1)))); sp = interp(t, t[cp], gp[cp], "cub"); out["noIMUfull_K%d" % K] = float(np.sqrt(np.mean(np.sum((sp[m] - gp[m]) ** 2, 1))))
    return out
if __name__ == "__main__":
    names = f.available(); jobs = [(n, l, 0) for l in ("clean", "L2") for n in names]
    with Pool(10) as p: res = p.map(scen, jobs, chunksize=1)
    json.dump(res, open(os.path.join(f.fbs.ROOT, "research/results/e11b_checkpoint_fusion.json"), "w"), indent=1)
    for lvl in ("clean", "L2"):
        for grp, sel in [("ALL", lambda n: True), ("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room"))]:
            rr = [r for r in res if r["lvl"] == lvl and sel(r["name"])]; g = lambda k: np.mean([r[k] for r in rr])
            for K in (6, 12):
                print("%-5s %-5s K=%2d | withheld-checkpoint error: Scenario B spring %.2f m44_drag %.2f | no-IMU spline lin %.2f cub %.2f | fusion spring lin %.2f cub %.2f | fusion m44_drag lin %.2f cub %.2f || whole-path RMSE: no-IMU spline %.2f fusion spring %.2f fusion m44_drag %.2f" % (
                    lvl, grp, K, g("B_spring_K%d" % K), g("B_m44_drag_K%d" % K), g("noIMU_lin_K%d" % K), g("noIMU_cub_K%d" % K), g("A_spring_lin_K%d" % K), g("A_spring_cub_K%d" % K), g("A_m44_drag_lin_K%d" % K), g("A_m44_drag_cub_K%d" % K), g("noIMUfull_K%d" % K), g("Afull_spring_K%d" % K), g("Afull_m44_drag_K%d" % K)))
