#!/usr/bin/env python3
"""E02 - IMU-only position estimators on the real, undegraded streams. Research instrument.
Ground truth is used only to score. Run: .venv/bin/python research/experiments/e02_baselines.py
Writes research/results/e02_baselines.json"""
import sys, os, json, itertools, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import fbs
ELL = [0.5, 1, 2, 4, 8]; SP = [0.25, 0.5, 1, 2]; TB = [5, 20]; SB = [0.1, 0.3]
GRID = list(itertools.product(ELL, SP, TB, SB))
def score(s, p, ok=None):
    ok = s.ok if ok is None else ok; Rm, _ = fbs.umeyama(p[ok], s.gt_p[ok])
    return {"ate_se3": fbs.ate(p[ok], s.gt_p[ok]), "ate_yaw": fbs.ate(p[ok], s.gt_p[ok], "yaw"), "rpe1": fbs.rpe_disp(s.t[ok], p[ok], s.gt_p[ok], Rm)}
def work(n):
    s = fbs.load(n); r = {"name": n, "group": fbs.group(n)}
    q, bg, quiet = fbs.estimate_attitude(s.t, s.w, s.a); r["quiet"] = bool(quiet); r["bg"] = bg.tolist()
    r["tilt_deg"] = float(np.mean(fbs.tilt_deg(q[s.ok], s.gt_q[s.ok])))
    aw = fbs.world_acc_centered(q, s.a); sn = fbs.hf_std(aw); r["sn"] = sn; r["acc_lp_std"] = float(fbs.lowpass(aw, 200.0, 5.0).std())
    z = np.zeros_like(aw); r["motionless"] = score(s, z + 1e-9 * np.arange(len(z))[:, None] * 0)
    r["strapdown"] = score(s, fbs.strapdown_pos(s.t, aw))
    for fc in [0.05, 0.1, 0.2, 0.4]: r["heave_hp_%.2f" % fc] = score(s, fbs.highpass_pos(s.t, aw, fc))
    sa = r["acc_lp_std"]
    xs = fbs.run_linear(s.t, aw, fbs.model_singer(1.0, sa, 20.0, 0.2), sn)[0]; r["singer_rts"] = score(s, xs[:, :, 0])
    for tv in [1.0, 3.0]:
        xs = fbs.run_linear(s.t, aw, fbs.model_ou_vel(tv, 0.5, 1.0, sa, 20.0, 0.2), sn)[0]; r["ouvel_rts_%g" % tv] = score(s, xs[:, :, 0])
    g = []
    for (ell, sp, tb, sb) in GRID:
        xs, Ps, xf, Pf, ll = fbs.run_linear(s.t, aw, fbs.model_matern52(ell, sp, tb, sb), sn)
        g.append([ell, sp, tb, sb, score(s, xs[:, :, 0])["ate_se3"], float(ll), score(s, xf[:, :, 0])["ate_se3"], fbs.rpe_disp(s.t[s.ok], xs[s.ok, :, 0], s.gt_p[s.ok], fbs.umeyama(xs[s.ok, :, 0], s.gt_p[s.ok])[0])])
    r["grid"] = g
    # diagnostic only: ground-truth orientation (oracle attitude) to isolate the attitude contribution
    ok = s.ok; i0, i1 = np.where(ok)[0][[0, -1]]; sl = slice(i0, i1 + 1)
    awo = fbs.world_acc_centered(s.gt_q[sl], s.a[sl]); sno = fbs.hf_std(awo); go = []
    class V: pass
    v = V(); v.t = s.t[sl]; v.gt_p = s.gt_p[sl]; v.ok = s.ok[sl]
    for (ell, sp, tb, sb) in GRID:
        xs = fbs.run_linear(v.t, awo, fbs.model_matern52(ell, sp, tb, sb), sno)[0]; go.append(score(v, xs[:, :, 0])["ate_se3"])
    r["grid_oracle_att"] = go
    return r
if __name__ == "__main__":
    names = fbs.available()
    with Pool(min(10, len(names))) as p: res = p.map(work, names)
    R = {r["name"]: r for r in res}; names = [r["name"] for r in res]
    G = np.array([[x[4] for x in R[n]["grid"]] for n in names]); L = np.array([[x[5] for x in R[n]["grid"]] for n in names]); GO = np.array([R[n]["grid_oracle_att"] for n in names])
    M = np.array([R[n]["motionless"]["ate_se3"] for n in names]); skill = 1 - G / M[:, None]; grp = np.array([R[n]["group"] for n in names]); plat = np.array([fbs.platform(n) for n in names])
    def pick(train): return int(np.argmax(skill[train].mean(0)))
    sel = {"loso": [], "loeo": [], "lopo": [], "ml": [], "oracle": [], "oracle_att_loso": []}
    for i, n in enumerate(names):
        idx = np.arange(len(names))
        sel["loso"].append(G[i, pick(idx != i)]); sel["loeo"].append(G[i, pick(grp != grp[i])]); sel["lopo"].append(G[i, pick(plat != plat[i])] if (plat != plat[i]).any() else np.nan)
        sel["ml"].append(G[i, int(np.argmax(L[i]))]); sel["oracle"].append(G[i].min())
        so = 1 - GO / M[:, None]; sel["oracle_att_loso"].append(GO[i, int(np.argmax(so[idx != i].mean(0)))])
        R[n]["sel"] = {k: float(v[-1]) for k, v in sel.items()}; R[n]["ml_cfg"] = GRID[int(np.argmax(L[i]))]; R[n]["oracle_cfg"] = GRID[int(np.argmin(G[i]))]
    best_all = pick(np.ones(len(names), bool)); R["_global_best_cfg"] = GRID[best_all]
    json.dump(R, open(os.path.join(fbs.ROOT, "research/results/e02_baselines.json"), "w"), indent=1)
    cols = ["motionless", "strapdown", "heave_hp_0.10", "heave_hp_0.20", "singer_rts", "ouvel_rts_3"]
    print("ATE (SE3-aligned RMSE, m)   tilt  " + " ".join("%13s" % c[-13:] for c in cols) + "   GP_loso  GP_loeo  GP_lopo    GP_ml GP_oracle  GP_oratt  filter(best)")
    for i, n in enumerate(names):
        print("%-16s %s %5.2f " % (n, "Q" if R[n]["quiet"] else "-", R[n]["tilt_deg"]) + " ".join("%13.3f" % R[n][c]["ate_se3"] for c in cols) + "  " + " ".join("%8.3f" % sel[k][i] for k in ["loso", "loeo", "lopo", "ml", "oracle", "oracle_att_loso"]) + "  %8.3f" % R[n]["grid"][best_all][6])
    print("global best cfg (ell, sp, tau_b, sb):", GRID[best_all])
    for gname in ["MH", "V1", "V2", "TUM", "ALL"]:
        m = np.ones(len(names), bool) if gname == "ALL" else grp == gname
        if m.sum(): print("%-4s mean ATE: motionless %.3f | GP loso %.3f loeo %.3f lopo %.3f ml %.3f oracle %.3f oracle-att %.3f | skill(loso) %.1f%%  skill(oracle-att) %.1f%%" % (
            gname, M[m].mean(), *(np.nanmean(np.array(sel[k])[m]) for k in ["loso", "loeo", "lopo", "ml", "oracle", "oracle_att_loso"]),
            100 * np.mean(1 - np.array(sel["loso"])[m] / M[m]), 100 * np.mean(1 - np.array(sel["oracle_att_loso"])[m] / M[m])))
