#!/usr/bin/env python3
"""E03b - gap-aware front end (v2) versus v1 under dropouts and combined degradation; attitude time-constant scan.
Research instrument; corruptions are our assumptions. Run: .venv/bin/python research/experiments/e03b_dropout.py [n_seeds]"""
import sys, os, json, numpy as np
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbs
from e03_degrade import CFGS
RUN = ["clean", "noise_L2", "noise_L3", "gbias_L2", "gbias_L3", "gaps_3x0.5s", "gaps_3x1s", "gaps_3x2s", "combo_L1", "combo_L2", "combo_L3"]
VARS = {"v2_tau1.5": dict(tau=1.5), "v2_tau3": dict(tau=3.0), "v2_tau6": dict(tau=6.0), "v2_tau10": dict(tau=10.0),
        "v2_tau3_bj0.5": dict(tau=3.0, bj=0.5), "v2_tau6_bj0.5": dict(tau=6.0, bj=0.5), "v2_tau3_zero_bj0.5": dict(tau=3.0, bj=0.5, bridge="zero")}
def one(args):
    n, cname, seed = args; s = fbs.load(n); ok = s.ok
    w, a, valid, meta = fbs.degrade(s.t, s.w, s.a, CFGS[cname], seed=1000 * seed + sum(n.encode()) % 997)
    gq = fbs.qmul(s.gt_q, fbs.qinv(Rot.from_matrix(meta["Rr"]).as_quat())[None]); r = {"name": n, "cfg": cname, "motionless": fbs.ate(np.zeros((ok.sum(), 3)), s.gt_p[ok])}
    Rz, tz = fbs.umeyama(np.zeros((ok.sum(), 3)), s.gt_p[ok])
    for var, kw in VARS.items():
        q, bg, quiet, wf, af = fbs.front_end2(s.t, w, a, valid, tau=kw["tau"], bridge=kw.get("bridge", "lin"))
        p = fbs.gp_smooth(s.t, q, af, valid, bias_jump=kw.get("bj", 0.0))[0]
        r[var] = fbs.ate(p[ok], s.gt_p[ok]); r[var + "_tilt"] = float(np.mean(fbs.tilt_deg(q[ok & valid], gq[ok & valid])))
        if meta["gaps"]:
            Rm, tt = fbs.umeyama(p[ok], s.gt_p[ok]); ge = {"gp": [], "hold": [], "abs_gp": [], "abs_ml": [], "rot_zero": [], "rot_lin": [], "post2s": [], "post2s_ml": []}
            for (t0, dur) in meta["gaps"]:
                m = (s.t >= t0) & (s.t < t0 + dur) & ok
                if m.sum() < 10: continue
                i0 = np.where(m)[0][0] - 1; i1 = np.where(m)[0][-1] + 1; dg = s.gt_p[m] - s.gt_p[i0]
                ge["gp"].append(np.linalg.norm((p[m] - p[i0]) @ Rm.T - dg, axis=1).mean()); ge["hold"].append(np.linalg.norm(dg, axis=1).mean())
                ge["abs_gp"].append(np.linalg.norm(p[m] @ Rm.T + tt - s.gt_p[m], axis=1).mean()); ge["abs_ml"].append(np.linalg.norm(tz - s.gt_p[m], axis=1).mean())
                pm = (s.t >= t0 + dur) & (s.t < t0 + dur + 2.0) & ok
                if pm.sum() > 10: ge["post2s"].append(np.linalg.norm(p[pm] @ Rm.T + tt - s.gt_p[pm], axis=1).mean()); ge["post2s_ml"].append(np.linalg.norm(tz - s.gt_p[pm], axis=1).mean())
                # rotation across the gap: truth = integral of the real (undegraded) gyro; candidates: no rotation / linear-rate bridge
                tg = s.t[i0:i1 + 1]; wt = s.w[i0:i1 + 1] - (s.gt_bg.mean(0) if s.gt_bg is not None else 0.0); dq_true = fbs.integrate_gyro(tg, wt)[-1]
                wl = np.linspace(wt[0], wt[-1], len(tg)); dq_lin = fbs.integrate_gyro(tg, wl)[-1]
                ge["rot_zero"].append(fbs.ang_deg(dq_true, np.array([0, 0, 0, 1.0]))); ge["rot_lin"].append(fbs.ang_deg(dq_true, dq_lin))
            for k, v in ge.items(): r[var + "_gap_" + k] = float(np.mean(v)) if v else float("nan")
    return r
if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 2; names = fbs.available()
    jobs = [(n, c, sd) for c in RUN for n in names for sd in range(seeds if c != "clean" else 1)]
    with Pool(10) as p: res = p.map(one, jobs, chunksize=4)
    json.dump(res, open(os.path.join(fbs.ROOT, "research/results/e03b_dropout.json"), "w"))
    print("skill vs motionless %% (ALL/MAV/handheld) and mean tilt (deg) per variant")
    print("%-13s " % "config" + " ".join("%24s" % v for v in VARS))
    for c in RUN:
        rr = [r for r in res if r["cfg"] == c]; line = "%-13s " % c
        for var in VARS:
            sk = lambda sel: 100 * np.mean([1 - r[var] / r["motionless"] for r in rr if sel(r["name"])])
            line += " %5.1f/%5.1f/%5.1f t%4.2f " % (sk(lambda n: True), sk(lambda n: not n.startswith("room")), sk(lambda n: n.startswith("room")), np.mean([r[var + "_tilt"] for r in rr]))
        print(line)
    print("gap metrics (m): anchored GP / hold | absolute in-gap GP / motionless | absolute 2 s after gap GP / motionless | rotation across gap: none / linear-rate bridge (deg)")
    for c in RUN:
        rr = [r for r in res if r["cfg"] == c]
        if "v2_tau3_gap_gp" not in rr[0]: continue
        for var in ["v2_tau3", "v2_tau3_bj0.5", "v2_tau6_bj0.5", "v2_tau3_zero_bj0.5"]:
            g = lambda k, sel=(lambda n: True): np.nanmean([r[var + "_gap_" + k] for r in rr if sel(r["name"])])
            print("%-13s %-20s %.3f / %.3f | %.3f / %.3f | %.3f / %.3f | %.1f / %.1f   (handheld anchored %.3f / %.3f; MAV %.3f / %.3f)" % (c, var, g("gp"), g("hold"), g("abs_gp"), g("abs_ml"), g("post2s"), g("post2s_ml"), g("rot_zero"), g("rot_lin"),
                  g("gp", lambda n: n.startswith("room")), g("hold", lambda n: n.startswith("room")), g("gp", lambda n: not n.startswith("room")), g("hold", lambda n: not n.startswith("room"))))
