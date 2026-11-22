#!/usr/bin/env python3
"""E03 - robustness of the stationary-GP smoother to synthetic degradation of the real IMU streams. Research instrument.
Corruptions are OUR assumptions (fbs.degrade), not the organiser's script. Ground truth only scores.
Variants: const R / adaptive R (rolling high-frequency variance) / adaptive R + adaptive attitude time constant.
Run: .venv/bin/python research/experiments/e03_degrade.py [n_seeds]"""
import sys, os, json, numpy as np
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import fbs
GP = (1, 2, 20, 0.1)
CFGS = {"clean": {}, "noise_L1": dict(gn=0.01, an=0.1), "noise_L2": dict(gn=0.05, an=0.5), "noise_L3": dict(gn=0.2, an=2.0),
        "noise_L2_exp": dict(gn=0.05, an=0.5, shape="exp"), "noise_L2_step": dict(gn=0.05, an=0.5, shape="step"),
        "gyro_noise_L3_only": dict(gn=0.2), "acc_noise_L3_only": dict(an=2.0),
        "gbias_L1": dict(gb=0.005), "gbias_L2": dict(gb=0.02), "gbias_L3": dict(gb=0.05), "gbias_L2_rw": dict(gb=0.02, bias_shape="rw"),
        "abias_L1": dict(ab=0.05), "abias_L2": dict(ab=0.2), "abias_L3": dict(ab=0.5), "abias_L2_rw": dict(ab=0.2, bias_shape="rw"),
        "gaps_3x0.5s": dict(gaps=[0.5] * 3), "gaps_3x1s": dict(gaps=[1.0] * 3), "gaps_3x2s": dict(gaps=[2.0] * 3), "rot_only": dict(rot=True),
        "combo_L1": dict(gn=0.01, an=0.1, gb=0.005, ab=0.05, gaps=[2.0] * 3, rot=True), "combo_L2": dict(gn=0.05, an=0.5, gb=0.02, ab=0.2, gaps=[2.0] * 3, rot=True),
        "combo_L3": dict(gn=0.2, an=2.0, gb=0.05, ab=0.5, gaps=[2.0] * 3, rot=True)}
def one(args):
    n, cname, seed = args; s = fbs.load(n); ok = s.ok; cfg = CFGS[cname]
    w, a, valid, meta = fbs.degrade(s.t, s.w, s.a, cfg, seed=1000 * seed + sum(n.encode()) % 997)
    gq = fbs.qmul(s.gt_q, fbs.qinv(Rot.from_matrix(meta["Rr"]).as_quat())[None]); r = {"name": n, "cfg": cname, "seed": seed, "motionless": fbs.ate(np.zeros((ok.sum(), 3)), s.gt_p[ok])}
    for var, (ad_tau, ad_R) in {"const": (False, False), "adaptR": (False, True), "adaptR_tau": (True, True)}.items():
        q, bg, quiet, wf, af, tau = fbs.front_end(s.t, w, a, valid, adaptive_tau=ad_tau)
        aw = fbs.qrot(q, af); aw = aw - aw[valid].mean(0)
        Rk = fbs.rolling_hf_std(s.t, aw, valid) ** 2 if ad_R else np.full(len(s.t), fbs.hf_std(aw[valid]) ** 2)
        xs = fbs.run_linear(s.t, aw, fbs.model_matern52(*GP), 0.0, mask=valid, Rk=Rk)[0]; p = xs[:, :, 0]
        r[var] = fbs.ate(p[ok], s.gt_p[ok]); r[var + "_tilt"] = float(np.mean(fbs.tilt_deg(q[ok & valid], gq[ok & valid]))); r[var + "_tau"] = tau
        if var == "const": r["strapdown"] = fbs.ate(fbs.strapdown_pos(s.t, aw)[ok], s.gt_p[ok])
        if var == "adaptR" and meta["gaps"]:
            Rm, tt = fbs.umeyama(p[ok], s.gt_p[ok]); ge = {"gp": [], "hold": [], "lin": [], "abs_gp": [], "abs_motionless": []}
            Rz, tz = fbs.umeyama(np.zeros((ok.sum(), 3)), s.gt_p[ok])
            for (t0, dur) in meta["gaps"]:
                m = (s.t >= t0) & (s.t < t0 + dur) & ok
                if m.sum() < 10: continue
                i0 = np.where(m)[0][0] - 1; i1 = np.where(m)[0][-1] + 1; dg = s.gt_p[m] - s.gt_p[i0]; lam = ((s.t[m] - s.t[i0]) / (s.t[i1] - s.t[i0]))[:, None]
                ge["gp"].append(np.linalg.norm((p[m] - p[i0]) @ Rm.T - dg, axis=1).mean()); ge["hold"].append(np.linalg.norm(dg, axis=1).mean())
                ge["lin"].append(np.linalg.norm((lam * (p[i1] - p[i0])) @ Rm.T - dg, axis=1).mean())
                ge["abs_gp"].append(np.linalg.norm(p[m] @ Rm.T + tt - s.gt_p[m], axis=1).mean()); ge["abs_motionless"].append(np.linalg.norm(tz - s.gt_p[m], axis=1).mean())
            for k, v in ge.items(): r["gap_" + k] = float(np.mean(v)) if v else float("nan")
    return r
if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 2; names = fbs.available()
    jobs = [(n, c, sd) for c in CFGS for n in names for sd in range(seeds if c != "clean" else 1)]
    with Pool(10) as p: res = p.map(one, jobs, chunksize=4)
    json.dump(res, open(os.path.join(fbs.ROOT, "research/results/e03_degrade.json"), "w"))
    print("%-20s | skill vs motionless (%%), mean over sequences x seeds: ALL / MAV / handheld                  | tilt deg (const, adaptR_tau) | strapdown ATE | gap anchored err: GP / hold / linear | gap abs: GP / motionless" % "config")
    for c in CFGS:
        rr = [r for r in res if r["cfg"] == c]; line = "%-20s |" % c
        for var in ["const", "adaptR", "adaptR_tau"]:
            sk = lambda sel: 100 * np.mean([1 - r[var] / r["motionless"] for r in rr if sel(r["name"])])
            line += " %s %5.1f /%5.1f /%5.1f |" % (var, sk(lambda n: True), sk(lambda n: not n.startswith("room")), sk(lambda n: n.startswith("room")))
        line += " %5.2f %5.2f | %9.1f |" % (np.mean([r["const_tilt"] for r in rr]), np.mean([r["adaptR_tau_tilt"] for r in rr]), np.median([r["strapdown"] for r in rr]))
        if "gap_gp" in rr[0]: line += " %.3f / %.3f / %.3f | %.3f / %.3f" % tuple(np.nanmean([r["gap_" + k] for r in rr]) for k in ["gp", "hold", "lin", "abs_gp", "abs_motionless"])
        print(line)
