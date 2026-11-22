#!/usr/bin/env python3
"""E10 - mission simulator and local composite score (NOT the official score; every unknown rule is a labelled scenario).
A mission = one real sequence cut at a failure time T (IMU 'dies'), degraded by OUR simulator, with 6 checkpoints evenly
spaced up to T (the last one at T) and 5 s of blind rows appended by the estimator. Checkpoints are evaluator-only here
(Scenario B). Ground truth only scores. Run: .venv/bin/python research/experiments/e10_mission.py [seeds]"""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import fbs3 as f
LEVELS = {"clean": {}, "gaps": dict(gaps=[2.0] * 3), "L1": dict(gn=0.01, an=0.1, gb=0.005, ab=0.05, gaps=[2.0] * 3, rot=True),
          "L2": dict(gn=0.05, an=0.5, gb=0.02, ab=0.2, gaps=[2.0] * 3, rot=True)}
CANDS = ["motionless", "strapdown", "heave05", "gp_matern", "gp_spring", "spring_drag_gated", "m44_drag_gated", "spring_drag_universal"]
KD = json.load(open(os.path.join(f.fbs.ROOT, "research/results/e05_drag.json")))
def k_loso(n):
    ks = [v["fits"]["2.0"]["k"] for m, v in KD.items() if m != n]; return float(np.mean(ks))
def mission(args):
    n, lvl, kf, seed = args; s = f.load(n); fs = 1 / np.median(np.diff(s.t)); nb = int(round(5 * fs)); oi = np.where(s.ok)[0]; t0, t1 = s.t[oi[0]], s.t[oi[-1]]
    iT = int(np.searchsorted(s.t, [t1 - 5.05, t0 + 0.6 * (t1 - t0), t0 + 0.8 * (t1 - t0)][kf]))
    while not s.ok[iT:iT + nb + 1].all() and iT > 2000: iT -= 40
    sl = slice(0, iT + 1); w, a, valid, meta = f.degrade(s.t[sl], s.w[sl], s.a[sl], LEVELS[lvl], seed=1000 * seed + 7 * kf + sum(n.encode()) % 997)
    M = iT + 1 + nb; gp = s.gt_p[:M]; ok = s.ok[:M]; t = s.t[:M]; pre = np.arange(M) <= iT; okp = ok & pre; b = np.arange(iT + 1, M)
    cp = np.searchsorted(s.t, np.linspace(t0 + 10, s.t[iT], 6)).clip(oi[0], iT); cp = np.array([c if ok[c] else oi[np.argmin(np.abs(oi - c))] for c in cp])
    P = {}; kd = k_loso(n)
    e = f.estimate3(s.tn[sl], w, a, n_blind=nb); P["gp_spring"] = e["p"]; tu, _, q, vu, af = e["internal"]; N0 = iT + 1
    aw = f.qrot(q[:N0], af[:N0]); aw = aw - aw[vu[:N0]].mean(0); dt = np.diff(tu[:N0], prepend=tu[0])[:, None]
    vS = np.cumsum(aw * dt, 0); pS = np.cumsum(vS * dt, 0); tau = (tu[N0:] - tu[N0 - 1])[:, None]
    P["strapdown"] = np.vstack([pS, pS[-1] + vS[-1] * tau]); pH = f.highpass_pos(tu[:N0], aw, 0.05); P["heave05"] = np.vstack([pH, np.repeat(pH[-1:], nb, 0)]); P["motionless"] = np.zeros((M, 3))
    P["gp_matern"] = f.estimate3(s.tn[sl], w, a, model=f.model_matern52(*f.MATERN), n_blind=nb)["p"]
    P["spring_drag_gated"] = f.estimate3(s.tn[sl], w, a, n_blind=nb, drag=(kd, 2.0))["p"] if e["lock"] < 0.04 else P["gp_spring"]
    P["m44_drag_gated"] = f.estimate3(s.tn[sl], w, a, model=f.model_matern52(4, 4, 20, 0.1), n_blind=nb, drag=(kd, 2.0))["p"] if e["lock"] < 0.04 else P["gp_matern"]
    P["spring_drag_universal"] = f.estimate3(s.tn[sl], w, a, n_blind=nb, drag=(kd, 2.0), gate=9.9)["p"]
    out = {"name": n, "lvl": lvl, "kf": kf, "seed": seed, "lock": e["lock"], "quiet": e["quiet"], "T": float(s.t[iT]),
           "ori_blind_hold_mean": float(np.mean(f.ang_deg(np.repeat(s.gt_q[iT:iT + 1], nb, 0), s.gt_q[b]))), "tilt": float(np.mean(f.tilt_deg(q[:N0][okp[:N0] & vu[:N0]], f.qmul(s.gt_q[:N0], f.qinv(f.Rot.from_matrix(meta["Rr"]).as_quat())[None])[okp[:N0] & vu[:N0]])))}
    for c, p in P.items():
        Rm, tt = f.umeyama(p[okp], gp[okp]); er = np.linalg.norm(p @ Rm.T + tt - gp, axis=1); R2, t2 = f.umeyama(p[cp], gp[cp]); r = {}
        r["ate"] = float(np.sqrt(np.mean(er[okp] ** 2))); r["rpe1"] = f.rpe_disp(t[okp], p[okp], gp[okp], Rm); r["wp"] = float(er[cp].mean()); r["wp_max"] = float(er[cp].max()); r["wp_hit1"] = float((er[cp] < 1.0).mean())
        r["wp_final"] = float(er[cp[-1]]); r["wp_cpalign"] = float(np.linalg.norm(p[cp] @ R2.T + t2 - gp[cp], axis=1).mean())
        r["b_abs"] = float(er[b].mean()); r["b_end"] = float(er[b[-1]]); da = np.linalg.norm((p[b] - p[iT]) @ Rm.T - (gp[b] - gp[iT]), axis=1); r["b_anc"] = float(da.mean()); r["b_anc_end"] = float(da[-1])
        ga, gn_, gr = [], [], []
        for (g0, dur) in meta["gaps"]:
            m = (t >= g0) & (t < g0 + dur) & okp
            if m.sum() < 10: continue
            i0 = np.where(m)[0][0] - 1; ga.append(er[m].mean()); gn_.append(np.linalg.norm((p[m] - p[i0]) @ Rm.T - (gp[m] - gp[i0]), axis=1).mean())
            pm = (t >= g0 + dur) & (t < g0 + dur + 2.0) & okp
            if pm.sum() > 10: gr.append(er[pm].mean())
        r["g_abs"] = float(np.mean(ga)) if ga else float("nan"); r["g_anc"] = float(np.mean(gn_)) if gn_ else float("nan"); r["g_rec"] = float(np.mean(gr)) if gr else float("nan"); out[c] = r
    return out
if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 2; names = f.available()
    jobs = [(n, l, kf, sd) for l in LEVELS for n in names for kf in range(3) for sd in range(seeds if l != "clean" else 1)]
    with Pool(10) as p: res = p.map(mission, jobs, chunksize=2)
    json.dump(res, open(os.path.join(f.fbs.ROOT, "research/results/e10_mission.json"), "w"))
    def agg(rows, c, k): return float(np.nanmean([r[c][k] for r in rows]))
    for lvl in LEVELS:
        rr = [r for r in res if r["lvl"] == lvl]; print("\n== %s: %d missions | tilt %.2f deg | blind orientation (hold) %.1f deg | drag gate on: EuRoC %d/%d, TUM %d/%d" % (lvl, len(rr), np.mean([r["tilt"] for r in rr]), np.mean([r["ori_blind_hold_mean"] for r in rr]),
              sum(r["lock"] < 0.04 for r in rr if not r["name"].startswith("room")), sum(not r["name"].startswith("room") for r in rr), sum(r["lock"] < 0.04 for r in rr if r["name"].startswith("room")), sum(r["name"].startswith("room") for r in rr)))
        print("%-22s  ATE   RPE1s | waypoint: mean  max  final hit<1m cp-aligned | blind: abs  end  anchored anc-end | gaps: in-gap anchored after | accuracy score lin(A/B) exp(A/B) | EuRoC ATE wp b_abs | TUM ATE wp b_abs" % "candidate")
        for c in CANDS:
            comp = []
            for mode in ("lin", "exp"):
                for wts in ((30, 20, 10), (30, 20, 20)):
                    sc = []
                    for r in rr:
                        parts = []
                        for k, wgt in zip(("wp", "b_abs", "g_abs"), wts):
                            if np.isnan(r[c][k]): continue
                            v = np.clip(1 - r[c][k] / r["motionless"][k], 0, 1) if mode == "lin" else np.exp(-r[c][k] / 1.0); parts.append((v, wgt))
                        sc.append(sum(v * w_ for v, w_ in parts) / sum(w_ for _, w_ in parts))
                    comp.append(100 * np.mean(sc))
            eu = [r for r in rr if not r["name"].startswith("room")]; tm = [r for r in rr if r["name"].startswith("room")]
            print("%-22s %6.2f %5.2f | %5.2f %5.2f %5.2f  %4.2f   %5.2f    | %5.2f %5.2f  %5.2f  %5.2f   | %5.2f  %5.2f  %5.2f | %5.1f %5.1f %5.1f %5.1f | %5.2f %5.2f %5.2f | %5.2f %5.2f %5.2f" % (c, agg(rr, c, "ate"), agg(rr, c, "rpe1"), agg(rr, c, "wp"), agg(rr, c, "wp_max"), agg(rr, c, "wp_final"), agg(rr, c, "wp_hit1"), agg(rr, c, "wp_cpalign"),
                  agg(rr, c, "b_abs"), agg(rr, c, "b_end"), agg(rr, c, "b_anc"), agg(rr, c, "b_anc_end"), agg(rr, c, "g_abs"), agg(rr, c, "g_anc"), agg(rr, c, "g_rec"), *comp, agg(eu, c, "ate"), agg(eu, c, "wp"), agg(eu, c, "b_abs"), agg(tm, c, "ate"), agg(tm, c, "wp"), agg(tm, c, "b_abs")))
