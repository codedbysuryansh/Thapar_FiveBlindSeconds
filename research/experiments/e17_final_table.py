#!/usr/bin/env python3
"""E17 - FINAL Phase 2 table on the mission simulator (OUR simulator, OUR composite - not the organiser's).
Candidates: motionless, naive strapdown, heave high-pass 0.05 Hz, Phase 1B core (spring, no drag), e13 choice
(estimate_final) and the final v4 pipeline (estimate_v4), plus v4 with the blind rows held / damped.
Levels: clean (1 seed), gaps / L1 / L2 (2 seeds), gyro-bias ramp 0.05 rad/s (1 seed). k_drag leave-one-flight-out.
Run: .venv/bin/python research/experiments/e17_final_table.py"""
import sys, os, json, numpy as np
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbs3 as f
from e10_mission import LEVELS, KD
LV = dict(LEVELS); LV.update(gb05=dict(gb=0.05)); SEEDS = {"clean": 1, "gaps": 2, "L1": 2, "L2": 2, "gb05": 1}
IX = dict(wp=0, b_abs=1, b_anc=2, gap=3, ate=4, rpe=5, b_end=6, anc_end=7, ate_yaw=8, ate_none=9, wp_max=10, hit=11, cal=12, cal_b=13)
def work(args):
    n, lvl, kf, seed = args; s = f.load(n); fs = 1 / np.median(np.diff(s.t)); nb = int(round(5 * fs)); oi = np.where(s.ok)[0]; t0, t1 = s.t[oi[0]], s.t[oi[-1]]
    iT = int(np.searchsorted(s.t, [t1 - 5.05, t0 + 0.6 * (t1 - t0), t0 + 0.8 * (t1 - t0)][kf]))
    while not s.ok[iT:iT + nb + 1].all() and iT > 2000: iT -= 40
    sl = slice(0, iT + 1); w, a, vd, meta = f.degrade(s.t[sl], s.w[sl], s.a[sl], LV[lvl], seed=1000 * seed + 7 * kf + sum(n.encode()) % 997)
    M = iT + 1 + nb; gp = s.gt_p[:M]; gq = s.gt_q[:M]; ok = s.ok[:M]; t = s.t[:M]; okp = ok & (np.arange(M) <= iT); b = np.arange(iT + 1, M); tb = t[b] - t[iT]
    cp = np.searchsorted(s.t, np.linspace(t0 + 10, s.t[iT], 6)).clip(oi[0], iT); cp = np.array([c if ok[c] else oi[np.argmin(np.abs(oi - c))] for c in cp]); out = {}
    def score(p, sig=None):
        Rm, tt = f.umeyama(p[okp], gp[okp]); er = np.linalg.norm(p @ Rm.T + tt - gp, axis=1); ga = [er[(t >= g0) & (t < g0 + d) & okp].mean() for g0, d in meta["gaps"] if ((t >= g0) & (t < g0 + d) & okp).sum() > 10]
        anc = np.linalg.norm((p[b] - p[iT]) @ Rm.T - (gp[b] - gp[iT]), axis=1); Ry, ty = f.yaw_align(p[okp], gp[okp])
        cal = [float(np.sqrt(np.mean((er[okp] / sig[okp]) ** 2))), float(np.sqrt(np.mean((er[b] / sig[b]) ** 2)))] if sig is not None else [float("nan")] * 2
        return [float(er[cp].mean()), float(er[b].mean()), float(anc.mean()), float(np.mean(ga)) if ga else float("nan"), float(np.sqrt(np.mean(er[okp] ** 2))), f.rpe_disp(t[okp], p[okp], gp[okp], Rm), float(er[b[-1]]), float(anc[-1]),
                f.rmse(p[okp] @ Ry.T + ty - gp[okp]), f.rmse(p[okp] - gp[okp]), float(er[cp].max()), float((er[cp] < 1.0).mean())] + cal
    kd = float(np.mean([v["fits"]["2.0"]["k"] for m, v in KD.items() if m != n])); tn = s.tn[sl]
    e4 = f.estimate_v4(tn, w, a, n_blind=nb, k_drag=kd); e3 = f.estimate_final(tn, w, a, n_blind=nb, k_drag=kd); e1 = f.estimate3(tn, w, a, n_blind=nb)
    # with removed rows there is no row-for-row correspondence; the simulator keeps NaN rows, so lengths match M
    out["motionless"] = score(np.zeros((M, 3))); out["p1b_spring"] = score(e1["p"], e1["sig"]); out["e13_final"] = score(e3["p"], e3["sig"]); out["v4"] = score(e4["p"], e4["sig"])
    aw = np.nan_to_num(e4["aw"]); aw[iT + 1:] = 0; ps = f.strapdown_pos(e4["t"], aw); ps[b] = ps[iT] + (ps[iT] - ps[iT - 1]) * np.arange(1, nb + 1)[:, None]; out["strapdown"] = score(ps)
    ph = f.highpass_pos(e4["t"][:iT + 1], aw[:iT + 1], 0.05); out["heave05"] = score(np.vstack([ph, np.repeat(ph[-1:], nb, 0)]))
    p = e4["p"].copy(); p[b] = p[iT]; out["v4|hold"] = score(p); p = e4["p"].copy(); p[b] = p[iT] + e4["v"][iT][None] * (0.5 * (1 - np.exp(-tb / 0.5)))[:, None]; out["v4|damp"] = score(p)
    qR = Rot.from_matrix(meta["Rr"]).as_quat(); qo = f.qmul(e4["q"], np.broadcast_to(qR, e4["q"].shape)); n1 = int(round(fs)); pre = np.where(okp)[0]; pre = pre[pre + n1 <= iT]; pre = pre[okp[pre + n1]]; rel = lambda Q, i, j: f.qmul(f.qinv(Q[i]), Q[j])
    qn = float(np.abs(np.linalg.norm(e4["q"], axis=1) - 1).max()); contract = bool(len(e4["p"]) == M and np.all(np.diff(e4["t"]) > 0) and qn < 1e-9 and np.isfinite(e4["p"]).all())
    return dict(name=n, level=lvl, kf=kf, seed=seed, rotor=e4["rotor"], D=e4["drift_D"], drift=e4["drift_used"], zupt=e4["zupt_rows"], tilt=float(np.mean(f.tilt_deg(qo[okp], gq[okp]))), rpe_rot=float(np.mean(f.ang_deg(rel(qo, pre, pre + n1), rel(gq, pre, pre + n1)))) if len(pre) else float("nan"),
                hold_ori=[float(x) for x in (lambda e: (e.mean(), e[-1]))(f.ang_deg(np.broadcast_to([0, 0, 0, 1.0], (len(b), 4)), rel(gq, np.full(len(b), iT), b)))], contract=contract, m=out)
def comp(r, v, w, mode):
    ref = r["m"]["motionless"]; parts = []
    for key, wg in zip(("wp", "b_abs", "gap"), w):
        e = r["m"][v][IX[key]]; e0 = ref[IX[key]]
        if np.isnan(e): continue
        parts.append(((np.clip(1 - e / e0, 0, 1) if mode == "lin" else np.exp(-e / 1.0)), wg))
    return sum(a * b_ for a, b_ in parts) / sum(b_ for _, b_ in parts) * sum(w)
if __name__ == "__main__":
    names = f.available(); jobs = [(n, l, kf, sd) for l in LV for n in names for kf in range(3) for sd in range(SEEDS[l])]
    with Pool(10) as p: res = p.map(work, jobs, chunksize=1)
    json.dump(res, open(os.path.join(f.fbs.ROOT, "research/results/e17_final_table.json"), "w")); V = ["motionless", "strapdown", "heave05", "p1b_spring", "e13_final", "v4", "v4|hold", "v4|damp"]
    mean = lambda rr, v, k: float(np.nanmean([r["m"][v][IX[k]] for r in rr]))
    for grp, sel in (("ALL", lambda r: True), ("EuRoC", lambda r: not r["name"].startswith("room")), ("TUM", lambda r: r["name"].startswith("room"))):
        for lvl in LV:
            rr = [r for r in res if r["level"] == lvl and sel(r)]
            print("\n[%s | %s | %d missions] rotor %.2f drift-corrected %.2f tilt %.2f deg rot-RPE %.2f deg hold-orientation %.1f/%.1f deg contract-ok %.2f" % (grp, lvl, len(rr), np.mean([r["rotor"] for r in rr]), np.mean([r["drift"] for r in rr]), np.mean([r["tilt"] for r in rr]), np.nanmean([r["rpe_rot"] for r in rr]), np.mean([r["hold_ori"][0] for r in rr]), np.mean([r["hold_ori"][1] for r in rr]), np.mean([r["contract"] for r in rr])))
            print("  %-11s    wp wpmax  hit | b_abs b_end  anc ancEnd |  gap |  ATE  yaw   none | RPE1s | cal  calB | lin A(60) B(70) | exp A  B" % "")
            for v in V:
                print("  %-11s %5.2f %5.2f %4.2f | %5.2f %5.2f %5.2f %5.2f | %5.2f | %5.2f %5.2f %6.2f | %5.3f | %4.2f %4.2f | %5.1f %5.1f | %5.1f %5.1f" % (v, mean(rr, v, "wp"), mean(rr, v, "wp_max"), mean(rr, v, "hit"), mean(rr, v, "b_abs"), mean(rr, v, "b_end"), mean(rr, v, "b_anc"), mean(rr, v, "anc_end"), mean(rr, v, "gap"),
                      mean(rr, v, "ate"), mean(rr, v, "ate_yaw"), mean(rr, v, "ate_none"), mean(rr, v, "rpe"), mean(rr, v, "cal"), mean(rr, v, "cal_b"), np.mean([comp(r, v, (30, 20, 10), "lin") for r in rr]), np.mean([comp(r, v, (30, 20, 20), "lin") for r in rr]), np.mean([comp(r, v, (30, 20, 10), "exp") for r in rr]), np.mean([comp(r, v, (30, 20, 20), "exp") for r in rr])))
