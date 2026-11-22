#!/usr/bin/env python3
"""E16 - three refinements suggested by e14/e15, tested on the mission simulator (ours, not the organiser's):
 (a) front4: time-varying horizontal gyro bias from the tilt-correction drift, also when rest anchors exist;
 (b) lever arm as three online states (valid in any sensor orientation), drag noise sd in {2, 1.4, 1};
 (c) blind-window rule: model prediction, hold, or damped coast p_T + v_T tau (1 - exp(-t/tau)).
Run: .venv/bin/python research/experiments/e16_refine.py"""
import sys, os, json, numpy as np
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbs3 as f
from e10_mission import LEVELS, KD
LV = dict(LEVELS); LV.update(gb02=dict(gb=0.02), gb05=dict(gb=0.05), ab10=dict(ab=1.0))
def work(args):
    n, lvl, kf = args; s = f.load(n); fs = 1 / np.median(np.diff(s.t)); nb = int(round(5 * fs)); oi = np.where(s.ok)[0]; t0, t1 = s.t[oi[0]], s.t[oi[-1]]
    iT = int(np.searchsorted(s.t, [t1 - 5.05, t0 + 0.6 * (t1 - t0), t0 + 0.8 * (t1 - t0)][kf]))
    while not s.ok[iT:iT + nb + 1].all() and iT > 2000: iT -= 40
    sl = slice(0, iT + 1); w, a, vd, meta = f.degrade(s.t[sl], s.w[sl], s.a[sl], LV[lvl], seed=7 * kf + sum(n.encode()) % 997)
    M = iT + 1 + nb; gp = s.gt_p[:M]; gq = s.gt_q[:M]; ok = s.ok[:M]; t = s.t[:M]; okp = ok & (np.arange(M) <= iT); b = np.arange(iT + 1, M); tb = t[b] - t[iT]
    cp = np.searchsorted(s.t, np.linspace(t0 + 10, s.t[iT], 6)).clip(oi[0], iT); cp = np.array([c if ok[c] else oi[np.argmin(np.abs(oi - c))] for c in cp]); out = {}; qR = Rot.from_matrix(meta["Rr"]).as_quat()
    def score(p, vT):
        Rm, tt = f.umeyama(p[okp], gp[okp]); al = lambda x: x @ Rm.T + tt; er = np.linalg.norm(al(p) - gp, axis=1); ga = [er[(t >= g0) & (t < g0 + d) & okp].mean() for g0, d in meta["gaps"] if ((t >= g0) & (t < g0 + d) & okp).sum() > 10]
        dg = gp[b] - gp[iT]; anc = lambda dp: float(np.linalg.norm(dp @ Rm.T - dg, axis=1).mean()); ab = lambda dp: float(np.linalg.norm(al(p[iT] + dp) - gp[b], axis=1).mean())
        dm = p[b] - p[iT]; dh = np.zeros_like(dm); d5 = vT[None] * (0.5 * (1 - np.exp(-tb / 0.5)))[:, None]; d1 = vT[None] * (1.0 * (1 - np.exp(-tb / 1.0)))[:, None]
        return [float(er[cp].mean()), ab(dm), anc(dm), anc(dh), anc(d5), anc(d1), float(np.mean(ga)) if ga else float("nan"), float(np.sqrt(np.mean(er[okp] ** 2))), f.rpe_disp(t[okp], p[okp], gp[okp], Rm), ab(dh), ab(d5)]
    out["motionless"] = score(np.zeros((M, 3)), np.zeros(3))
    tu, wu, au, vu, rows = f.ingest(s.tn[sl], w, a); N0 = len(tu); zm = f.rest_info(tu, wu, au, vu)[1]; wf = f.fill_gaps(tu, wu, vu); dtn = np.median(np.diff(tu)); io = np.concatenate([rows, np.arange(N0, N0 + nb)])
    ext = lambda x: np.concatenate([x, np.repeat(x[-1:], nb, 0)]); te = np.concatenate([tu, tu[-1] + dtn * np.arange(1, nb + 1)]); ve = np.concatenate([vu, np.zeros(nb, bool)]); ze = np.concatenate([zm, np.zeros(nb, bool)])
    kd = float(np.mean([v["fits"]["2.0"]["k"] for m, v in KD.items() if m != n])); MAT = f.model_matern52(*f.MATERN); SPR = f.model_spring(*f.SPRING); ROT = f.model_spring(*f.ROTOR); info = {}
    for tag, front in (("f3", f.front3), ("f4", f.front4)):
        q, bg, quiet, af = front(tu, wu, au, vu); lock = f.thrust_lock_local(tu, af, vu); rotor = lock < 0.04; bgt = bg if np.ndim(bg) == 2 else np.broadcast_to(bg, wf.shape)
        qe, afe = ext(q), ext(af); wl = ext(f.lowpass(wf - bgt, fs, 15.0)); qo = f.qmul(q[rows], np.broadcast_to(qR, (len(rows), 4)))
        n1 = int(round(fs)); pre = np.where(okp)[0]; pre = pre[pre + n1 <= iT]; pre = pre[okp[pre + n1]]; rel = lambda Q, i, j: f.qmul(f.qinv(Q[i]), Q[j])
        info[tag] = dict(rotor=bool(rotor), tilt=float(np.mean(f.tilt_deg(qo[okp[:iT + 1]], gq[:iT + 1][okp[:iT + 1]]))), rpe_rot=float(np.mean(f.ang_deg(rel(qo, pre, pre + n1), rel(gq, pre, pre + n1)))) if len(pre) else float("nan"))
        def put(key, p, v): out[tag + "|" + key] = score(p[io], v[io][iT])
        for key, mdl in (("matern", MAT), ("spring", SPR)): p, xs, Ps = f.lin_smooth_z(te, qe, afe, ve, mdl, ze); put(key, p, xs[:, :, 1])
        if rotor:
            p, xs, Ps = f.run_joint6(te, qe, afe, ROT, ve, k_drag=kd, sd=2.0, zmask=ze); put("drag_sd2", p, xs[:, [1, 5, 9]])
            for sd in (2.0, 1.4, 1.0):
                p, xs, Ps = f.run_joint6(te, qe, afe, ROT, ve, k_drag=kd, sd=sd, zmask=ze, wl=wl); put("lev_sd%g" % sd, p, xs[:, [1, 5, 9]]); info[tag]["r_sd%g" % sd] = (xs[0, 14:17] @ meta["Rr"]).tolist()
    return dict(name=n, level=lvl, kf=kf, info=info, m=out)
if __name__ == "__main__":
    names = f.available(); jobs = [(n, l, kf) for l in LV for n in names for kf in range(3)]
    with Pool(10) as p: res = p.map(work, jobs, chunksize=1)
    json.dump(res, open(os.path.join(f.fbs.ROOT, "research/results/e16_refine.json"), "w"))
    K = ["wp", "b_abs", "anc_model", "anc_hold", "anc_d0.5", "anc_d1", "gap", "ate", "rpe", "abs_hold", "abs_d0.5"]
    def g(r, v, i): m = r["m"]; return m[v][i] if v in m else m[v.split("|")[0] + "|matern"][i]
    for grp, sel in (("EuRoC", lambda r: not r["name"].startswith("room")), ("TUM", lambda r: r["name"].startswith("room"))):
        for lvl in LV:
            rr = [r for r in res if r["level"] == lvl and sel(r)]; print("\n[%s | %s | %d]  tilt f3 %.2f f4 %.2f deg | rot-RPE f3 %.2f f4 %.2f deg | motionless wp %.2f b_abs %.2f hold-anc %.2f" % (grp, lvl, len(rr), np.mean([r["info"]["f3"]["tilt"] for r in rr]), np.mean([r["info"]["f4"]["tilt"] for r in rr]),
                  np.nanmean([r["info"]["f3"]["rpe_rot"] for r in rr]), np.nanmean([r["info"]["f4"]["rpe_rot"] for r in rr]), np.mean([r["m"]["motionless"][0] for r in rr]), np.mean([r["m"]["motionless"][1] for r in rr]), np.mean([r["m"]["motionless"][3] for r in rr])))
            for v in (["matern", "spring"] if grp == "TUM" else ["matern", "spring", "drag_sd2", "lev_sd2", "lev_sd1.4", "lev_sd1"]):
                print("   %-10s " % v + " || ".join("%s wp %.2f ATE %.2f RPE %.3f gap %.2f | blind abs model %.2f hold %.2f d.5 %.2f | anc model %.2f hold %.2f d.5 %.2f d1 %.2f" % (tag, *[np.nanmean([g(r, tag + "|" + v, i) for r in rr]) for i in (0, 7, 8, 6, 1, 9, 10, 2, 3, 4, 5)]) for tag in ("f3", "f4")))
    R = np.array([r["info"]["f3"]["r_sd1"] for r in res if "r_sd1" in r["info"]["f3"] and r["level"] == "clean"]); R2 = np.array([r["info"]["f3"]["r_sd1"] for r in res if "r_sd1" in r["info"]["f3"] and r["level"] == "L2"])
    print("\nlever arm estimated online (rotated back to the original sensor frame): clean mean %s sd %s | L2 mean %s sd %s | e15 oracle fit (0.020, 0.016, 0.099)" % (R.mean(0).round(3), R.std(0).round(3), R2.mean(0).round(3), R2.std(0).round(3)))
