#!/usr/bin/env python3
"""E14 - final-architecture decision on the mission simulator (OUR simulator and OUR composite, not the organiser's).
One front end per mission; position-model variants on top:
  matern / spring          no drag, no zero-velocity rows (Phase 1B cores)
  final                    e13 choice = gate ? spring(20,2,.3,1)+drag(sd 2, white) : Matern(1,2)      [= fbs3.estimate_final]
  final_zupt               + zero-velocity rows at accepted rest windows
  rotor-only refinements   lever-arm corrected drag rows (r leave-one-flight-out), red+white drag noise (e15 spectrum),
                           white sd 1, and sensitivities (k x0.8 / x1.2, drag-bias prior 0.5 m/s^2)
Levels: clean, gaps, L1, L2 and three single-factor stress levels (accel bias 0.3 / 1.0 m/s^2 ramp, gyro bias 0.05 rad/s ramp).
Metrics per variant: waypoint mean/max/hit, blind abs/anchored mean+end, in-gap, ATE (SE(3), yaw-only, none), RPE 1 s,
uncertainty calibration; per mission: tilt, rotational RPE, blind orientation (hold vs gyro extrapolation).
Run: .venv/bin/python research/experiments/e14_final.py"""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbs3 as f
from e10_mission import LEVELS, KD
LV = dict(LEVELS); LV.update(ab03=dict(ab=0.3), ab10=dict(ab=1.0), gb05=dict(gb=0.05))
E15 = json.load(open(os.path.join(f.fbs.ROOT, "research/results/e15_drag_model.json")))
def loso(n):
    ks = [v["fits"]["2.0"]["k"] for m, v in KD.items() if m != n]; rs = [v["B"][3:6] for m, v in E15.items() if m != n]; return float(np.mean(ks)), np.mean(rs, 0)
RED = (2.0, 0.04)
def work(args):
    n, lvl, kf = args; s = f.load(n); fs = 1 / np.median(np.diff(s.t)); nb = int(round(5 * fs)); oi = np.where(s.ok)[0]; t0, t1 = s.t[oi[0]], s.t[oi[-1]]
    iT = int(np.searchsorted(s.t, [t1 - 5.05, t0 + 0.6 * (t1 - t0), t0 + 0.8 * (t1 - t0)][kf]))
    while not s.ok[iT:iT + nb + 1].all() and iT > 2000: iT -= 40
    sl = slice(0, iT + 1); w, a, vd, meta = f.degrade(s.t[sl], s.w[sl], s.a[sl], LV[lvl], seed=7 * kf + sum(n.encode()) % 997)
    M = iT + 1 + nb; gp = s.gt_p[:M]; gq = s.gt_q[:M]; ok = s.ok[:M]; t = s.t[:M]; okp = ok & (np.arange(M) <= iT); b = np.arange(iT + 1, M)
    cp = np.searchsorted(s.t, np.linspace(t0 + 10, s.t[iT], 6)).clip(oi[0], iT); cp = np.array([c if ok[c] else oi[np.argmin(np.abs(oi - c))] for c in cp]); out = {}
    def score(p, sig=None):
        Rm, tt = f.umeyama(p[okp], gp[okp]); er = np.linalg.norm(p @ Rm.T + tt - gp, axis=1); ga = [er[(t >= g0) & (t < g0 + d) & okp].mean() for g0, d in meta["gaps"] if ((t >= g0) & (t < g0 + d) & okp).sum() > 10]
        anc = np.linalg.norm((p[b] - p[iT]) @ Rm.T - (gp[b] - gp[iT]), axis=1); Ry, ty = f.yaw_align(p[okp], gp[okp])
        cal = [float(np.sqrt(np.mean((er[okp] / sig[okp]) ** 2))), float(np.sqrt(np.mean((er[b] / sig[b]) ** 2)))] if sig is not None else [float("nan")] * 2
        return [float(er[cp].mean()), float(er[b].mean()), float(anc.mean()), float(np.mean(ga)) if ga else float("nan"), float(np.sqrt(np.mean(er[okp] ** 2))), f.rpe_disp(t[okp], p[okp], gp[okp], Rm), float(er[b[-1]]), float(anc[-1]),
                f.rmse(p[okp] @ Ry.T + ty - gp[okp]), f.rmse(p[okp] - gp[okp]), float(er[cp].max()), float((er[cp] < 1.0).mean())] + cal
    out["motionless"] = score(np.zeros((M, 3)))
    # ---- front end once ----
    tu, wu, au, vu, rows = f.ingest(s.tn[sl], w, a); N0 = len(tu); q, bg, quiet, af = f.front3(tu, wu, au, vu); lock = f.thrust_lock_local(tu, af, vu); zm = f.rest_info(tu, wu, au, vu)[1]
    wf = f.fill_gaps(tu, wu, vu); bgt = bg if np.ndim(bg) == 2 else np.broadcast_to(bg, wf.shape); dtn = np.median(np.diff(tu))
    ext = lambda x: np.concatenate([x, np.repeat(x[-1:], nb, 0)]); te = np.concatenate([tu, tu[-1] + dtn * np.arange(1, nb + 1)]); ve = np.concatenate([vu, np.zeros(nb, bool)]); ze = np.concatenate([zm, np.zeros(nb, bool)])
    qe, afe, wfe, bge = ext(q), ext(af), ext(wf), ext(bgt); z0 = np.zeros(len(te), bool); rotor = lock < 0.04; kd, rl = loso(n) if n in KD else (f.K_DRAG, np.array(f.LEVER))
    MAT = f.model_matern52(*f.MATERN); SPR = f.model_spring(*f.SPRING); ROT = f.model_spring(*f.ROTOR)
    def ax(model, zmask): p, xs, Ps = f.lin_smooth_z(te, qe, afe, ve, model, zmask); return p, np.sqrt(3 * Ps[:, 0, 0])
    def jt(**kw): p, xs, Ps = f.run_joint5(te, qe, afe, ROT, ve, **kw); return p, np.sqrt(Ps[:, 0, 0] + Ps[:, 4, 4] + Ps[:, 8, 8])
    R = {}; R["matern"] = ax(MAT, z0); R["spring"] = ax(SPR, z0); R["matern_zupt"] = ax(MAT, ze)
    if rotor:
        lev = f.lever_acc(te, wfe, bge, rl)
        R["final"] = jt(k_drag=kd, sd=2.0); R["final_zupt"] = jt(k_drag=kd, sd=2.0, zmask=ze); R["zupt_lever"] = jt(k_drag=kd, sd=2.0, zmask=ze, lev=lev)
        R["zupt_sd1"] = jt(k_drag=kd, sd=1.0, zmask=ze); R["zupt_lever_sd1"] = jt(k_drag=kd, sd=1.0, zmask=ze, lev=lev)
        R["zupt_red"] = jt(k_drag=kd, sd=0.55, zmask=ze, fast=RED); R["zupt_lever_red"] = jt(k_drag=kd, sd=0.55, zmask=ze, lev=lev, fast=RED)
        R["zupt_lever_red8"] = jt(k_drag=kd, sd=0.8, zmask=ze, lev=lev, fast=(2.0, 0.06))
        R["sens_k_lo"] = jt(k_drag=0.8 * kd, sd=2.0, zmask=ze, lev=lev); R["sens_k_hi"] = jt(k_drag=1.2 * kd, sd=2.0, zmask=ze, lev=lev); R["sens_sc05"] = jt(k_drag=kd, sd=2.0, zmask=ze, lev=lev, sc=0.5)
    else:
        R["final"] = R["matern"]; R["final_zupt"] = R["matern_zupt"]
    for k, (p, sig) in R.items():
        io = np.concatenate([rows, np.arange(N0, N0 + nb)]); pp = p[io]; out[k] = score(pp, sig[io]); ph = pp.copy(); ph[b] = ph[iT]; out[k + "|hold"] = score(ph)
    # ---- orientation (independent of the position model) ----
    qo = qe[np.concatenate([rows, np.arange(N0, N0 + nb)])]; n1 = int(round(fs)); pre = np.where(okp)[0]; pre = pre[pre + n1 <= iT]; pre = pre[okp[pre + n1]]
    rel = lambda Q, i, j: f.qmul(f.qinv(Q[i]), Q[j]); rpe_rot = float(np.mean(f.ang_deg(rel(qo, pre, pre + n1), rel(gq, pre, pre + n1)))) if len(pre) else float("nan")
    tilt = float(np.mean(f.tilt_deg(qo[okp], gq[okp]))); tr = rel(gq, np.full(len(b), iT), b); tb = t[b] - t[iT]
    lastv = np.where(vu)[0][-40:]; wbar = (wf[lastv] - bgt[lastv]).mean(0); ori = {"hold": f.ang_deg(np.broadcast_to([0, 0, 0, 1.0], tr.shape), tr)}
    for tau in (0.25, 0.5, 1.0, 1e9): ori["cv_tau%g" % tau] = f.ang_deg(f.qexp(wbar[None] * (tau * (1 - np.exp(-tb / tau)))[:, None] if tau < 1e8 else wbar[None] * tb[:, None]), tr)
    return dict(name=n, level=lvl, kf=kf, rotor=bool(rotor), lock=float(lock), zupt_s=float(zm.sum() / fs), ends_at_rest=bool(zm[-int(fs):].any()), tilt=tilt, rpe_rot=rpe_rot, ori={k: [float(v.mean()), float(v[-1])] for k, v in ori.items()}, m=out)
if __name__ == "__main__":
    names = f.available(); jobs = [(n, l, kf) for l in LV for n in names for kf in range(3)]
    with Pool(10) as p: res = p.map(work, jobs, chunksize=1)
    json.dump(res, open(os.path.join(f.fbs.ROOT, "research/results/e14_final.json"), "w")); print("missions", len(res))
