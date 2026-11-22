#!/usr/bin/env python3
"""E43 - the five blind seconds on OFFICIAL missions (blind rows = the trailing zero rows of the input; FAQ: no rows appended).
The pre-failure estimate is computed once (v5 front end, fixed platform gate); every predictor then fills the blind rows from the
state at the last valid sample. Scoring = evo-style SE(3) alignment over ALL rows (so the predictor also moves the alignment a
little), plus anchored displacement errors. Nothing from the blind window or from ground truth enters a predictor except the rows
labelled ORACLE (diagnostics only)."""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p4, fbs6 as f
f.fbs5.GS_K2 = 0.25
CFGS = [("full", dict(seed=s)) for s in (42, 1, 2)] + [("crop60", dict(seed=s, crop=0.6)) for s in (42, 7)] + [("crop80", dict(seed=42, crop=0.8))]
DS = dict(z=5.0, rule="triple")
def damp(tau, td): return tau if td is None else td * (1 - np.exp(-tau / td))
def ar_forecast(v10, p, steps, ridge=1e-2):
    out = np.zeros((steps, 3))
    for j in range(3):
        x = v10[:, j]; n = len(x)
        if n < 4 * p: out[:, j] = x[-1] if n else 0.0; continue
        X = np.stack([x[p - k - 1:n - k - 1] for k in range(p)], 1); y = x[p:]; c = np.linalg.solve(X.T @ X + ridge * len(y) * (np.var(x) + 1e-12) * np.eye(p), X.T @ y); h = list(x[-p:][::-1])
        for s in range(steps): nx = float(np.dot(c, h[:p])); out[s, j] = nx; h.insert(0, nx)
    return out
def pipeline(m):
    t, wu, au, valid, rows = f.ingest2(m.tn, m.w, m.a); L, nt = f.split_tail(t, wu, au, valid); t0, wu0, au0, v0 = t[:L + 1], wu[:L + 1], au[:L + 1], valid[:L + 1]
    wu0, au0, v0, _ = f.repair(t0, wu0, au0, v0, g_lo=5.0, g_hi=15.5); cm = f.common_of(t0, wu0, au0, v0)
    q3, base, h, anc = f.fe_const(t0, wu0, au0, v0, True, True)
    if cm["lock"] < 0.04: q3, base, _ = f.fe_spline(t0, wu0, au0, v0, True, True, 3.0, base, h)
    P = f.prep_from(t0, wu0, au0, v0, q3, base, True, cm)
    e = f.fbs4.estimate_v6(None, None, None, n_blind=nt, k_drag=m.kd, P=P, full=True, gap_ref=2.0); return e, rows, L, nt, P
def one(args):
    n, order, cname, ckw = args; m = p4.official(n, order, **ckw); e, rows, L, nt, P = pipeline(m); xs = e["xs"]; N0 = e["N0"]; pos, vel, acc = xs[:, [0, 4, 8]], xs[:, [1, 5, 9]], xs[:, [2, 6, 10]]
    fs = 1.0 / P["dtn"]; tau = (np.arange(1, nt + 1)) / fs; pT, vT = pos[N0 - 1], vel[N0 - 1]; D = {"hold": np.zeros((nt, 3)), "model": pos[N0:] - pT}
    for td in (0.25, 0.5, 1.0, 2.0, None): D["cv_%s" % ("inf" if td is None else "%g" % td)] = damp(tau, td)[:, None] * vT
    for lam in (0.25, 0.5): D["model_x%g" % lam] = lam * D["model"]
    v10 = vel[:N0][::int(round(fs / 10))][-600:]; t10 = np.arange(1, 51) * 0.1
    for p_ in (5, 10): vf = ar_forecast(v10, p_, 50); dp = np.cumsum(vf, 0) * 0.1; D["ar%d" % p_] = np.stack([np.interp(tau, t10, dp[:, j]) for j in range(3)], 1)
    cen = pos[:N0].mean(0)
    for tc in (5.0, 20.0): D["centroid_%g" % tc] = (1 - np.exp(-tau / tc))[:, None] * (cen - pT)
    D["cv1+cen5"] = D["cv_1"] + D["centroid_5"]
    for a_, b_ in (("model", "ar5"), ("model", "ar10"), ("model", "cv_1"), ("model", "hold"), ("ar5", "hold")): D["%s+%s" % (a_, b_)] = 0.5 * (D[a_] + D[b_])
    for td in (0.5, 1.0, 2.0): D["ar5_damp%g" % td] = np.cumsum(np.diff(np.vstack([np.zeros(3), D["ar5"]]), axis=0) * np.exp(-tau / td)[:, None], 0)
    tl = np.concatenate([P["t"], P["t"][-1] + np.arange(1, nt + 1) / fs]); slot = rows; blind_rows = np.where(slot > L)[0]; okb = m.ok[blind_rows]
    gi = np.where(m.ok & (slot <= L))[0][-1]; Rm, tt = f.umeyama(pos[slot[m.ok & (slot <= L)]], m.gp[m.ok & (slot <= L)])
    vgt = m.s.gt_v[m.idx[gi]] @ Rm
    for td in (1.0, None): D["ORACLE_v_%s" % ("inf" if td is None else "%g" % td)] = damp(tau, td)[:, None] * vgt
    res = {}
    for k, d in D.items():
        full = np.concatenate([pos[:N0], pT + d])[slot]; res[k] = p4.score(m, full, e["q"][slot])
    # orientation predictors (position = model)
    q = P["q"]; wc = P["wc"]; vd = P["valid"]; qT = q[-1]; ww = f.qrot(q, wc); O = {"hold": np.repeat(qT[None], nt, 0)}
    for win in (1.0, 5.0):
        k = int(win * fs); sel = vd[-k:]; wb = wc[-k:][sel].mean(0) if sel.any() else np.zeros(3); yr = ww[-k:][sel][:, 2].mean() if sel.any() else 0.0
        for td in (0.5, 2.0, None):
            g = damp(tau, td); nm = "%g_%s" % (win, "inf" if td is None else "%g" % td); O["gyro_w" + nm] = f.qmul(np.repeat(qT[None], nt, 0), f.qexp(g[:, None] * wb)); O["yaw_w" + nm] = f.qmul(np.stack([np.zeros(nt), np.zeros(nt), np.sin(yr * g / 2), np.cos(yr * g / 2)], 1), np.repeat(qT[None], nt, 0))
    pm = np.concatenate([pos[:N0], pT + D["model"]])[slot]
    for k, qq in O.items():
        qf = np.concatenate([q, qq])[slot]; res["ori:" + k] = p4.score(m, pm, qf)
    return dict(name=n, order=order, cfg=cname, seed=ckw["seed"], m=res, rotor=bool(e["rotor"]), disp=float(np.linalg.norm(m.gp[blind_rows[okb]] - m.gp[gi], axis=1).mean()) if okb.any() else float("nan"), speed=float(np.linalg.norm(m.s.gt_v[m.idx[gi]])), vel_err=float(np.linalg.norm(vT - vgt)), n_tail=int(nt))
if __name__ == "__main__":
    names = f.available()
    with Pool(10) as p: res = p.map(one, [(n, o, c, kw) for o in ("A", "B") for c, kw in CFGS for n in names], chunksize=1)
    json.dump(res, open(os.path.join(p4.ROOT, "research/results/e43b_blind_official.json"), "w")); keys = [k for k in res[0]["m"] if not k.startswith("ori:")]; okeys = [k for k in res[0]["m"] if k.startswith("ori:")]
    for grp, sel in (("EuRoC", lambda r: not r["name"].startswith("room")), ("TUM", lambda r: r["name"].startswith("room"))):
        for cf in ("full", "crop60", "crop80"):
            rr = [r for r in res if sel(r) and r["cfg"] == cf]
            print("\n=== %s %s (%d missions; true 5 s displacement %.2f m; speed at failure %.2f m/s; velocity error at failure %.2f m/s)" % (grp, cf, len(rr), np.nanmean([r["disp"] for r in rr]), np.mean([r["speed"] for r in rr]), np.mean([r["vel_err"] for r in rr])))
            for k in keys: print("  %-14s " % k + "  ".join("%s %.3f" % (x, np.nanmean([r["m"][k][x] for r in rr])) for x in ("b_anc", "b_end", "b_abs", "ate", "wp")))
            for k in okeys: print("  %-22s ori_b %.2f  ori %.2f  rpe_r %.3f" % (k, np.nanmean([r["m"][k]["ori_b"] for r in rr]), np.nanmean([r["m"][k]["ori"] for r in rr]), np.nanmean([r["m"][k]["rpe_r"] for r in rr])))
