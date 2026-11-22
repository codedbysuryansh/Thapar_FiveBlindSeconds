#!/usr/bin/env python3
"""E20 - the five blind seconds, as a forecasting problem of its own.
For every mission (17 sequences x 8 failure points x clean / L1 / L2 / L3) the pre-failure estimate is computed ONCE (v4);
every predictor then extrapolates position and orientation from the same state. Nothing from the blind window or from
ground truth enters a predictor, except the two rows labelled ORACLE, which are diagnostics that bound what is achievable.
Position metrics (m): anc = mean error of the predicted DISPLACEMENT over the 5 s path (anchored at the last estimate),
end = the same at 5 s, abs = mean absolute position error in the blind window after the pre-failure SE(3) alignment.
Orientation metrics (deg): mean / end angular error in the blind window after heading alignment on the pre-failure rows.
Run: .venv/bin/python research/experiments/e20_blind.py"""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p3, fbs4 as f
KFS = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]; LEVELS = ("clean", "L1", "L2", "L3")
def damp(tau, td): return tau if td is None else td * (1 - np.exp(-tau / td))
def ar_forecast(v10, p, steps, ridge=1e-2):
    out = np.zeros((steps, 3))
    for j in range(3):
        x = v10[:, j]; n = len(x)
        if n < 4 * p: out[:, j] = x[-1] if n else 0.0; continue
        X = np.stack([x[p - k - 1:n - k - 1] for k in range(p)], 1); y = x[p:]; c = np.linalg.solve(X.T @ X + ridge * len(y) * np.var(x) * np.eye(p), X.T @ y); h = list(x[-p:][::-1])
        for s in range(steps): nx = float(np.dot(c, h[:p])); out[s, j] = nx; h.insert(0, nx)
    return out
def analog(pos5, vel5, acc5, steps, K=8, yawinv=False):
    n = len(pos5); last = n - 1; cand = np.arange(0, n - steps - 10)
    if len(cand) < 3 * K: return np.zeros((steps, 3))
    sv = vel5.std() + 1e-9; sa = acc5.std() + 1e-9; vT, aT = vel5[last], acc5[last]
    if yawinv:
        al = np.arctan2(vT[1], vT[0]) - np.arctan2(vel5[cand, 1], vel5[cand, 0]); c, s = np.cos(al), np.sin(al); rot = lambda x: np.stack([c * x[:, 0] - s * x[:, 1], s * x[:, 0] + c * x[:, 1], x[:, 2]], 1); vc, ac = rot(vel5[cand]), rot(acc5[cand])
    else: vc, ac = vel5[cand], acc5[cand]
    d = np.sqrt(((vc - vT) ** 2).sum(1) / sv ** 2 + ((ac - aT) ** 2).sum(1) / sa ** 2); order = np.argsort(d); pick = []
    for i in order:
        if all(abs(cand[i] - cand[j]) > 10 for j in pick): pick.append(i)
        if len(pick) == K: break
    w = 1.0 / (d[pick] + 0.3); w /= w.sum(); out = np.zeros((steps, 3))
    for wi, i in zip(w, pick):
        dp = pos5[cand[i] + 1:cand[i] + 1 + steps] - pos5[cand[i]]
        if yawinv: c, s = np.cos(al[i]), np.sin(al[i]); dp = np.stack([c * dp[:, 0] - s * dp[:, 1], s * dp[:, 0] + c * dp[:, 1], dp[:, 2]], 1)
        out += wi * dp
    return out
def one(args):
    n, kf, lvl, seed = args; m = p3.mission(n, lvl, kf, seed); e = f.estimate_v6(m.tn, m.w, m.a, n_blind=m.nb, k_drag=m.kd, full=True); N0 = e["N0"]; xs = e["xs"]; pos, vel, acc = xs[:, [0, 4, 8]], xs[:, [1, 5, 9]], xs[:, [2, 6, 10]]
    nb = m.nb; fs = m.fs; tau = (np.arange(nb) + 1) / fs; iT = m.iT; b = m.b; A = m.okp; Rm, tt = f.umeyama(pos[:N0][A[:N0]], m.gp[:N0][A[:N0]]); pT, vT, aT = pos[N0 - 1], vel[N0 - 1], acc[N0 - 1]; Dtrue = m.gp[b] - m.gp[iT]
    D = {"hold": np.zeros((nb, 3)), "model": pos[N0:] - pT}
    for td in (0.25, 0.5, 1.0, 2.0, 4.0, None): D["cv_%s" % ("inf" if td is None else "%g" % td)] = damp(tau, td)[:, None] * vT
    for lam in (0.25, 0.5, 0.75): D["model_x%g" % lam] = lam * D["model"]
    v10 = vel[:N0][::20][-600:]; t10 = (np.arange(50) + 1) * 0.1
    for p_ in (5, 10, 20): vf = ar_forecast(v10, p_, 50); dp = np.cumsum(vf, 0) * 0.1; D["ar%d" % p_] = np.stack([np.interp(tau, t10, dp[:, j]) for j in range(3)], 1)
    sl5 = slice((N0 - 1) % 40, N0, 40); t5 = (np.arange(25) + 1) * 0.2
    for nm, yi in (("analog", False), ("analog_yaw", True)): dp = analog(pos[sl5], vel[sl5], acc[sl5], 25, yawinv=yi); D[nm] = np.stack([np.interp(tau, t5, dp[:, j]) for j in range(3)], 1)
    cen = pos[:N0].mean(0)
    for tc in (5.0, 10.0, 20.0): D["centroid_%g" % tc] = (1 - np.exp(-tau / tc))[:, None] * (cen - pT); D["cv1+cen_%g" % tc] = D["cv_1"] + D["centroid_%g" % tc]
    D["analog+model"] = 0.5 * (D["analog"] + D["model"]); D["analog_yaw+cv1"] = 0.5 * (D["analog_yaw"] + D["cv_1"])
    vgt = m.s.gt_v[iT] @ Rm
    for td in (0.5, 1.0, 2.0, 4.0, None): D["ORACLE_v_%s" % ("inf" if td is None else "%g" % td)] = damp(tau, td)[:, None] * vgt
    pos_m = {}
    for k, d in D.items():
        ea = np.linalg.norm(d @ Rm.T - Dtrue, axis=1); eabs = np.linalg.norm((pT + d) @ Rm.T + tt - m.gp[b], axis=1); pos_m[k] = [float(ea.mean()), float(ea[-1]), float(eabs.mean()), float(eabs[-1])]
    # ---- orientation
    P = e["P"]; q = P["q"][:N0]; wc = P["wc"][:N0]; vd = P["valid"][:N0]; qR = f.Rot.from_matrix(m.meta["Rr"]).as_quat(); qo = f.qmul(q, qR[None]); B = A[:N0]; dq = f.qmul(m.gq[:N0][B], f.qinv(qo[B])); xr = f.qrot(dq, np.array([1.0, 0, 0])); psi = np.arctan2(xr[:, 1], xr[:, 0]); q0 = p3.qyaw(float(np.arctan2(np.sin(psi).mean(), np.cos(psi).mean())))
    qT = q[-1]; O = {"hold": np.repeat(qT[None], nb, 0)}; ww = f.qrot(q, wc)
    for win in (0.2, 1.0, 5.0):
        k = int(win * fs); sel = vd[-k:]; wb = wc[-k:][sel].mean(0) if sel.any() else np.zeros(3); yr = ww[-k:][sel][:, 2].mean() if sel.any() else 0.0
        for td in (0.25, 0.5, 1.0, 2.0, None):
            g = damp(tau, td); O["gyro_w%g_%s" % (win, "inf" if td is None else "%g" % td)] = f.qmul(np.repeat(qT[None], nb, 0), f.qexp(g[:, None] * wb)); O["yaw_w%g_%s" % (win, "inf" if td is None else "%g" % td)] = f.qmul(np.stack([np.zeros(nb), np.zeros(nb), np.sin(yr * g / 2), np.cos(yr * g / 2)], 1), np.repeat(qT[None], nb, 0))
    k = int(20 * fs); ez = np.array([0, 0, 1.0]); ub = f.qrot(f.qinv(q[-k:]), np.broadcast_to(ez, (len(q[-k:]), 3))).mean(0); ub /= np.linalg.norm(ub); zw = f.qrot(qT, ub); qfix = f.shortest_arc(zw[None], ez[None])[0]; rv = f.qlog(qfix[None])[0]
    for tr in (0.5, 1.0, 2.0):
        g = 1 - np.exp(-tau / tr); O["level_%g" % tr] = f.qmul(f.qexp(g[:, None] * rv), np.repeat(qT[None], nb, 0))
        for win in (5.0,):
            kk = int(win * fs); sel = vd[-kk:]; yr = ww[-kk:][sel][:, 2].mean() if sel.any() else 0.0; gy = damp(tau, 2.0); O["level_%g+yaw_w5_2" % tr] = f.qmul(np.stack([np.zeros(nb), np.zeros(nb), np.sin(yr * gy / 2), np.cos(yr * gy / 2)], 1), O["level_%g" % tr])
    ori_m = {}
    for k_, qq in O.items():
        qq = qq / np.linalg.norm(qq, axis=1, keepdims=True); qa = f.qmul(q0[None], f.qmul(qq, qR[None])); er = f.ang_deg(qa, m.gq[b]); ori_m[k_] = [float(er.mean()), float(er[-1]), float(np.mean(f.tilt_deg(qa, m.gq[b])))]
    return dict(name=n, kf=kf, lvl=lvl, seed=seed, rotor=bool(e["rotor"]), pos=pos_m, ori=ori_m, disp=[float(np.linalg.norm(Dtrue, axis=1).mean()), float(np.linalg.norm(Dtrue[-1]))], speed_T=float(np.linalg.norm(m.s.gt_v[iT])), vel_err=float(np.linalg.norm(vT - vgt)))
if __name__ == "__main__":
    names = f.available(); jobs = [(n, kf, l, sd) for n in names for kf in KFS for l in LEVELS for sd in range(1 if l == "clean" else 2)]
    with Pool(10) as p: res = p.map(one, jobs, chunksize=2)
    json.dump(res, open(os.path.join(p3.ROOT, "research/results/e20_blind.json"), "w")); PK = list(res[0]["pos"]); OK_ = list(res[0]["ori"])
    for grp, sel in (("EuRoC", lambda r: not r["name"].startswith("room")), ("TUM", lambda r: r["name"].startswith("room"))):
        print("\n================ %s : position predictors [anc / end / abs], n per level = %s; true 5 s displacement mean %.2f m, end %.2f m; speed at failure %.2f m/s; velocity estimate error at failure (clean) %.2f m/s" % (grp, [sum(sel(r) and r["lvl"] == l for r in res) for l in LEVELS], np.mean([r["disp"][0] for r in res if sel(r)]), np.mean([r["disp"][1] for r in res if sel(r)]), np.mean([r["speed_T"] for r in res if sel(r)]), np.mean([r["vel_err"] for r in res if sel(r) and r["lvl"] == "clean"])))
        print("%-18s " % "predictor" + " | ".join("%-17s" % l for l in LEVELS))
        for k in PK: print("%-18s " % k + " | ".join("%.2f / %.2f / %.2f" % tuple(np.mean([r["pos"][k][i] for r in res if sel(r) and r["lvl"] == l]) for i in (0, 1, 2)) for l in LEVELS))
        print("\n%s : orientation predictors [mean / end / tilt, deg]" % grp)
        best = sorted(OK_, key=lambda k: np.mean([r["ori"][k][0] for r in res if sel(r)]))[:14]
        for k in ["hold"] + [x for x in best if x != "hold"]: print("%-22s " % k + " | ".join("%5.1f / %5.1f / %4.1f" % tuple(np.mean([r["ori"][k][i] for r in res if sel(r) and r["lvl"] == l]) for i in (0, 1, 2)) for l in LEVELS))
    print("\n================ by failure point (clean, anc): kf | EuRoC hold model cv_1 | TUM hold model cv_1 | true displacement EuRoC TUM")
    for kf in KFS:
        g = lambda k, s_: np.mean([r["pos"][k][0] for r in res if r["kf"] == kf and r["lvl"] == "clean" and r["name"].startswith("room") == s_]); dd = lambda s_: np.mean([r["disp"][0] for r in res if r["kf"] == kf and r["lvl"] == "clean" and r["name"].startswith("room") == s_])
        print("  %-4s | %.2f %.2f %.2f | %.2f %.2f %.2f | %.2f %.2f" % (kf, g("hold", False), g("model", False), g("cv_1", False), g("hold", True), g("model", True), g("cv_1", True), dd(False), dd(True)))
