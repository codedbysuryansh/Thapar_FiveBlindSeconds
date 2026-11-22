#!/usr/bin/env python3
"""E04 - five-second blind prediction from the ESTIMATED state (not the oracle state). Research instrument.
For each failure time T the estimator sees only the IMU stream up to T (attitude, bias, smoother all re-run on the
truncated stream), then predicts 5 s with no data. Ground truth only scores. Optional degradation via e03 (import).
Run: .venv/bin/python research/experiments/e04_forecast.py [step_s]"""
import sys, os, json, numpy as np
from multiprocessing import Pool
from scipy.linalg import expm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import fbs
GP = (1, 2, 20, 0.1); HOR = 5.0
PRED = ["hold", "cv", "damp0.5", "damp1", "damp2", "damp4", "shrink_damp1", "shrink_damp2", "gp_prior", "ca"]
def predict(kind, tau, p, v, a, Pv, F4, x4):
    if kind == "hold": return np.zeros((len(tau), 3))
    if kind == "cv": return tau[:, None] * v
    if kind == "ca": return tau[:, None] * v + 0.5 * tau[:, None] ** 2 * a
    if kind.startswith("damp"): td = float(kind[4:]); return (td * (1 - np.exp(-tau / td)))[:, None] * v
    if kind.startswith("shrink_damp"):
        td = float(kind[11:]); w = (v @ v) / (v @ v + Pv); return (td * (1 - np.exp(-tau / td)))[:, None] * v * w
    if kind == "gp_prior":
        out = np.zeros((len(tau), 3))
        for i in range(0, len(tau), 20):
            E = expm(F4 * tau[i])[0]; out[i:i + 20] = (x4 @ E) - p
        return out
def work(n, step=5.0):
    s = fbs.load(n); fs = 1 / np.median(np.diff(s.t)); mav = fbs.platform(n) == "mav"; nb = int(HOR * fs)
    okidx = np.where(s.ok)[0]; t0, t1 = s.t[okidx[0]], s.t[okidx[-1]]; rows = []
    F4 = fbs.model_matern52(*GP)[0]
    ests = {"gp": dict(gp=GP, k_drag=0.0)}
    if mav: ests["drag"] = dict(gp=(4, 4, 20, 0.1), k_drag=0.18, sd=2.0)
    for T in np.arange(max(30.0, t0 + 20), t1 - HOR - 0.1, step):
        iT = int(np.searchsorted(s.t, T)); sl = slice(0, iT + 1); fut = slice(iT + 1, iT + 1 + nb)
        if not s.ok[fut].all() or not s.ok[iT]: continue
        t, w, a = s.t[sl], s.w[sl], s.a[sl]; q, bg, quiet = fbs.estimate_attitude(t, w, a); sn = fbs.hf_std(fbs.world_acc_centered(q, a))
        tau = s.t[fut] - s.t[iT]; dgt = s.gt_p[fut] - s.gt_p[iT]; row = {"T": float(T), "gt_disp_end": float(np.linalg.norm(dgt[-1])), "gt_speed": float(np.linalg.norm(s.gt_v[iT]))}
        # orientation: body-frame increment predicted from the last bias-corrected gyro samples
        wl = (w[-int(0.2 * fs):] - bg).mean(0); dq_gt = fbs.qmul(fbs.qinv(s.gt_q[iT])[None], s.gt_q[fut])
        for name, th in [("hold", None), ("const_w", 1e9), ("damp_w0.25", 0.25), ("damp_w0.5", 0.5), ("damp_w1", 1.0)]:
            ang = np.zeros((len(tau), 3)) if th is None else (th * (1 - np.exp(-tau / th)))[:, None] * wl if th < 1e8 else tau[:, None] * wl
            e = fbs.ang_deg(fbs.qexp(ang), dq_gt); row["ori_%s_mean" % name] = float(e.mean()); row["ori_%s_end" % name] = float(e[-1])
        for en, kw in ests.items():
            r = fbs.run_joint(t, q, a, sn=sn, **kw); pok = s.ok[sl]; Rm, tt = fbs.umeyama(r["p"][pok], s.gt_p[sl][pok])
            pT, vT = r["p"][-1], r["v"][-1]; aT = r["xs"][-1, [2, 6, 10]]; Pv = float(r["Ps"][-1][[1, 5, 9], [1, 5, 9]].sum()); x4 = r["xs"][-1, :12].reshape(3, 4)[:, :3]
            F3 = fbs.model_matern52(*kw["gp"])[0][:3, :3]
            row[en + "_vel_err"] = float(np.linalg.norm(Rm @ vT - s.gt_v[iT])); row[en + "_pos_err_T"] = float(np.linalg.norm(Rm @ pT + tt - s.gt_p[iT])); row[en + "_vel_sigma"] = float(np.sqrt(Pv))
            for k in PRED:
                d = predict(k, tau, pT, vT, aT, Pv, F3, x4); e = np.linalg.norm(d @ Rm.T - dgt, axis=1); eabs = np.linalg.norm((pT + d) @ Rm.T + tt - s.gt_p[fut], axis=1)
                row["%s_%s_mean" % (en, k)] = float(e.mean()); row["%s_%s_end" % (en, k)] = float(e[-1]); row["%s_%s_abs" % (en, k)] = float(eabs.mean())
        # oracle-state references on the same window (ground-truth velocity; NOT available to an estimator)
        for td in [1.0, 2.0]:
            e = np.linalg.norm((td * (1 - np.exp(-tau / td)))[:, None] * s.gt_v[iT] - dgt, axis=1); row["oracle_damp%g_mean" % td] = float(e.mean())
        rows.append(row)
    return n, rows
if __name__ == "__main__":
    step = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0; names = fbs.available()
    with Pool(min(10, len(names))) as p: res = p.starmap(work, [(n, step) for n in names])
    out = dict(res); json.dump(out, open(os.path.join(fbs.ROOT, "research/results/e04_forecast.json"), "w"))
    grp = {"MH": [], "V1": [], "V2": [], "TUM": [], "ALL": []}
    for n, rows in res:
        grp[fbs.group(n)] += rows; grp["ALL"] += rows
    def m(rows, k): v = [r[k] for r in rows if k in r]; return np.mean(v) if v else float("nan")
    print("windows:", {g: len(v) for g, v in grp.items()})
    print("ANCHORED 5 s displacement error, mean over window (m); 'hold' equals the oracle-hold floor by construction")
    print("grp   est   vel_err  " + " ".join("%12s" % k for k in PRED) + "   oracle_d1 oracle_d2")
    for g, rows in grp.items():
        for en in ["gp", "drag"]:
            if np.isnan(m(rows, en + "_hold_mean")): continue
            print("%-5s %-5s %6.3f   " % (g, en, m(rows, en + "_vel_err")) + " ".join("%12.3f" % m(rows, "%s_%s_mean" % (en, k)) for k in PRED) + "   %8.3f %8.3f" % (m(rows, "oracle_damp1_mean"), m(rows, "oracle_damp2_mean")))
    print("ENDPOINT (t = 5 s) error (m)")
    for g, rows in grp.items():
        for en in ["gp", "drag"]:
            if np.isnan(m(rows, en + "_hold_end")): continue
            print("%-5s %-5s          " % (g, en) + " ".join("%12.3f" % m(rows, "%s_%s_end" % (en, k)) for k in PRED))
    print("ABSOLUTE blind-window error under the pre-failure SE3 alignment (m) and position error at T")
    for g, rows in grp.items():
        for en in ["gp", "drag"]:
            if np.isnan(m(rows, en + "_hold_abs")): continue
            print("%-5s %-5s posT %6.3f " % (g, en, m(rows, en + "_pos_err_T")) + " ".join("%12.3f" % m(rows, "%s_%s_abs" % (en, k)) for k in PRED))
    print("ORIENTATION error over the blind window, mean / endpoint (deg)")
    for g, rows in grp.items():
        print("%-5s " % g + "  ".join("%s %.1f/%.1f" % (k, m(rows, "ori_%s_mean" % k), m(rows, "ori_%s_end" % k)) for k in ["hold", "const_w", "damp_w0.25", "damp_w0.5", "damp_w1"]))
