#!/usr/bin/env python3
"""E07 - one model for smoothing, dropout bridging and the blind window: spring-damper prior vs Matern-5/2. Research instrument.
Stage A: .venv/bin/python research/experiments/e07_spring.py A      (clean + 3x2 s gaps, whole grid, LOSO selection)
Stage B: .venv/bin/python research/experiments/e07_spring.py B i j  (5 s model-consistent forecast for grid indices i j ...)
Ground truth only scores; corruptions are our assumptions."""
import sys, os, json, itertools, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbs
from e03_degrade import CFGS
GRID = [("matern", (1, 2, 20, 0.1))] + [("spring", c) for c in itertools.product([20, 40, 80], [0.5, 1, 2, 4], [0.3, 1.0], [1.0, 3.0], [20], [0.1])]
def mk(g): return fbs.model_matern52(*g[1]) if g[0] == "matern" else fbs.model_spring(*g[1])
def stageA(args):
    n, cname, seed = args; s = fbs.load(n); ok = s.ok
    w, a, valid, meta = fbs.degrade(s.t, s.w, s.a, CFGS[cname], seed=1000 * seed + sum(n.encode()) % 997)
    q, bg, quiet, wf, af = fbs.front_end2(s.t, w, a, valid); ml = fbs.ate(np.zeros((ok.sum(), 3)), s.gt_p[ok]); Rz, tz = fbs.umeyama(np.zeros((ok.sum(), 3)), s.gt_p[ok]); out = []
    for g in GRID:
        p = fbs.lin_smooth(s.t, q, af, valid, mk(g), bias_jump=0.5)[0]; row = [fbs.ate(p[ok], s.gt_p[ok])]
        if meta["gaps"]:
            Rm, tt = fbs.umeyama(p[ok], s.gt_p[ok]); e = [[], [], [], []]
            for (t0, dur) in meta["gaps"]:
                m = (s.t >= t0) & (s.t < t0 + dur) & ok
                if m.sum() < 10: continue
                i0 = np.where(m)[0][0] - 1; dg = s.gt_p[m] - s.gt_p[i0]
                e[0].append(np.linalg.norm((p[m] - p[i0]) @ Rm.T - dg, axis=1).mean()); e[1].append(np.linalg.norm(dg, axis=1).mean())
                e[2].append(np.linalg.norm(p[m] @ Rm.T + tt - s.gt_p[m], axis=1).mean()); e[3].append(np.linalg.norm(tz - s.gt_p[m], axis=1).mean())
            row += [float(np.mean(x)) if x else np.nan for x in e]
        out.append(row)
    return n, cname, ml, out
def stageB(args):
    n, sel = args; s = fbs.load(n); fs = 1 / np.median(np.diff(s.t)); nb = int(5 * fs); okidx = np.where(s.ok)[0]; t0, t1 = s.t[okidx[0]], s.t[okidx[-1]]; rows = []
    for T in np.arange(max(40.0, t0 + 20), t1 - 5.1, 10.0):
        iT = int(np.searchsorted(s.t, T)); sl = slice(0, iT + 1); fut = np.arange(iT + 1, iT + 1 + nb, 20)
        if not s.ok[iT + 1:iT + 1 + nb].all() or not s.ok[iT]: continue
        t = s.t[sl]; valid = np.ones(len(t), bool); q, bg, quiet, wf, af = fbs.front_end2(t, s.w[sl], s.a[sl], valid)
        tau = s.t[fut] - s.t[iT]; dgt = s.gt_p[fut] - s.gt_p[iT]; row = {"hold": float(np.linalg.norm(dgt, axis=1).mean())}
        for gi in sel:
            g = GRID[gi]; model = mk(g); p, xs = fbs.lin_smooth(t, q, af, valid, model)[:2]; Rm, _ = fbs.umeyama(p[s.ok[sl]], s.gt_p[sl][s.ok[sl]]); x = xs[-1]
            d = fbs.model_predict(model[0], x, tau); row["m%d_model" % gi] = float(np.linalg.norm(d @ Rm.T - dgt, axis=1).mean())
            d1 = (1.0 * (1 - np.exp(-tau / 1.0)))[:, None] * x[:, 1]; row["m%d_damp1" % gi] = float(np.linalg.norm(d1 @ Rm.T - dgt, axis=1).mean())
            row["m%d_vel_err" % gi] = float(np.linalg.norm(Rm @ x[:, 1] - s.gt_v[iT]))
        rows.append(row)
    return n, rows
if __name__ == "__main__":
    names = fbs.available(); mav = np.array([not n.startswith("room") for n in names])
    if sys.argv[1] == "A":
        jobs = [(n, "clean", 0) for n in names] + [(n, "gaps_3x2s", sd) for n in names for sd in (0, 1)]
        with Pool(10) as p: res = p.map(stageA, jobs, chunksize=2)
        json.dump([[r[0], r[1], r[2], r[3]] for r in res], open(os.path.join(fbs.ROOT, "research/results/e07_spring_A.json"), "w"))
        C = np.array([[row[0] for row in r[3]] for r in res if r[1] == "clean"]); M = np.array([r[2] for r in res if r[1] == "clean"]); sk = 1 - C / M[:, None]
        Gp = [r for r in res if r[1] == "gaps_3x2s"]; GA = np.array([[row[0] for row in r[3]] for r in Gp]); GM = np.array([r[2] for r in Gp]); gsk = 1 - GA / GM[:, None]
        anc = np.nanmean(np.array([[row[1] for row in r[3]] for r in Gp]), 0); hold = np.nanmean([r[3][0][2] for r in Gp]); gabs = np.nanmean(np.array([[row[3] for row in r[3]] for r in Gp]), 0); gml = np.nanmean([r[3][0][4] for r in Gp])
        loso = [C[i, int(np.argmax(sk[np.arange(len(names)) != i].mean(0)))] for i in range(len(names))]
        print("LOSO-selected clean skill: ALL %.1f%%  MAV %.1f%%  handheld %.1f%%" % (100 * np.mean(1 - np.array(loso) / M), 100 * np.mean((1 - np.array(loso) / M)[mav]), 100 * np.mean((1 - np.array(loso) / M)[~mav])))
        print("gap reference: anchored hold %.3f m, absolute motionless %.3f m" % (hold, gml))
        print("idx model   params                         clean skill ALL/MAV/hand   gaps3x2s skill ALL/MAV/hand   gap anchored  gap abs")
        gm2 = np.repeat(mav, 2); order = np.argsort(-(sk.mean(0) + gsk.mean(0)))
        for i in list(order[:14]) + [0]:
            print("%3d %-7s %-30s  %5.1f /%5.1f /%5.1f        %5.1f /%5.1f /%5.1f        %.3f        %.3f" % (i, GRID[i][0], str(GRID[i][1]), 100 * sk[:, i].mean(), 100 * sk[mav, i].mean(), 100 * sk[~mav, i].mean(), 100 * gsk[:, i].mean(), 100 * gsk[gm2, i].mean(), 100 * gsk[~gm2, i].mean(), anc[i], gabs[i]))
    else:
        sel = [int(x) for x in sys.argv[2:]]
        with Pool(10) as p: res = p.map(stageB, [(n, sel) for n in names])
        json.dump(dict(res), open(os.path.join(fbs.ROOT, "research/results/e07_spring_B.json"), "w"))
        for gname, f in [("ALL", lambda n: True), ("MAV", lambda n: not n.startswith("room")), ("handheld", lambda n: n.startswith("room"))]:
            rows = [r for n, rr in res if f(n) for r in rr]; line = "%-8s n=%d hold %.3f |" % (gname, len(rows), np.mean([r["hold"] for r in rows]))
            for gi in sel: line += " m%d %s: model %.3f damp1 %.3f velerr %.3f |" % (gi, GRID[gi][0], np.mean([r["m%d_model" % gi] for r in rows]), np.mean([r["m%d_damp1" % gi] for r in rows]), np.mean([r["m%d_vel_err" % gi] for r in rows]))
            print(line)
