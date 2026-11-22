#!/usr/bin/env python3
"""E13 - model selection against the mission components (waypoint / blind / in-gap error), not against clean ATE.
One front end per mission, many position models on top; settings are chosen leave-one-sequence-out on the local
composite (lin-vs-motionless, weights 30/20/10). Variants: Matern and spring-damper grids, bias drift on/off,
heading-uncertainty shrinkage after gaps, blind rows = model prediction or hold, and (EuRoC, gated) rotor drag.
OUR simulator, OUR composite: neither is the organiser's. Run: .venv/bin/python research/experiments/e13_optimise.py"""
import sys, os, json, itertools, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbs3 as f
from e10_mission import LEVELS, k_loso
MODELS = {"M(1,2)": ("m", (1, 2, 20, 0.1)), "M(2,2)": ("m", (2, 2, 20, 0.1)), "M(4,4)": ("m", (4, 4, 20, 0.1))}
for c in itertools.product([20, 60], [0.5, 1, 2], [0.3, 1.0], [1.0, 3.0]): MODELS["S%s" % (c,)] = ("s", c + (20, 0.1))
DRAGM = ["M(2,2)", "M(4,4)"] + ["S%s" % (c,) for c in itertools.product([20, 60], [0.5, 2], [0.3], [1.0, 3.0])]
def mk(key): kind, c = MODELS[key]; return f.model_matern52(*c) if kind == "m" else f.model_spring(*c)
def work(args):
    n, lvl, kf = args; s = f.load(n); fs = 1 / np.median(np.diff(s.t)); nb = int(round(5 * fs)); oi = np.where(s.ok)[0]; t0, t1 = s.t[oi[0]], s.t[oi[-1]]
    iT = int(np.searchsorted(s.t, [t1 - 5.05, t0 + 0.6 * (t1 - t0), t0 + 0.8 * (t1 - t0)][kf]))
    while not s.ok[iT:iT + nb + 1].all() and iT > 2000: iT -= 40
    sl = slice(0, iT + 1); w, a, valid, meta = f.degrade(s.t[sl], s.w[sl], s.a[sl], LEVELS[lvl], seed=7 * kf + sum(n.encode()) % 997)
    M = iT + 1 + nb; gp = s.gt_p[:M]; ok = s.ok[:M]; t = s.t[:M]; okp = ok & (np.arange(M) <= iT); b = np.arange(iT + 1, M)
    cp = np.searchsorted(s.t, np.linspace(t0 + 10, s.t[iT], 6)).clip(oi[0], iT); cp = np.array([c if ok[c] else oi[np.argmin(np.abs(oi - c))] for c in cp]); out = {}
    def score(p):
        Rm, tt = f.umeyama(p[okp], gp[okp]); er = np.linalg.norm(p @ Rm.T + tt - gp, axis=1); ga = [er[(t >= g0) & (t < g0 + d) & okp].mean() for g0, d in meta["gaps"] if ((t >= g0) & (t < g0 + d) & okp).sum() > 10]
        return [float(er[cp].mean()), float(er[b].mean()), float(np.linalg.norm((p[b] - p[iT]) @ Rm.T - (gp[b] - gp[iT]), axis=1).mean()), float(np.mean(ga)) if ga else float("nan"), float(np.sqrt(np.mean(er[okp] ** 2))), f.rpe_disp(t[okp], p[okp], gp[okp], Rm), float(er[b[-1]])]
    def variants(tag, p, fac):
        for sh in (1.0, 0.6):
            p2 = p.copy()
            if sh < 1: p2[:, :2] *= (fac ** (np.log(sh) / np.log(0.6)))[:, None]
            out["%s|sh%g|model" % (tag, sh)] = score(p2); p3 = p2.copy(); p3[b] = p3[iT]; out["%s|sh%g|hold" % (tag, sh)] = score(p3)
    out["motionless"] = score(np.zeros((M, 3)))
    for drift in (False, True):
        tu, q, af, vu, lock, N0 = f.prep3(s.tn[sl], w, a, n_blind=nb, drift=drift); fac = f.gap_shrink_factors(tu, vu, 0.6)
        for key in MODELS: variants("%s|drift%d|nodrag" % (key, drift), f.lin_smooth(tu, q, af, vu, mk(key), bias_jump=0.5)[0], fac)
        if not drift and lock < 0.04:
            for key in DRAGM:
                for sd in (1.0, 2.0): variants("%s|drift0|drag%g" % (key, sd), f.run_joint3(tu, q, af, mk(key), vu, k_drag=k_loso(n), sd=sd)[0], fac)
    return n, lvl, kf, out
def comp(o, k, w=(30, 20, 10)):
    v = o[k]; r = o["motionless"]; parts = [(np.clip(1 - v[i] / r[i], 0, 1), wg) for i, wg in zip((0, 1, 3), w) if not np.isnan(v[i])]; return sum(a * b_ for a, b_ in parts) / sum(b_ for _, b_ in parts)
if __name__ == "__main__":
    names = f.available(); jobs = [(n, l, kf) for l in ("clean", "L2") for n in names for kf in range(3)]
    with Pool(10) as p: res = p.map(work, jobs, chunksize=1)
    json.dump([[r[0], r[1], r[2], r[3]] for r in res], open(os.path.join(f.fbs.ROOT, "research/results/e13_optimise.json"), "w"))
    def table(pool_names, keys, title):
        rr = [r for r in res if r[0] in pool_names]; C = np.array([[100 * comp(r[3], k) for k in keys] for r in rr]); seqs = np.array([r[0] for r in rr]); held = []; picks = []
        for n in pool_names:
            j = int(np.argmax(C[seqs != n].mean(0))); held.append(C[seqs == n, j].mean()); picks.append(keys[j])
        best = int(np.argmax(C.mean(0))); print("\n== %s: %d missions, %d variants | LOSO-selected composite %.1f | best single variant %.1f: %s | most-picked: %s (%d/%d)" % (title, len(rr), len(keys), np.mean(held), C.mean(0)[best], keys[best], max(set(picks), key=picks.count), picks.count(max(set(picks), key=picks.count)), len(picks)))
        return rr, C, keys
    def show(rr, key):
        g = lambda i, sel=lambda r: True: np.nanmean([r[3][key][i] for r in rr if sel(r)]); m = lambda i: np.nanmean([r[3]["motionless"][i] for r in rr])
        print("   %-44s composite %5.1f | wp %.2f (ref %.2f) blind abs %.2f (%.2f) anchored %.2f (%.2f) end %.2f in-gap %.2f (%.2f) ATE %.2f (%.2f) RPE %.2f | clean comp %.1f L2 comp %.1f" % (key, 100 * np.mean([comp(r[3], key) for r in rr]), g(0), m(0), g(1), m(1), g(2), m(2), g(6), g(3), m(3), g(4), m(4), g(5),
              100 * np.mean([comp(r[3], key) for r in rr if r[1] == "clean"]), 100 * np.mean([comp(r[3], key) for r in rr if r[1] == "L2"])))
    nod = [k for k in res[0][3] if "nodrag" in k]; rr, C, keys = table(names, nod, "no drag, all 17 sequences")
    order = np.argsort(-C.mean(0))
    for j in list(order[:6]): show(rr, keys[j])
    for k in ["S(20, 0.5, 0.3, 1.0)|drift0|nodrag|sh1|model", "S(20, 0.5, 0.3, 1.0)|drift0|nodrag|sh1|hold", "S(20, 0.5, 0.3, 1.0)|drift1|nodrag|sh1|model", "S(20, 0.5, 0.3, 1.0)|drift0|nodrag|sh0.6|model", "M(1,2)|drift0|nodrag|sh1|model", "M(1,2)|drift0|nodrag|sh0.6|hold"]: show(rr, k)
    for grp, pool in [("TUM only", [n for n in names if n.startswith("room")]), ("EuRoC only, drag allowed", [n for n in names if not n.startswith("room")])]:
        ks = nod if grp == "TUM only" else [k for k in [r for r in res if not r[0].startswith("room")][0][3] if k != "motionless"]; rr, C, keys = table(pool, ks, grp); order = np.argsort(-C.mean(0))
        for j in list(order[:5]): show(rr, keys[j])
        if grp != "TUM only":
            nd = [j for j in order if "nodrag" in keys[j]][:2]
            for j in nd: show(rr, keys[j])
            for k in ["S(20, 0.5, 0.3, 1.0)|drift0|drag2|sh1|model", "M(4,4)|drift0|drag2|sh1|model", "M(4,4)|drift0|drag2|sh1|hold"]: show(rr, k)
