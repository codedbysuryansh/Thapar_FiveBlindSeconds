#!/usr/bin/env python3
"""E11 - Scenario A: what if checkpoint information IS supplied to the algorithm? (The slides do not say; Scenario B,
evaluator-only checkpoints, is e10.) Variants, all using only what the scenario supplies:
  A1  positions + timestamps of 6 checkpoints -> position fixes in the smoother after solving the frame from them.
  A1-LOO  one interior checkpoint withheld and scored (how good is the trajectory BETWEEN supplied fixes?).
  A1-3    only 3 checkpoints supplied (first, middle, last).
  A2  positions only, times unknown (ordered): times start evenly spaced and are re-assigned to the closest approach.
  target  the strike point at T+5 s is also supplied: blind path = cubic Hermite from the state at T to the target.
Ground truth appears only as (a) the supplied checkpoint coordinates of the scenario and (b) scoring.
Run: .venv/bin/python research/experiments/e11_checkpoints.py"""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbs3 as f
from e10_mission import LEVELS
def scen(args):
    n, lvl, seed = args; s = f.load(n); fs = 1 / np.median(np.diff(s.t)); nb = int(round(5 * fs)); oi = np.where(s.ok)[0]; t0, t1 = s.t[oi[0]], s.t[oi[-1]]
    iT = int(np.searchsorted(s.t, t1 - 5.05))
    while not s.ok[iT:iT + nb + 1].all() and iT > 2000: iT -= 40
    sl = slice(0, iT + 1); w, a, valid, meta = f.degrade(s.t[sl], s.w[sl], s.a[sl], LEVELS[lvl], seed=1000 * seed + sum(n.encode()) % 997); tn = s.tn[sl]
    M = iT + 1 + nb; gp = s.gt_p[:M]; ok = s.ok[:M]; t = s.t[:M]; okp = ok & (np.arange(M) <= iT); b = np.arange(iT + 1, M); rng = np.random.default_rng(seed + 5)
    near = lambda c: int(c if ok[c] else oi[np.argmin(np.abs(oi - c))])
    cp = np.array([near(c) for c in np.searchsorted(s.t, np.linspace(t0 + 10, s.t[iT], 6)).clip(oi[0], iT)])
    e0 = f.estimate3(tn, w, a, n_blind=nb); R0, tt0 = f.umeyama(e0["p"][okp], gp[okp]); er0 = np.linalg.norm(e0["p"] @ R0.T + tt0 - gp, axis=1)
    out = {"name": n, "lvl": lvl, "B_ate": float(np.sqrt(np.mean(er0[okp] ** 2))), "B_wp": float(er0[cp].mean()), "B_blind": float(er0[b].mean()), "B_blind_end": float(er0[b[-1]])}
    def run(idx, pos, iters=2, sigma=0.1):
        p = e0["p"]
        for _ in range(iters):
            Rm, tt = f.umeyama(p[idx], pos); e = f.estimate3(tn, w, a, n_blind=nb, pos_fix=(idx, (pos - tt) @ Rm, sigma)); p = e["p"]
        Rm, tt = f.umeyama(p[idx], pos); return p, Rm, tt, e
    def metrics(tag, p, Rm, tt, idx):
        er = np.linalg.norm(p @ Rm.T + tt - gp, axis=1); mids = np.array([near((i + j) // 2) for i, j in zip(idx[:-1], idx[1:])])
        out[tag + "_ate"] = float(np.sqrt(np.mean(er[okp] ** 2))); out[tag + "_mid"] = float(er[mids].mean()); out[tag + "_blind"] = float(er[b].mean()); out[tag + "_blind_end"] = float(er[b[-1]])
        ga = [er[(t >= g0) & (t < g0 + d) & okp].mean() for g0, d in meta["gaps"] if ((t >= g0) & (t < g0 + d) & okp).sum() > 10]; out[tag + "_gap"] = float(np.mean(ga)) if ga else float("nan"); return er
    pA, RA, tA, eA = run(cp, gp[cp]); metrics("A1", pA, RA, tA, cp)
    loo = []
    for j in range(1, 5):
        idx = np.delete(cp, j); p, Rm, tt, _ = run(idx, gp[idx]); loo.append(np.linalg.norm(p[cp[j]] @ Rm.T + tt - gp[cp[j]]))
    out["A1_loo"] = float(np.mean(loo))
    c3 = cp[[0, 2, 5]]; p3, R3, t3, _ = run(c3, gp[c3]); er3 = metrics("A1k3", p3, R3, t3, c3); out["A1k3_other_cp"] = float(er3[cp[[1, 3, 4]]].mean())
    # target supplied as well: Hermite from (p_T, v_T) to the target at T+5 s with zero end velocity, in the checkpoint frame
    pT = pA[iT] @ RA.T + tA; vT = eA["v"][iT] @ RA.T; tgt = gp[b[-1]]; u = ((t[b] - t[iT]) / 5.0)[:, None]; h00 = 2 * u ** 3 - 3 * u ** 2 + 1; h10 = u ** 3 - 2 * u ** 2 + u; h01 = -2 * u ** 3 + 3 * u ** 2
    path = h00 * pT + h10 * 5.0 * vT + h01 * tgt; lin = pT + u * (tgt - pT); out["target_hermite"] = float(np.linalg.norm(path - gp[b], axis=1).mean()); out["target_linear"] = float(np.linalg.norm(lin - gp[b], axis=1).mean())
    out["A1_blind_hold"] = float(np.linalg.norm(pT - gp[b], axis=1).mean())
    # A2: positions known, times unknown. True times are jittered so that 'evenly spaced' is not secretly correct.
    tc = np.sort(np.linspace(t0 + 10, s.t[iT] - 8, 5) + rng.uniform(-4, 4, 5)); c2 = np.array([near(c) for c in np.searchsorted(s.t, tc).clip(oi[0], iT)] + [cp[-1]])
    guess = np.array([near(c) for c in np.searchsorted(s.t, np.linspace(t0 + 10, s.t[iT], 6)).clip(oi[0], iT)]); win = int(8 * fs); p = e0["p"]
    for it in range(4):
        Rm, tt = f.umeyama(p[guess], gp[c2]); pa = p @ Rm.T + tt; new = guess.copy()
        for j in range(5):
            lo = max(oi[0], guess[j] - win); hi = min(iT - 1, guess[j] + win); new[j] = lo + int(np.argmin(np.linalg.norm(pa[lo:hi] - gp[c2[j]], axis=1)))
        guess = np.sort(new); Rm, tt = f.umeyama(p[guess], gp[c2]); p = f.estimate3(tn, w, a, n_blind=nb, pos_fix=(guess, (gp[c2] - tt) @ Rm, 0.5))["p"]
    Rm, tt = f.umeyama(p[guess], gp[c2]); er = np.linalg.norm(p @ Rm.T + tt - gp, axis=1); out["A2_ate"] = float(np.sqrt(np.mean(er[okp] ** 2))); out["A2_time_err_s"] = float(np.mean(np.abs(s.t[guess[:5]] - s.t[c2[:5]])))
    out["A2_init_time_err_s"] = float(np.mean(np.abs(np.linspace(t0 + 10, s.t[iT], 6)[:5] - s.t[c2[:5]]))); out["A2_wp_true_time"] = float(er[c2].mean())
    return out
if __name__ == "__main__":
    names = f.available(); jobs = [(n, l, 0) for l in ("clean", "L2") for n in names]
    with Pool(10) as p: res = p.map(scen, jobs, chunksize=1)
    json.dump(res, open(os.path.join(f.fbs.ROOT, "research/results/e11_checkpoints.json"), "w"), indent=1)
    keys = ["B_ate", "B_wp", "B_blind", "B_blind_end", "A1_ate", "A1_mid", "A1_loo", "A1_gap", "A1_blind", "A1_blind_end", "A1_blind_hold", "A1k3_ate", "A1k3_other_cp", "target_hermite", "target_linear", "A2_ate", "A2_wp_true_time", "A2_time_err_s", "A2_init_time_err_s"]
    for lvl in ("clean", "L2"):
        for grp, sel in [("ALL", lambda n: True), ("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room"))]:
            rr = [r for r in res if r["lvl"] == lvl and sel(r["name"])]; print("%-5s %-5s " % (lvl, grp) + "  ".join("%s %.2f" % (k, np.nanmean([r[k] for r in rr])) for k in keys))
