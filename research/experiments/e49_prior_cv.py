#!/usr/bin/env python3
"""E49 - are the motion priors (tuned in Phase 1B/2 on clean data) right for OFFICIAL data, and how much do they overfit?
Rotor model grid: T_p {10, 20, 40} s x tau_v {1, 2, 4} s x sigma_a {0.5, 1, 2} m/s^2 x drag-row sd {1.0, 1.4, 2.0} (81) on EuRoC;
hand-held Matern grid: ell {0.5, 1, 2} s x sigma {1, 2, 4} m x tau_b {10, 20, 40} s x sigma_b {0.05, 0.1, 0.2} (81) on TUM-VI.
Front end fixed per mission (rotor: spline; hand-held: const). Selection: rotor leave-one-environment-out (MH / V1 / V2), hand-held
leave-one-sequence-out (6 rooms); criterion = mean ATE / motionless ATE. Reports default, in-sample best, and cross-validated choice."""
import sys, os, json, itertools, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p4, fbs6 as f
f.fbs5.GS_K2 = 0.5
CFGS = [("full", dict(seed=42)), ("full", dict(seed=1)), ("crop60", dict(seed=42, crop=0.6))]
RG = list(itertools.product((10.0, 20.0, 40.0), (1.0, 2.0, 4.0), (0.5, 1.0, 2.0), (1.0, 1.4, 2.0))); HG = list(itertools.product((0.5, 1.0, 2.0), (1.0, 2.0, 4.0), (10.0, 20.0, 40.0), (0.05, 0.1, 0.2)))
RDEF = (20.0, 2.0, 1.0, 1.4); HDEF = (1.0, 2.0, 20.0, 0.1)
def run(args):
    n, o, cname, ckw = args; m = p4.official(n, o, **ckw); rotor = m.rotor_true
    t, wu, au, valid, rows = f.ingest2(m.tn, m.w, m.a); L, nt = f.split_tail(t, wu, au, valid); t0, wu0, au0, v0 = t[:L + 1], wu[:L + 1], au[:L + 1], valid[:L + 1]
    wu0, au0, v0, _ = f.repair(t0, wu0, au0, v0, g_lo=5.0, g_hi=15.5); cm = f.common_of(t0, wu0, au0, v0); q3, base, h, anc = f.fe_const(t0, wu0, au0, v0, True, True)
    if rotor: q3, base, _ = f.fe_spline(t0, wu0, au0, v0, True, True, 3.0, base, h)
    P = f.prep_from(t0, wu0, au0, v0, q3, base, True, cm); out = {"motionless": p4.score(m, np.zeros((len(m.tn), 3)))["ate"]}
    for g in (RG if rotor else HG):
        if rotor: model = f.model_spring(g[0], g[1], 0.3, g[2], 20.0, 0.1); e = f.fbs4.estimate_v6(None, None, None, n_blind=nt, k_drag=m.kd, P=P, gap_ref=2.0, model=model, force="drag", sd=g[3])
        else: model = f.model_matern52(*g); e = f.fbs4.estimate_v6(None, None, None, n_blind=nt, k_drag=m.kd, P=P, gap_ref=2.0, model=model, force="matern")
        sc = p4.score(m, e["p"][rows]); out[str(g)] = [sc["ate"], sc["wp"], sc["b_abs"]]
    return dict(name=n, order=o, cfg=cname, rotor=rotor, m=out)
if __name__ == "__main__":
    names = f.available()
    with Pool(10) as p: res = p.map(run, [(n, o, c, kw) for o in ("A", "B") for c, kw in CFGS for n in names], chunksize=1)
    json.dump(res, open(os.path.join(p4.ROOT, "research/results/e49_prior_cv.json"), "w"))
    grp = lambda n: "TUM" if n.startswith("room") else n[:2]
    for plat, G, DEF, folds in (("rotor (EuRoC)", RG, RDEF, ("MH", "V1", "V2")), ("hand-held (TUM)", HG, HDEF, tuple("room%d" % i for i in range(1, 7)))):
        rr = [r for r in res if r["rotor"] == plat.startswith("rotor")]; key = (lambda r: grp(r["name"])) if plat.startswith("rotor") else (lambda r: r["name"])
        nat = lambda r, g: r["m"][str(g)][0] / r["m"]["motionless"]; mean = lambda L, g, i=0: float(np.mean([r["m"][str(g)][i] for r in L]))
        best = min(G, key=lambda g: np.mean([nat(r, g) for r in rr])); print("\n=== %s: %d missions | default %s ATE %.3f wp %.3f | in-sample best %s ATE %.3f wp %.3f" % (plat, len(rr), DEF, mean(rr, DEF), mean(rr, DEF, 1), best, mean(rr, best), mean(rr, best, 1)))
        cv = []; dflt = []
        for h in folds:
            tr = [r for r in rr if key(r) != h]; te = [r for r in rr if key(r) == h]
            if not te: continue
            b = min(G, key=lambda g: np.mean([nat(r, g) for r in tr])); cv += [r["m"][str(b)][0] for r in te]; dflt += [r["m"][str(DEF)][0] for r in te]; print("   held-out %-6s -> %s  ATE %.3f (default %.3f)" % (h, b, mean(te, b), mean(te, DEF)))
        print("   cross-validated ATE %.3f vs default %.3f (%+.1f %%)" % (np.mean(cv), np.mean(dflt), 100 * (np.mean(cv) / np.mean(dflt) - 1)))
        top = sorted(G, key=lambda g: np.mean([nat(r, g) for r in rr]))[:8]; print("   top-8 in-sample: " + "; ".join("%s %.3f" % (g, mean(rr, g)) for g in top))
