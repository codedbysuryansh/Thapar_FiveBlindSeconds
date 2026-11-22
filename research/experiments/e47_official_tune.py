#!/usr/bin/env python3
"""E47 - the remaining free parameters of v5c on OFFICIAL missions, chosen by GROUPED cross-validation (leave one environment group out:
MH, V1, V2, TUM; criterion = mean ATE / motionless ATE over the other groups), plus the rate governor and the platform guard.
  k2 (gravity-smoother rotation-uncertainty law) in {0, 0.1, 0.25, 0.5, 1}: const and spline solutions for each;
  spline smoothness at k2 = 0.5: lam2 {0.3, 3, 30}, knot spacing {10, 20, 40} s;
  smoother rate governor at k2 = 0.5: dec {1, 2, 4, 8} (block-averaged rows, Hermite interpolation);
  guard: radius of the forced hand-held solution (ratio test against the rotor solution)."""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p4, fbs6 as f
CFGS = [("full", dict(seed=s)) for s in (42, 1, 2)] + [("crop60", dict(seed=42, crop=0.6)), ("crop80", dict(seed=42, crop=0.8)), ("lvl0.5", dict(seed=42, max_level=0.5)), ("lvl1.5", dict(seed=42, max_level=1.5))]
K2S = [0.0, 0.1, 0.25, 0.5, 1.0]
def radius(p): return float(np.sqrt(np.mean(np.sum((p - p.mean(0)) ** 2, 1))))
def solve(m, k2, fe_kw, be_kw, both=True):
    f.fbs5.GS_K2 = k2; kw = dict(fe="v5", gs=True, rep=True, both=both); kw.update(fe_kw); e = f.estimate_off(m.tn, m.w, m.a, k_drag=m.kd, be=dict(gap_ref=2.0, **be_kw), **kw); return e
def run(args):
    n, order, cname, ckw = args; m = p4.official(n, order, **ckw); out = {"motionless": p4.score(m, np.zeros((len(m.tn), 3)))}; info = {}
    for k2 in K2S:
        e = solve(m, k2, {}, {}); out["k2=%g|C" % k2] = p4.score(m, e["pC"], e["qC"]); out["k2=%g|S" % k2] = p4.score(m, e["pS"], e["qS"]); info["k2=%g" % k2] = dict(D=float(min(e["info"].get("D", 0), 1e3)), h=e["info"]["h"], rotor=e["rotor"], coarse=bool(e["info"].get("coarse")))
    for nm, kw in (("lam0.3", dict(lam2=0.3)), ("lam30", dict(lam2=30.0)), ("knot10", dict(lam2=3.0)), ("knot40", dict(lam2=3.0))):
        if nm.startswith("knot"):
            old = f.fe_spline.__defaults__; kn = float(nm[4:])
            f.fe_spline.__defaults__ = (3.0, kn) + old[2:]; e = solve(m, 0.5, dict(fe="spline", **kw), {}, both=False); f.fe_spline.__defaults__ = old
        else: e = solve(m, 0.5, dict(fe="spline", **kw), {}, both=False)
        out["k2=0.5|S " + nm] = p4.score(m, e["p"], e["q"])
    for d in (2, 4, 8):
        e = solve(m, 0.5, {}, dict(dec=d)); out["k2=0.5|C dec%d" % d] = p4.score(m, e["pC"], e["qC"]); out["k2=0.5|S dec%d" % d] = p4.score(m, e["pS"], e["qS"])
    e = solve(m, 0.5, dict(fe="const"), dict(force="matern"), both=False); info["hand_radius"] = radius(e["p"]); out["k2=0.5|C hand"] = p4.score(m, e["p"], e["q"])
    e = solve(m, 0.5, dict(fe="const"), dict(force="drag"), both=False); info["rotor_radius"] = radius(e["p"]); out["k2=0.5|C rotor"] = p4.score(m, e["p"], e["q"])
    f.fbs5.GS_K2 = 1.0
    return dict(name=n, order=order, cfg=cname, seed=ckw["seed"], m=out, info=info, rotor_true=m.rotor_true)
if __name__ == "__main__":
    names = f.available()
    with Pool(10) as p: res = p.map(run, [(n, o, c, kw) for o in ("A", "B") for c, kw in CFGS for n in names], chunksize=1)
    json.dump(res, open(os.path.join(p4.ROOT, "research/results/e47_official_tune.json"), "w"))
    grp = lambda n: "TUM" if n.startswith("room") else n[:2]; G = ["MH", "V1", "V2", "TUM"]
    def nate(r, v): return r["m"][v]["ate"] / r["m"]["motionless"]["ate"]
    for o in ("A", "B", "AB"):
        rr = [r for r in res if o == "AB" or r["order"] == o]
        print("\n==== order %s (%d missions)" % (o, len(rr)))
        for plat, sel in (("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room"))):
            g = [r for r in rr if sel(r["name"])]
            print("  %s: " % plat + "  ".join("k2=%g C %.3f S %.3f" % (k2, np.mean([r["m"]["k2=%g|C" % k2]["ate"] for r in g]), np.mean([r["m"]["k2=%g|S" % k2]["ate"] for r in g])) for k2 in K2S))
            print("        spline at k2 0.5: " + "  ".join("%s %.3f" % (v.split(" ", 1)[1], np.mean([r["m"][v]["ate"] for r in g])) for v in ("k2=0.5|S lam0.3", "k2=0.5|S lam30", "k2=0.5|S knot10", "k2=0.5|S knot40")))
            print("        governor (C/S): " + "  ".join("dec%d %.3f/%.3f (wp %.3f)" % (d, np.mean([r["m"]["k2=0.5|C dec%d" % d]["ate"] for r in g]), np.mean([r["m"]["k2=0.5|S dec%d" % d]["ate"] for r in g]), np.mean([r["m"]["k2=0.5|C dec%d" % d]["wp"] for r in g])) for d in (2, 4, 8)))
        for kind in ("C", "S"):
            sel = []
            for h in G:
                tr = [r for r in rr if grp(r["name"]) != h]; best = min(K2S, key=lambda k2: np.mean([nate(r, "k2=%g|%s" % (k2, kind)) for r in tr])); te = [r for r in rr if grp(r["name"]) == h]
                sel.append("%s->k2 %g (ATE %.3f; k2=0.5 %.3f; k2=1 %.3f)" % (h, best, np.mean([r["m"]["k2=%g|%s" % (best, kind)]["ate"] for r in te]), np.mean([r["m"]["k2=0.5|%s" % kind]["ate"] for r in te]), np.mean([r["m"]["k2=1|%s" % kind]["ate"] for r in te])))
            print("  grouped choice of k2 (%s): " % kind + "; ".join(sel))
    ratios = {plat: [r["info"]["rotor_radius"] / max(r["info"]["hand_radius"], 1e-6) for r in res if r["rotor_true"] == (plat == "EuRoC")] for plat in ("EuRoC", "TUM")}
    print("\nguard ratio rotor/hand radius: EuRoC max %.2f (99th pct %.2f) | TUM min %.2f (1st pct %.2f)" % (max(ratios["EuRoC"]), np.percentile(ratios["EuRoC"], 99), min(ratios["TUM"]), np.percentile(ratios["TUM"], 1)))
