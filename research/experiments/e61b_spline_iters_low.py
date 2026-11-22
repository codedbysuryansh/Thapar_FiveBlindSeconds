#!/usr/bin/env python3
"""E61 - convergence of the rotor gyro-bias spline (attitude6: iters 6 / tol 3e-4 in v5) on official missions. E60 found that under the
order-B gyro degradation (7 EuRoC flights get ~0.18 rad/s gyro noise and 0.05-0.13 rad/s bias) the rotor estimate moves by 0.1-0.3 m
(sd) when the sensor frame is rotated, while order A / TUM-VI are invariant to <= 2 cm. More iterations reduced that spread on two
flights. Here: iters 6 / 12 / 20 (tol 3e-4, 3e-4, 1e-5) on full sequences, seeds 42 and 1, both orders, 17 sequences."""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p4, fbs5, fbs6 as f, blind5 as B
SET = {"it6": (6, 3e-4), "it2": (2, 3e-4), "it3": (3, 3e-4), "it4": (4, 3e-4)}
ORIG = fbs5.attitude6
def one(args):
    n, o, sd = args; m = p4.official(n, o, seed=sd); out = {}
    for k, (it, tol) in SET.items():
        def a6(*a, **kw):
            if kw.get("iters", 6) == 6: kw["iters"] = it
            kw["tol"] = tol; return ORIG(*a, **kw)
        f.attitude6 = a6
        F = f.front_final(m.tn, m.w, m.a); r = f.back_final(F, k_drag=m.kd, blind=B.final_blind); out[k] = p4.score(m, r["p"], r["q"])
    f.attitude6 = ORIG; return dict(name=n, order=o, seed=sd, m=out, rotor=bool(F["rotor"]))
if __name__ == "__main__":
    names = f.available()
    with Pool(10) as p: res = p.map(one, [(n, o, sd) for o in ("A", "B") for sd in (42, 1) for n in names], chunksize=1)
    json.dump(res, open(os.path.join(p4.ROOT, "research/results/e61b_spline_iters_low.json"), "w")); rng = np.random.default_rng(0)
    for o in ("A", "B"):
        for g, sel in (("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room"))):
            rr = [r for r in res if r["order"] == o and sel(r["name"])]; seqs = sorted({r["name"] for r in rr})
            line = "[%s %-5s] " % (o, g) + "  ".join("%s ATE %.3f wp %.3f ori %.1f" % (k, np.mean([r["m"][k]["ate"] for r in rr]), np.mean([r["m"][k]["wp"] for r in rr]), np.mean([r["m"][k]["ori"] for r in rr])) for k in SET)
            for k in ("it2", "it3", "it4"):
                d = np.array([np.mean([r["m"][k]["ate"] - r["m"]["it6"]["ate"] for r in rr if r["name"] == s]) for s in seqs]); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
                line += " | %s-it6 %+.3f [%+.3f, %+.3f] better %d/%d" % (k, d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), int((d < 0).sum()), len(d))
            print(line)
