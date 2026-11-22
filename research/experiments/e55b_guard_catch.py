#!/usr/bin/env python3
"""E55b - does the guard catch a hand-held recording that the gate misses? Every TUM window of E55 is forced through the ROTOR branch
(gate=1.0) with the guard ACTIVE (ratio 12); also records the radius ratio for a margin analysis. Same for EuRoC (to see how close
the guard comes to firing on true rotor data)."""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p4, fbs6 as f
from e55_gate_windows import WIN
def one(args):
    n, o, wn, a, b = args; m = p4.official(n, o, seed=42, crop=b, start=a); F = f.front_final(m.tn, m.w, m.a, dict(gate=1.0))
    r = f.back_final(F, k_drag=m.kd, want=True); c = F["cfg"]; kd = m.kd
    eh = f.fbs4.estimate_v6(None, None, None, n_blind=F["nt"], k_drag=kd, P=F["P"], force="matern", dec=c["dec_hand"], gap_ref=c["gap_ref"])
    er = f.fbs4.estimate_v6(None, None, None, n_blind=F["nt"], k_drag=kd, P=F["P"], force="drag", dec=c["dec_rotor"], gap_ref=c["gap_ref"])
    ratio = f.radius(er["p"]) / max(f.radius(eh["p"]), 1e-6); sc = p4.score(m, r["p"], None)
    return dict(name=n, order=o, win=wn, ratio=float(ratio), guard=bool(r["guard"]), ate=sc["ate"], lock=float(F["lock"]))
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    names = f.available(); jobs = [(n, o, wn, a, b) for o in ("A", "B") for wn, a, b in WIN for n in names]
    with Pool(10) as p: res = p.map(one, jobs, chunksize=1)
    json.dump(res, open(os.path.join(p4.ROOT, "research/results/e55b_guard_catch.json"), "w"))
    for g, sel in (("EuRoC", lambda r: not r["name"].startswith("room")), ("TUM", lambda r: r["name"].startswith("room"))):
        rr = [r for r in res if sel(r)]; ra = np.array([r["ratio"] for r in rr])
        print("[%s forced rotor + guard] guard fired %d/%d; radius ratio min %.2f median %.2f max %.2f; ATE mean %.3f" % (g, sum(r["guard"] for r in rr), len(rr), ra.min(), np.median(ra), ra.max(), np.mean([r["ate"] for r in rr])))
        for wn, a, b in WIN:
            q = [r for r in rr if r["win"] == wn]; print("   %-7s fired %2d/%2d  ratio min %.2f max %.2f  ATE %.3f" % (wn, sum(r["guard"] for r in q), len(q), min(r["ratio"] for r in q), max(r["ratio"] for r in q), np.mean([r["ate"] for r in q])))
