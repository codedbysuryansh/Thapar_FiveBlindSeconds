#!/usr/bin/env python3
"""E55 - platform gate and guard on SHORT / OFFSET windows of the official degradation (E53 found 5/22 EuRoC crop40 missions run as
hand-held: MH_02 by the gate, MH_01 / MH_04 by the guard). Windows: prefixes, suffixes and middles of every sequence, orders A/B,
seed 42. Variants: frozen v5 (gate + guard), forced rotor without guard, forced hand-held. Reports lock values, decisions and the
ATE cost of every decision."""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p4, fbs6 as f
WIN = [("p15", 0.0, 0.15), ("p25", 0.0, 0.25), ("p40", 0.0, 0.40), ("p60", 0.0, 0.60), ("s30", 0.30, 1.0), ("s60", 0.60, 1.0), ("m20_50", 0.20, 0.50), ("m50_80", 0.50, 0.80), ("full", 0.0, 1.0)]
def one(args):
    n, o, wn, a, b = args; m = p4.official(n, o, seed=42, crop=b, start=a); out = dict(name=n, order=o, win=wn, dur=float(m.T))
    for tag, cfg in (("v5", None), ("rotor", dict(gate=1.0, guard=1e9)), ("hand", dict(gate=-1.0))):
        F = f.front_final(m.tn, m.w, m.a, cfg); r = f.back_final(F, k_drag=m.kd); sc = p4.score(m, r["p"], None)
        out[tag] = dict(ate=sc["ate"], pre=sc["pre_ate"], b_abs=sc["b_abs"], rotor=r["rotor"], guard=r["guard"], mode=r["mode"], lock=float(F["lock"]))
    return out
if __name__ == "__main__":
    names = f.available(); jobs = [(n, o, wn, a, b) for o in ("A", "B") for wn, a, b in WIN for n in names]
    with Pool(10) as p: res = p.map(one, jobs, chunksize=1)
    json.dump(res, open(os.path.join(p4.ROOT, "research/results/e55_gate_windows.json"), "w"))
    for g, sel in (("EuRoC", lambda r: not r["name"].startswith("room")), ("TUM", lambda r: r["name"].startswith("room"))):
        print("\n[%s]  window: duration | lock min/median/max | v5 rotor-decisions (guard fired) | ATE v5 / forced-rotor / forced-hand | oracle-best" % g)
        for wn, a, b in WIN:
            rr = [r for r in res if sel(r) and r["win"] == wn]; lk = [r["v5"]["lock"] for r in rr]
            best = np.mean([min(r["rotor"]["ate"], r["hand"]["ate"]) for r in rr])
            print("  %-7s %6.1f s | %.3f %.3f %.3f | %2d/%2d (%d) | %.3f / %.3f / %.3f | %.3f" % (wn, np.mean([r["dur"] for r in rr]), min(lk), np.median(lk), max(lk),
                  sum(r["v5"]["rotor"] for r in rr), len(rr), sum(r["v5"]["guard"] for r in rr), np.mean([r["v5"]["ate"] for r in rr]), np.mean([r["rotor"]["ate"] for r in rr]), np.mean([r["hand"]["ate"] for r in rr]), best))
    print("\nwrong-platform decisions (v5):")
    for r in res:
        truth = not r["name"].startswith("room")
        if r["v5"]["rotor"] != truth: print("  %-16s %s %-7s dur %5.1f lock %.3f guard %s  ATE v5 %.3f rotor %.3f hand %.3f" % (r["name"], r["order"], r["win"], r["dur"], r["v5"]["lock"], r["v5"]["guard"], r["v5"]["ate"], r["rotor"]["ate"], r["hand"]["ate"]))
