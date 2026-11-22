#!/usr/bin/env python3
"""E42 - drift-gate threshold per platform, from stored const (|C) and spline (|S) solutions of the same missions.
Rule: spline if (coarse start fired) or D > thr, else const. Reports mean metrics per order / platform / threshold, the per-
mission oracle (best of the two, an upper bound), and a grouped choice: for each environment group (MH, V1, V2, TUM) the
threshold is chosen on the OTHER groups of the same platform family and scored on the held-out group.
Usage: e42_gate_sweep.py <results json> <variant base name>"""
import sys, os, json, numpy as np
R = json.load(open(sys.argv[1])); V = sys.argv[2]; THR = [0.0, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3, 4e-3, 6e-3, 1e9]
grp = lambda n: "TUM" if n.startswith("room") else n[:2]
def pick(r, thr):
    i = r["info"][V]; return "S" if (i.get("coarse") or (i.get("D", 0) or 0) > thr) and (V + "|S") in r["m"] else "C"
def val(r, thr, k): return r["m"][V + "|" + pick(r, thr)][k]
for order in sorted({r["order"] for r in R}):
    for plat, sel in (("rotor (EuRoC)", lambda n: not n.startswith("room")), ("hand-held (TUM)", lambda n: n.startswith("room"))):
        rr = [r for r in R if r["order"] == order and sel(r["name"]) and (V + "|C") in r["m"]]
        if not rr: continue
        print("\norder %s | %s | %d missions | D quantiles (mrad/s) 10/50/90: %s | coarse fired %d" % (order, plat, len(rr), np.round(1e3 * np.nanpercentile([min(r["info"][V].get("D", 0) or 0, 1) for r in rr], [10, 50, 90]), 2), sum(bool(r["info"][V].get("coarse")) for r in rr)))
        for thr in THR:
            print("  thr %-8s" % ("inf" if thr > 1 else "%.1f" % (1e3 * thr)) + "  ".join("%s %.3f" % (k, np.nanmean([val(r, thr, k) for r in rr])) for k in ("ate", "wp", "b_abs", "drop", "ori")) + "   spline used %3d" % sum(pick(r, thr) == "S" for r in rr))
        print("  oracle   " + "  ".join("%s %.3f" % (k, np.nanmean([min(r["m"][V + "|C"]["ate"], r["m"][V + "|S"]["ate"]) if k == "ate" else r["m"][V + "|" + ("C" if r["m"][V + "|C"]["ate"] <= r["m"][V + "|S"]["ate"] else "S")][k] for r in rr])) for k in ("ate", "wp", "b_abs", "drop", "ori")))
    for fam, members in (("rotor", ("MH", "V1", "V2")),):
        out = []
        for g in members:
            tr = [r for r in R if r["order"] == order and grp(r["name"]) in members and grp(r["name"]) != g]; te = [r for r in R if r["order"] == order and grp(r["name"]) == g]
            best = min(THR, key=lambda thr: np.mean([val(r, thr, "ate") / max(r["m"]["motionless"]["ate"], 1e-9) for r in tr])); out.append((g, best, np.mean([val(r, best, "ate") for r in te]), np.mean([val(r, 2e-3, "ate") for r in te])))
        print("order %s | grouped choice (rotor): " % order + "; ".join("held-out %s -> thr %s, ATE %.3f (thr 2.0: %.3f)" % (g, "inf" if b > 1 else "%.1f" % (1e3 * b), a, d) for g, b, a, d in out))
