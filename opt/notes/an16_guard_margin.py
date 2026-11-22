"""AN16: margin of the radius guard (rotor radius rr > 12 x hand-held radius rh -> hand-held solution) on the harness sets.
For each candidate and set: per rotor-gate mission the ratio rr / rh (rr after the drag switch where it applies), the
largest ratios, and how many missions come within a factor 1.5 / 2 of the threshold. Usage: an16_guard_margin.py CAND [CAND...]"""
import sys, os, json
import numpy as np
OPT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for cid in sys.argv[1:]:
    for st in ("DEV", "DEV2", "CONF", "STRESS"):
        p = os.path.join(OPT, "candidates", cid, "results_%s.json" % st)
        if not os.path.exists(p): continue
        ms = json.load(open(p))["missions"]; R = []
        for m in ms:
            i = m.get("info", {})
            if not i.get("rotor_gate") or "rh" not in i: continue
            R.append((i["rr"] / max(i["rh"], 1e-6), i["rr0"] / max(i["rh"], 1e-6), m["tag"], i["rh"], i["rr0"], i["rr"], bool(i.get("tight")), bool(i.get("guard"))))
        if not R: print("%-30s %-6s no radius records" % (cid, st)); continue
        R.sort(reverse=True); r = np.array([x[0] for x in R])
        print("%-30s %-6s rotor missions %3d | ratio rr/rh max %.2f, median %.2f | >= 6: %d, >= 8: %d, > 12 (guard): %d | top: %s" % (
            cid, st, len(R), r.max(), np.median(r), int(np.sum(r >= 6)), int(np.sum(r >= 8)), int(np.sum(r > 12)),
            "; ".join("%s %.1f (rh %.2f rr0 %.2f rr %.2f%s)" % (x[2], x[0], x[3], x[4], x[5], " steady" if x[6] else "") for x in R[:4])))
