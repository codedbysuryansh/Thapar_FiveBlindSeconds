"""AN25b: evaluate switch rules on AN25 outputs (per case: loose and steady ATE, S_std, rr0, rh). For each rule the mean ATE of
the chosen solution relative to always-loose, cases switched, cases where the choice is worse than the better of the two by
> 10 %, and the worst choices. Usage: an25b_rules.py AN25.json [AN25.json ...]"""
import sys, json
import numpy as np
for f in sys.argv[1:]:
    R = [r for r in json.load(open(f)) if "benefit" in r and "S_std" in r and r["S_std"] == r["S_std"] and r["loose"]["rotor"]]
    lo = np.array([r["loose"]["ate"] for r in R]); st = np.array([r["steady"]["ate"] for r in R]); S = np.array([r["S_std"] for r in R]); rr0 = np.array([r["loose"]["rr0"] for r in R])
    best = np.minimum(lo, st); print("== %s: %d rotor cases; always-steady %+.2f %%, oracle %+.2f %%" % (f, len(R), 100 * (st.mean() / lo.mean() - 1), 100 * (best.mean() / lo.mean() - 1)))
    rules = [("radius >= 1.7", rr0 >= 1.7)] + [("S >= %.3f" % th, S >= th) for th in (0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.070)] \
        + [("S >= 0.045 or rr0 >= 1.7", (S >= 0.045) | (rr0 >= 1.7)), ("S >= 0.045 and rr0 >= 1.2", (S >= 0.045) & (rr0 >= 1.2))] + [("rr0>=1.7 | (S>=%.3f & rr0>=%.2f)" % (s_, r_), (rr0 >= 1.7) | ((S >= s_) & (rr0 >= r_))) for s_ in (0.045, 0.05, 0.055) for r_ in (1.25, 1.3)]
    for name, sw in rules:
        ch = np.where(sw, st, lo); bad = ch > 1.10 * best
        worst = sorted(range(len(R)), key=lambda i: -(ch[i] / best[i]))[:3]
        print("  %-26s switched %3d | mean ATE vs loose %+6.2f %% | choice > 10 %% worse than the better: %2d | worst: %s" % (name, int(sw.sum()), 100 * (ch.mean() / lo.mean() - 1), int(bad.sum()),
              "; ".join("%s %+.0f%%" % (str(R[i]["spec"])[:40].replace("'", ""), 100 * (ch[i] / best[i] - 1)) for i in worst)))
