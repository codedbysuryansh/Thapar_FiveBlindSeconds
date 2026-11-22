"""AN23: detailed T4 comparison of cached candidates (opt/regression/t4cache) against the champion: per family the mean ATE /
waypoint / blind / after-dropout change, cases > 25 % and > 50 % worse, decision changes (rotor/guard, bias mode), and the
worst cases. Usage: an23_t4_compare.py OUT.txt CAND [CAND ...]"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "regression"))
import p5_t4_cands as C                                                                          # noqa: E402
def load(cid):
    out = {}
    for c in C.cases_all(fresh=os.environ.get("FRESH") == "1"):
        f = os.path.join(C.CACHE, cid, C.ckey(c) + ".json")
        if os.path.exists(f): r = json.load(open(f)); r["_case"] = c; out[C.ckey(c)] = r
    return out
base = load("CHAMP-1.0.0"); L = []
for cid in sys.argv[2:]:
    R = load(cid); L.append("## %s (%d cached cases)" % (cid, len(R)))
    for fam in ("E46", "E55", "E60", "F55", "ALL"):
        ks = [k for k in R if (fam == "ALL" or R[k]["_case"][0] == fam) and "score" in R[k] and "score" in base.get(k, {})]
        if not ks: continue
        def rel(m): return 100 * (np.nanmean([R[k]["score"][m] for k in ks]) / np.nanmean([base[k]["score"][m] for k in ks]) - 1)
        ra = np.array([R[k]["score"]["ate"] / base[k]["score"]["ate"] for k in ks])
        gd = sum((R[k]["info"].get("rotor"), R[k]["info"].get("guard")) != (base[k]["info"].get("rotor"), base[k]["info"].get("guard")) for k in ks)
        md = sum(R[k]["info"].get("mode") != base[k]["info"].get("mode") for k in ks)
        err = sum("error" in R[k] for k in R if fam == "ALL" or R[k]["_case"][0] == fam)
        L.append("  %-4s n=%-3d errors %d | ATE %+6.2f %% wp %+6.2f blind %+6.2f drop %+6.2f | ATE wins %3d losses %3d | >25 %% worse %2d, >50 %% %2d | rotor/guard changes %2d, mode changes %d"
                 % (fam, len(ks), err, rel("ate"), rel("wp"), rel("b_abs"), rel("drop"), int(np.sum(ra < 1 - 1e-9)), int(np.sum(ra > 1 + 1e-9)), int(np.sum(ra > 1.25)), int(np.sum(ra > 1.5)), gd, md))
    ks = [k for k in R if "score" in R[k] and "score" in base.get(k, {})]
    worst = sorted(ks, key=lambda k: -R[k]["score"]["ate"] / base[k]["score"]["ate"])[:8]
    L.append("  worst: " + "; ".join("%s %+.0f%%" % (str(R[k]["case"][:4]).replace("'", ""), 100 * (R[k]["score"]["ate"] / base[k]["score"]["ate"] - 1)) for k in worst))
open(sys.argv[1], "w").write("\n".join(L) + "\n"); print("\n".join(L))
