"""AN10b: per-sequence means (20 missions each: 10 seeds x 2 orders) of two T3 runs, with the relative change, to list where a
candidate loses (failure cases). Usage: an10b_t3_per_sequence.py REF.json CAND.json OUT.txt"""
import sys, json
import numpy as np

def load(p):
    d = json.load(open(p)); d = d["missions"] if isinstance(d, dict) else d
    return {(m["name"], m["order"], int(m["seed"])): m for m in d}

ref, cand = load(sys.argv[1]), load(sys.argv[2]); keys = sorted(ref)
L = ["sequence              n   " + "  ".join("%-22s" % m for m in ("ate", "wp", "drop", "b_abs", "b_anc")) + "  mode(cand)"]
for s in sorted(set(k[0] for k in keys), key=lambda x: (x.startswith("room"), x)):
    ks = [k for k in keys if k[0] == s]; row = "%-20s %3d   " % (s, len(ks))
    for mt in ("ate", "wp", "drop", "b_abs", "b_anc"):
        r = np.mean([ref[k]["score"][mt] for k in ks]); c = np.mean([cand[k]["score"][mt] for k in ks])
        row += "%6.3f->%6.3f %+6.1f%%  " % (r, c, (c / r - 1) * 100)
    modes = {}
    for k in ks: modes[cand[k].get("mode", "?")] = modes.get(cand[k].get("mode", "?"), 0) + 1
    L.append(row + " " + ",".join("%s:%d" % kv for kv in sorted(modes.items())))
open(sys.argv[3], "w").write("\n".join(L) + "\n"); print("\n".join(L))
