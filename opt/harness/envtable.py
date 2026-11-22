#!/usr/bin/env python3
"""envtable.py SET ID [ID ...] - composite and ATE change per environment group (MH, V1, V2, TUM) from the per-sequence means."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import fbsopt as H   # noqa: E402
W = dict(wp=30.0, b_abs=20.0, drop=10.0)
def main():
    st = sys.argv[1]; ids = sys.argv[2:]; groups = ["MH", "V1", "V2", "TUM"]
    print("%-34s " % "candidate" + " ".join("%14s" % g for g in groups) + "   (composite %% / ATE %%)")
    for cid in ids:
        per = json.load(open(os.path.join(H.cdir(cid), "compare_%s_vs_CHAMP-1.0.0.json" % st)))["per_sequence"]; cells = []
        for g in groups:
            ss = [s for s in per if H.env_group(s) == g]
            c = sum(w * 100 * (np.mean([per[s][k][0] for s in ss]) / np.mean([per[s][k][1] for s in ss]) - 1) for k, w in W.items()) / 60
            a = 100 * (np.mean([per[s]["ate"][0] for s in ss]) / np.mean([per[s]["ate"][1] for s in ss]) - 1); cells.append("%+6.1f/%+6.1f" % (c, a))
        print("%-34s " % cid[:34] + " ".join("%14s" % c for c in cells))
if __name__ == "__main__":
    main()
