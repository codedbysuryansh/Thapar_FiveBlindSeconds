#!/usr/bin/env python3
"""loeo.py SET --family ID [ID ...] [--ref CHAMP-1.0.0] [--level env|seq] [--dataset EuRoC|TUM-VI]
Honest selection within a family of settings: for each held-out group (environment MH / V1 / V2, or each sequence with
--level seq), pick the family member with the best composite on the remaining groups of the same dataset (the champion is
always a member), then score the held-out group with that pick. Reports the picks and the pooled held-out composite / ATE,
next to the in-sample best (which over-states the gain)."""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import fbsopt as H   # noqa: E402
W = dict(wp=30.0, b_abs=20.0, drop=10.0)
def comp(per, seqs):
    """Composite % (cand vs ref) over sequences seqs; per[s][k] = (cand, ref)."""
    tot = 0.0
    for k, w in W.items():
        a = np.mean([per[s][k][0] for s in seqs]); b = np.mean([per[s][k][1] for s in seqs]); tot += w * 100 * (a - b) / b
    return tot / sum(W.values())
def metric(per, seqs, k):
    a = np.mean([per[s][k][0] for s in seqs]); b = np.mean([per[s][k][1] for s in seqs]); return 100 * (a - b) / b
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("set"); ap.add_argument("--family", nargs="+", required=True); ap.add_argument("--ref", default="CHAMP-1.0.0")
    ap.add_argument("--level", default="env"); ap.add_argument("--dataset", default="EuRoC"); a = ap.parse_args()
    per = {}
    for cid in a.family:
        rep = json.load(open(os.path.join(H.cdir(cid), "compare_%s_vs_%s.json" % (a.set, a.ref)))); per[cid] = rep["per_sequence"]
    any_ = per[a.family[0]]; seqs = [s for s in H.SEQS if s in any_ and H.dataset(s) == a.dataset]
    zero = {s: {k: (any_[s][k][1], any_[s][k][1]) for k in ("ate", "wp", "b_abs", "drop")} for s in seqs}; per[a.ref] = zero
    members = [a.ref] + list(a.family)
    groups = sorted({H.env_group(s) for s in seqs}) if a.level == "env" else seqs
    gof = (lambda s: H.env_group(s)) if a.level == "env" else (lambda s: s)
    held = {}; picks = {}
    for g in groups:
        train = [s for s in seqs if gof(s) != g]; test = [s for s in seqs if gof(s) == g]
        best = min(members, key=lambda m: comp(per[m], train)); picks[g] = best
        for s in test: held[s] = per[best][s]
    ins = min(members, key=lambda m: comp(per[m], seqs))
    print("%s, %s level, %d members (+ reference)" % (a.dataset, a.level, len(a.family)))
    for g in groups: print("  held out %-16s pick %s" % (g, picks[g]))
    print("  LOEO pooled: composite %+.2f %%, ATE %+.2f %%, wp %+.2f %%, blind %+.2f %%, drop %+.2f %%" % (comp(held, seqs), metric(held, seqs, "ate"), metric(held, seqs, "wp"), metric(held, seqs, "b_abs"), metric(held, seqs, "drop")))
    print("  in-sample best %s: composite %+.2f %%, ATE %+.2f %%" % (ins, comp(per[ins], seqs), metric(per[ins], seqs, "ate")))
    print("  per held-out sequence ATE change: " + ", ".join("%s %+.1f" % (s[:5], 100 * (held[s]["ate"][0] - held[s]["ate"][1]) / held[s]["ate"][1]) for s in seqs))
if __name__ == "__main__":
    main()
