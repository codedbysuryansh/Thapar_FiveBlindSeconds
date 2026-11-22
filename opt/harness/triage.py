#!/usr/bin/env python3
"""triage.py SET ID [ID ...] [--ref CHAMP-1.0.0] - screening decision from a DEV-type comparison (never a promotion):
  composite 95 % interval entirely above 0 (pooled or either dataset)  -> rejected (significantly worse)
  pooled interval entirely below 0 and no dataset significantly worse -> pending (improvement on the screen; needs DEV2 + CONF)
  otherwise                                                           -> rejected (no significant effect; champion value kept)
Candidates whose status is already 'validation', 'accepted' or 'promoted' are left alone."""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import fbsopt as H   # noqa: E402
def fmt(G, gn): c = G[gn]["composite"]; return "%s %+.2f %% [%+.2f, %+.2f]" % (gn, c["rel"], c["ci"][0], c["ci"][1])
def decide(a, reg):
    for cid in a.ids:
        e = reg["candidates"][cid]
        if e.get("status") in ("validation", "accepted", "promoted"): continue
        rep = json.load(open(os.path.join(H.cdir(cid), "compare_%s_vs_%s.json" % (a.set, a.ref)))); G = rep["groups"]; txt = "; ".join(fmt(G, g) for g in ("ALL", "EuRoC", "TUM-VI"))
        worse = [g for g in ("ALL", "EuRoC", "TUM-VI") if G[g]["composite"]["ci"][0] > 0]
        if rep["new_failures"]: st, why = "rejected", "new failures on %s: %s" % (a.set, rep["new_failures"][:5])
        elif worse: st, why = "rejected", "significantly worse on %s (%s): %s" % (a.set, ",".join(worse), txt)
        elif G["ALL"]["composite"]["ci"][1] < 0: st, why = "pending", "improvement on the %s screen, to be replicated (DEV2) and confirmed (CONF): %s" % (a.set, txt)
        else: st, why = "rejected", "no significant effect on %s, champion value kept: %s" % (a.set, txt)
        e["status"] = st; e["reason"] = why; print("%-34s %-9s %s" % (cid, st, why))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("set"); ap.add_argument("ids", nargs="+"); ap.add_argument("--ref", default="CHAMP-1.0.0"); a = ap.parse_args()
    with H.reg_txn() as reg: decide(a, reg)
if __name__ == "__main__":
    main()
