#!/usr/bin/env python3
"""E57 - fresh-seed confirmation of the blind rule chosen in E53 (blind5.final_blind) on the E52 protocol (full: seeds 42 and 5..13,
which E53 never used; crop60 / crop80 seeds 42, 7; max-level 0.5 / 1.5; clean control). Variants: motionless, v5 (smoother prediction,
orientation held) and v5b (= v5 + final blind rule). Paired bootstrap over sequences for v5b - v5."""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p4, fbs6 as f, blind5 as B
from e52_final_confirm import CFGS
def run(args):
    n, o, cname, ckw = args; m = p4.official(n, o, **ckw)
    if cname == "clean":
        s = m.s; m.tn = s.tn.copy(); m.idx = np.arange(len(s.tn)); m.w = s.w.copy(); m.a = s.a.copy(); tsec = (s.tn - s.tn[0]) * 1e-9; m.t = tsec; m.T = tsec[-1]; m.blind = tsec >= m.T - 5.0
        m.w[m.blind] = 0; m.a[m.blind] = 0; m.gp = s.gt_p; m.gq = s.gt_q; m.ok = s.ok.copy(); m.gap_after = np.zeros(len(s.tn), int)
    out = {"motionless": p4.score(m, np.zeros((len(m.tn), 3)), np.tile([0, 0, 0, 1.0], (len(m.tn), 1)))}
    F = f.front_final(m.tn, m.w, m.a); e = f.back_final(F, k_drag=m.kd); out["v5"] = p4.score(m, e["p"], e["q"])
    e2 = f.back_final(F, k_drag=m.kd, blind=B.final_blind); out["v5b"] = p4.score(m, e2["p"], e2["q"])
    return dict(name=n, order=o, cfg=cname, seed=ckw["seed"], m=out, rotor=e2["rotor"], guard=e2["guard"], rotor_true=m.rotor_true)
if __name__ == "__main__":
    names = f.available()
    with Pool(10) as p: res = p.map(run, [(n, o, c, kw) for o in ("A", "B") for c, kw in CFGS for n in names], chunksize=1)
    json.dump(res, open(os.path.join(p4.ROOT, "research/results/e57_final_v5b.json"), "w"))
    for r in res: r["cfgf"] = r["cfg"] if r["cfg"] in ("full", "clean") else "other"
    p4.table(res, ["motionless", "v5", "v5b"], keys=("ate", "wp", "b_abs", "b_anc", "b_end", "drop", "rpe_t", "ori", "ori_b"), by=("order", "cfgf"))
    rng = np.random.default_rng(0)
    print("\npaired v5b - v5 per sequence (metres / degrees), 95 % bootstrap CI over sequences, wins = sequences where v5b is lower")
    for g, sel in (("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room"))):
        for cf in ("full", "other"):
            seqs = sorted({r["name"] for r in res if sel(r["name"])})
            for x in ("b_abs", "b_anc", "b_end", "ate", "ori_b"):
                d = np.array([np.mean([r["m"]["v5b"][x] - r["m"]["v5"][x] for r in res if r["name"] == s and r["cfgf"] == cf]) for s in seqs]); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
                print("  %-5s %-5s %-6s %+.4f [%+.4f, %+.4f] wins %d/%d" % (g, cf, x, d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), int((d < 0).sum()), len(d)))
    print("\ngate errors %d / %d, guard fired %d" % (sum(r["rotor"] != r["rotor_true"] for r in res), len(res), sum(r["guard"] for r in res)))
