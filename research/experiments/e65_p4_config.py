#!/usr/bin/env python3
"""E65 - accuracy of the Phase 4 RATE configuration chosen from the runtime cost model (E64): gravity smoother at 12.5 Hz (E63/E63b:
accuracy-neutral), rotor smoother at 25 Hz (blocks of 8; E47: +0.7 % vs full rate), hand-held smoother unchanged at 100 Hz.
Same protocol as E52/E57 (full: seed 42 + 5..13; crop60 / crop80 seeds 42, 7; max-level 0.5 / 1.5; clean control; both orders);
paired comparison with the frozen research configuration (v5: full-rate gravity, rotor 50 Hz). Blind rule = blind5.final_blind."""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p4, fbs5, fbs6 as f, blind5 as B
from e52_final_confirm import CFGS
from e63_decimated_gravity import gravity_smooth_dec, ORIG
def run(args):
    n, o, cname, ckw = args; m = p4.official(n, o, **ckw)
    if cname == "clean":
        s = m.s; m.tn = s.tn.copy(); m.idx = np.arange(len(s.tn)); m.w = s.w.copy(); m.a = s.a.copy(); tsec = (s.tn - s.tn[0]) * 1e-9; m.t = tsec; m.T = tsec[-1]; m.blind = tsec >= m.T - 5.0
        m.w[m.blind] = 0; m.a[m.blind] = 0; m.gp = s.gt_p; m.gq = s.gt_q; m.ok = s.ok.copy(); m.gap_after = np.zeros(len(s.tn), int)
    out = {"motionless": p4.score(m, np.zeros((len(m.tn), 3)), np.tile([0, 0, 0, 1.0], (len(m.tn), 1)))}
    F = f.front_final(m.tn, m.w, m.a); e = f.back_final(F, k_drag=m.kd, blind=B.final_blind); out["v5"] = p4.score(m, e["p"], e["q"]); r0 = e["rotor"]
    g = lambda *a, **kw: gravity_smooth_dec(*a, d=16, **kw); fbs5.gravity_smooth = g; f.gravity_smooth = g
    try:
        F = f.front_final(m.tn, m.w, m.a, dict(dec_rotor=8)); e = f.back_final(F, k_drag=m.kd, blind=B.final_blind); out["v5_p4"] = p4.score(m, e["p"], e["q"]); r1 = e["rotor"]
    finally: fbs5.gravity_smooth = ORIG; f.gravity_smooth = ORIG
    return dict(name=n, order=o, cfg=cname, seed=ckw["seed"], m=out, rotor=[bool(r0), bool(r1)], rotor_true=m.rotor_true)
if __name__ == "__main__":
    names = f.available()
    with Pool(10) as p: res = p.map(run, [(n, o, c, kw) for o in ("A", "B") for c, kw in CFGS for n in names], chunksize=1)
    json.dump(res, open(os.path.join(p4.ROOT, "research/results/e65_p4_config.json"), "w"))
    for r in res: r["cfgf"] = r["cfg"] if r["cfg"] in ("full", "clean") else "other"
    p4.table(res, ["motionless", "v5", "v5_p4"], keys=("ate", "wp", "b_abs", "b_anc", "drop", "rpe_t", "ori"), by=("cfgf",))
    rng = np.random.default_rng(0)
    print("\npaired v5_p4 - v5 per sequence, 95 % bootstrap CI over sequences (both orders, all seeds of the condition)")
    for g, sel in (("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room"))):
        seqs = sorted({r["name"] for r in res if sel(r["name"])})
        for cf in ("full", "other", "clean"):
            for x in ("ate", "wp", "b_abs", "b_anc", "drop"):
                d = np.array([np.mean([r["m"]["v5_p4"][x] - r["m"]["v5"][x] for r in res if r["name"] == s and r["cfgf"] == cf]) for s in seqs]); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
                rel = 100 * np.mean([np.mean([r["m"]["v5_p4"][x] for r in res if r["name"] == s and r["cfgf"] == cf]) for s in seqs]) / np.mean([np.mean([r["m"]["v5"][x] for r in res if r["name"] == s and r["cfgf"] == cf]) for s in seqs]) - 100
                print("  %-5s %-5s %-5s %+.4f m [%+.4f, %+.4f]  (%+.2f %%)  worse on %d/%d" % (g, cf, x, d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), rel, int((d > 0).sum()), len(d)))
    print("\ngate decisions identical: %d / %d; gate errors (p4) %d" % (sum(r["rotor"][0] == r["rotor"][1] for r in res), len(res), sum(r["rotor"][1] != r["rotor_true"] for r in res)))
