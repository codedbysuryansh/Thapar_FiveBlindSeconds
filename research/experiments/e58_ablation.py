#!/usr/bin/env python3
"""E58 - leave-one-component-out ablation of the FROZEN v5 (fbs6.front_final/back_final + blind5.final_blind) on official missions:
full sequences, seeds 42 and 1, column orders A and B (68 missions). Each variant switches exactly one component off or back to its
Phase 2 form. Paired differences vs v5 with a bootstrap over sequences."""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p4, fbs6 as f, blind5 as B
VAR = {"v5": {}, "no_dedup": dict(dedup=False), "no_repair": dict(rep=False), "no_gravity_smoother": dict(gsm=False), "linear_jump_law": dict(jump="linear"),
       "k2_1.0": dict(k2=1.0), "no_gyro_bridge": dict(br=False), "rotor_const_bias": dict(rotor_spline=False), "hand_always_spline": dict(thr_hand=-1.0),
       "no_gap_scaled_bias": dict(gap_ref=None), "no_zupt": dict(zupt=False), "no_lever": dict(lever=False), "no_drag": dict(drag=False),
       "no_guard": dict(guard=1e9), "dec1_both": dict(dec_rotor=1, dec_hand=1), "all_hand_held": dict(gate=-1.0), "all_rotor_no_guard": dict(gate=1.0, guard=1e9),
       "all_rotor_guard": dict(gate=1.0)}
def one(args):
    n, o, sd = args; m = p4.official(n, o, seed=sd); out = {}
    for k, cfg in VAR.items():
        F = f.front_final(m.tn, m.w, m.a, cfg); r = f.back_final(F, k_drag=m.kd, blind=B.final_blind); out[k] = p4.score(m, r["p"], r["q"]); out[k]["rotor"] = r["rotor"]
    return dict(name=n, order=o, seed=sd, m=out)
if __name__ == "__main__":
    names = f.available(); jobs = [(n, o, sd) for o in ("A", "B") for sd in (42, 1) for n in names]
    with Pool(10) as p: res = p.map(one, jobs, chunksize=1)
    json.dump(res, open(os.path.join(p4.ROOT, "research/results/e58_ablation.json"), "w")); rng = np.random.default_rng(0)
    for g, sel in (("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room"))):
        rr = [r for r in res if sel(r["name"])]; seqs = sorted({r["name"] for r in rr}); print("\n[%s | %d missions]  mean ATE / wp / b_abs / drop / ori  | paired dATE vs v5 [95%% CI over sequences] wins(v5 better)" % (g, len(rr)))
        for k in VAR:
            mm = lambda x: np.nanmean([r["m"][k][x] for r in rr]); d = np.array([np.mean([r["m"][k]["ate"] - r["m"]["v5"]["ate"] for r in rr if r["name"] == s]) for s in seqs])
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
            print("  %-20s %.3f / %.3f / %.3f / %.3f / %5.1f | %+.3f [%+.3f, %+.3f] %d/%d" % (k, mm("ate"), mm("wp"), mm("b_abs"), mm("drop"), mm("ori"), d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), int((d > 0).sum()), len(d)))
