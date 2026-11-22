#!/usr/bin/env python3
"""E52 - confirmation of the FROZEN v5 (fbs6.estimate_final) on fresh seeds, both column orders, plus the clean control.
Official configs: full sequences with seeds 42 and 5..13 (seeds 1-4 were used for tuning in E41-E47), crop60 / crop80 (seeds 42, 7),
max-level 0.5 / 1.5 (seed 42). Clean control: the undegraded sequence with its last 5 s set to zero rows (official structure only).
Variants: motionless (zero position), v4raw (Phase 2 pipeline as is), v4T (Phase 2 front end + fixed gate + blind tail), v5 (frozen).
Drag coefficient: leave-one-sequence-out for every variant (honest); the submission uses the all-data value. Reports per-group means,
per-sequence means and percentile-bootstrap 95 % CIs over sequences for the ATE reduction vs motionless and vs v4T."""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p4, fbs6 as f
CFGS = [("full", dict(seed=s)) for s in [42] + list(range(5, 14))] + [("crop60", dict(seed=s, crop=0.6)) for s in (42, 7)] + [("crop80", dict(seed=s, crop=0.8)) for s in (42, 7)] + [("lvl0.5", dict(seed=42, max_level=0.5)), ("lvl1.5", dict(seed=42, max_level=1.5)), ("clean", dict(seed=42, max_level=0.0))]
def run(args):
    n, o, cname, ckw = args; m = p4.official(n, o, **ckw)
    if cname == "clean":   # undegraded values, official structure: all rows kept, last 5 s zero
        s = m.s; m.tn = s.tn.copy(); m.idx = np.arange(len(s.tn)); m.w = s.w.copy(); m.a = s.a.copy(); tsec = (s.tn - s.tn[0]) * 1e-9; m.t = tsec; m.T = tsec[-1]; m.blind = tsec >= m.T - 5.0
        m.w[m.blind] = 0; m.a[m.blind] = 0; m.gp = s.gt_p; m.gq = s.gt_q; m.ok = s.ok.copy(); m.gap_after = np.zeros(len(s.tn), int)
    out = {"motionless": p4.score(m, np.zeros((len(m.tn), 3)), np.tile([0, 0, 0, 1.0], (len(m.tn), 1)))}; info = {}
    try: e = f.estimate_v4(m.tn, m.w, m.a, n_blind=0, k_drag=m.kd); out["v4raw"] = p4.score(m, e["p"], e["q"])
    except Exception as ex: out["v4raw"] = dict.fromkeys(p4.KEYS, float("nan")); info["v4raw_err"] = repr(ex)[:120]
    e = f.estimate_off(m.tn, m.w, m.a, k_drag=m.kd, fe="v4", ing=dict(held_run=4, dedup=False), rep=False); out["v4T"] = p4.score(m, e["p"], e["q"])
    e = f.estimate_final(m.tn, m.w, m.a, k_drag=m.kd); out["v5"] = p4.score(m, e["p"], e["q"]); info.update({k: e[k] for k in ("rotor", "guard", "mode", "D", "h", "coarse", "lock", "n_tail", "implausible")})
    return dict(name=n, order=o, cfg=cname, seed=ckw["seed"], m=out, info={k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in info.items()}, rotor_true=m.rotor_true)
def boot(x, B=20000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x); idx = rng.integers(0, len(x), (B, len(x))); return np.percentile(x[idx].mean(1), [2.5, 97.5])
if __name__ == "__main__":
    names = f.available()
    with Pool(10) as p: res = p.map(run, [(n, o, c, kw) for o in ("A", "B") for c, kw in CFGS for n in names], chunksize=1)
    json.dump(res, open(os.path.join(p4.ROOT, "research/results/e52_final_confirm.json"), "w"))
    for r in res: r["cfgf"] = r["cfg"] if r["cfg"] in ("full", "clean") else "other"
    p4.table(res, ["motionless", "v4raw", "v4T", "v5"], keys=("ate", "wp", "b_abs", "b_anc", "drop", "rpe_t", "rpe_r", "ori"), by=("order", "cfgf"))
    print("\nper-sequence skill (full, 10 seeds, both orders): mean over missions; aggregate reduction = 1 - mean(ATE)/mean(ATE_ref); CI = bootstrap over sequences")
    for grp, sel in (("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room")), ("ALL", lambda n: True)):
        seqs = sorted({r["name"] for r in res if sel(r["name"])}); A = {v: np.array([np.mean([r["m"][v]["ate"] for r in res if r["name"] == s and r["cfg"] == "full"]) for s in seqs]) for v in ("motionless", "v4T", "v5")}
        for ref in ("motionless", "v4T"):
            sk = 1 - A["v5"] / A[ref]; print("  %-6s v5 vs %-10s aggregate %.1f %%  per-sequence mean %.1f %% [%.1f, %.1f]  wins %d/%d" % (grp, ref, 100 * (1 - A["v5"].mean() / A[ref].mean()), 100 * sk.mean(), *(100 * boot(sk)), int((A["v5"] < A[ref]).sum()), len(seqs)))
    print("\ngate errors %d / %d, guard fired %d, spline mode %d, coarse %d, implausible windows in %d missions" % (sum(r["info"]["rotor"] != r["rotor_true"] for r in res), len(res), sum(r["info"]["guard"] for r in res), sum(r["info"]["mode"] == "spline" for r in res), sum(r["info"]["coarse"] for r in res), sum(r["info"]["implausible"] > 0 for r in res)))
    print("\nper sequence (full, both orders, 10 seeds): motionless | v4T | v5 ATE, v5 wp, v5 b_abs, v5 ori")
    for s in sorted({r["name"] for r in res}):
        rr = [r for r in res if r["name"] == s and r["cfg"] == "full"]; mm = lambda v, k: np.mean([r["m"][v][k] for r in rr])
        print("  %-15s %.3f | %.3f | %.3f  %.3f  %.3f  %5.1f" % (s, mm("motionless", "ate"), mm("v4T", "ate"), mm("v5", "ate"), mm("v5", "wp"), mm("v5", "b_abs"), mm("v5", "ori")))
