#!/usr/bin/env python3
"""E45 - OFFICIAL missions after the two dropout fixes found in E44:
  (1) gravity smoother: rotation uncertainty injected at a gap grows as k2 T^2 / 2 (was c * |w|rms * T: every 5 ms drop reset it);
  (2) back end: accelerometer-bias variance injected at a gap scaled by min(1, T / 2 s) (was the full 0.25 (m/s^2)^2 at every gap).
Variants: v4T (Phase 2 front end, fixed gate, blind tail) | v5c = v5 + joint despiker + both fixes (|C const, |S spline) |
ablations: no despiker, no OU bridge, k2 0.5 / 2, gap_ref 1 / 4 s, no bias injection, old gravity jump, forced platforms.
Configs as E41 (full seeds 42,1,2,3,4; crop60 seeds 42,7; max-level 0.5 and 1.5), both column orders."""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p4, fbs6 as f
CFGS = [("full", dict(seed=s)) for s in (42, 1, 2, 3, 4)] + [("crop60", dict(seed=s, crop=0.6)) for s in (42, 7)] + [("lvl0.5", dict(seed=42, max_level=0.5)), ("lvl1.5", dict(seed=42, max_level=1.5))]
DS = dict(z=5.0, rule="triple"); BE = dict(gap_ref=2.0)
VARS = {"v4T": (dict(fe="v4", ing=dict(held_run=4, dedup=False), rep=False), {}),
        "v5c": (dict(fe="v5", gs=True, ds=DS, be=BE, both=True), {}),
        "v5c-ds": (dict(fe="v5", gs=True, be=BE, both=True), {}),
        "v5c-bridge": (dict(fe="v5", gs=True, ds=DS, be=BE, br=False), {}),
        "v5c k2=0.5": (dict(fe="v5", gs=True, ds=DS, be=BE), dict(GS_K2=0.5)), "v5c k2=2": (dict(fe="v5", gs=True, ds=DS, be=BE), dict(GS_K2=2.0)),
        "v5c gref=1": (dict(fe="v5", gs=True, ds=DS, be=dict(gap_ref=1.0)), {}), "v5c gref=4": (dict(fe="v5", gs=True, ds=DS, be=dict(gap_ref=4.0)), {}),
        "v5c bjump=0": (dict(fe="v5", gs=True, ds=DS, be=dict(bias_jump=0.0)), {}),
        "v5c oldjump": (dict(fe="v5", gs=True, ds=DS, be=BE), dict(GS_JUMP="linear")),
        "v5c rotor": (dict(fe="v5", gs=True, ds=DS, be=dict(gap_ref=2.0, force="drag")), {}), "v5c hand": (dict(fe="v5", gs=True, ds=DS, be=dict(gap_ref=2.0, force="matern")), {})}
def radius(p): return float(np.sqrt(np.mean(np.sum((p - p.mean(0)) ** 2, 1))))
def run(args):
    n, order, cname, ckw = args; m = p4.official(n, order, **ckw); out = {"motionless": p4.score(m, np.zeros((len(m.tn), 3)), np.tile([0, 0, 0, 1.0], (len(m.tn), 1)))}; info = {}
    for v, (kw, mod) in VARS.items():
        old = {k: getattr(f.fbs5, k) for k in mod}
        for k, x in mod.items(): setattr(f.fbs5, k, x)
        try:
            e = f.estimate_off(m.tn, m.w, m.a, k_drag=m.kd, **kw); out[v] = p4.score(m, e["p"], e["q"]); info[v] = dict(rotor=e["rotor"], lock=e["lock"], pick=e["pick"], radius=radius(e["p"]), **{k: (float(x) if isinstance(x, (int, float, np.floating)) else x) for k, x in e["info"].items()})
            if kw.get("both"):
                for k in ("C", "S"):
                    if "p" + k in e: out[v + "|" + k] = p4.score(m, e["p" + k], e["q" + k])
        except Exception as ex: out[v] = dict.fromkeys(p4.KEYS, float("nan")); info[v] = dict(err=repr(ex)[:160])
        for k, x in old.items(): setattr(f.fbs5, k, x)
    return dict(name=n, order=order, cfg=cname, seed=ckw["seed"], m=out, info=info, rotor_true=m.rotor_true, gt_radius=radius(m.gp[m.ok]))
if __name__ == "__main__":
    names = f.available()
    with Pool(10) as p: res = p.map(run, [(n, o, c, kw) for o in ("A", "B") for c, kw in CFGS for n in names], chunksize=1)
    json.dump(res, open(os.path.join(p4.ROOT, "research/results/e45_official_v5c.json"), "w"))
    vs = ["motionless", "v4T", "v5c", "v5c|C", "v5c|S"] + [k for k in VARS if k not in ("v4T", "v5c")]
    p4.table(res, vs, keys=("ate", "wp", "b_abs", "b_anc", "drop", "rpe_t", "rpe_r", "ori"), by=("order", "cfg"))
    print("\nerrors:", sorted({(v, r["info"][v]["err"]) for r in res for v in r["info"] if "err" in r["info"][v]})[:6])
    for o in ("A", "B"):
        rr = [r for r in res if r["order"] == o]; v = "v5c"
        print("order %s v5c: gate errors %d/%d | picks %s | coarse %d | health min %.3f | lock rotor max %.4f, hand-held min %.4f" % (o, sum(r["info"][v]["rotor"] != r["rotor_true"] for r in rr), len(rr), {k: sum(r["info"][v]["pick"] == k for r in rr) for k in ("C", "S")}, sum(bool(r["info"][v].get("coarse")) for r in rr), min(r["info"][v]["h"] for r in rr), max(r["info"][v]["lock"] for r in rr if r["rotor_true"]), min(r["info"][v]["lock"] for r in rr if not r["rotor_true"])))
        for plat, sel in (("EuRoC", True), ("TUM", False)):
            g = [r for r in rr if r["rotor_true"] == sel]; ratio = [r["info"]["v5c rotor"]["radius"] / max(r["info"]["v5c hand"]["radius"], 1e-6) for r in g]
            print("   %s radius ratio forced-rotor / forced-hand-held: min %.2f median %.2f max %.2f | gt radius median %.2f" % (plat, min(ratio), np.median(ratio), max(ratio), np.median([r["gt_radius"] for r in g])))
