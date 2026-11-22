#!/usr/bin/env python3
"""E25 - Phase 3 front-end candidates, one change at a time, against v4 on the conditions where v4 loses most.
Variants (cumulative unless noted):
  v4                          Phase 2 pipeline
  +repair                     conservative despiking + gravity-magnitude plausibility
  +bridge                     OU bridge of the gyro across dropouts (v4 logic otherwise)
  +gsmooth                    gravity from the gap-aware integrated-random-walk smoother
  +bjs                        post-gap accelerometer-bias variance scaled with the gap duration
  spline(l) / linear          tilt-bias regression always on (lam2 = l), on top of all of the above, with the coarse start
  spline3_nocoarse            the same without the coarse start
Run: .venv/bin/python research/experiments/e25_frontend.py [--only cond1,cond2] [--tag name]"""
import sys, os, json, argparse, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3, fbs5 as f
from e19_stress import COND
V4FE = dict(do_repair=False, mode="v4", gsmooth=False, bridge=False, coarse=False)
VARS = {"+repair": dict(fe=dict(V4FE, do_repair=True)), "+bridge": dict(fe=dict(V4FE, do_repair=True, bridge=True)), "+gsmooth": dict(fe=dict(V4FE, do_repair=True, bridge=True, gsmooth=True)),
        "+bjs": dict(fe=dict(V4FE, do_repair=True, bridge=True, gsmooth=True), bias_jump_s=True),
        "linear": dict(fe=dict(mode="linear"), bias_jump_s=True), "spline0.3": dict(fe=dict(mode="spline", lam2=0.3), bias_jump_s=True), "spline3": dict(fe=dict(mode="spline", lam2=3.0), bias_jump_s=True),
        "spline30": dict(fe=dict(mode="spline", lam2=30.0), bias_jump_s=True), "spline300": dict(fe=dict(mode="spline", lam2=300.0), bias_jump_s=True), "spline3_nocoarse": dict(fe=dict(mode="spline", lam2=3.0, coarse=False), bias_jump_s=True)}
CONDS = ["clean", "gap_3x0.25", "gap_3x2", "gap_3x4", "L1x", "L2", "L3", "gb05x", "gbrwx", "ramp_r_0.0005", "ramp_r_0.002", "ramp_r_0.01", "ramp_r_0.05", "ramp_x_0.01", "rampwin_0.05_10s_persist", "spikes", "gain_x2", "gain_x5", "acc_ramp_1", "noise_heavy"]
COND = dict(COND); COND["L1x"] = dict(lvl="L1"); COND["gb05x"] = dict(lvl="gb05"); COND["gbrwx"] = dict(lvl="gbrw")
def run(args):
    c, n, seed, only = args; kw = dict(COND[c]); kw.pop("post", None); m = p3.mission(n, kw.pop("lvl", "clean"), 1, seed, **kw); out = {}; info = {}
    e = f.estimate_v4(m.tn, m.w, m.a, n_blind=m.nb, k_drag=m.kd); out["v4"] = p3.score(m, e["p"], e["q"]); info["v4"] = dict(rotor=bool(e["rotor"]), drift=bool(e["drift_used"]))
    for v, o in VARS.items():
        if only and v not in only: continue
        try: e = f.estimate_p3(m.tn, m.w, m.a, n_blind=m.nb, k_drag=m.kd, **{k: (dict(x) if isinstance(x, dict) else x) for k, x in o.items()}); out[v] = p3.score(m, e["p"], e["q"]); info[v] = dict(rotor=bool(e["rotor"]), **{k: (float(x) if not isinstance(x, bool) else x) for k, x in e["info"].items()})
        except Exception as ex: out[v] = dict.fromkeys(p3.KEYS, float("nan")); info[v] = dict(err=repr(ex)[:200], rotor=False)
    return dict(name=n, lvl=c, seed=seed, m=out, info=info, rotor_true=m.rotor_true)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=None); ap.add_argument("--vars", default=None); ap.add_argument("--tag", default="main"); A = ap.parse_args(); names = f.available()
    conds = [c for c in CONDS if A.only is None or c in A.only.split(",")]; only = A.vars.split(",") if A.vars else None
    with Pool(10) as p: res = p.map(run, [(c, n, sd, only) for c in conds for n in names for sd in range(2)], chunksize=1)
    json.dump(res, open(os.path.join(p3.ROOT, "research/results/e25_frontend_%s.json" % A.tag), "w")); vs = ["v4"] + [v for v in VARS if v in res[0]["m"]]
    for grp, sel in (("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room"))):
        for key in ("ate", "wp", "b_abs", "tilt", "ori"):
            print("\n%s  mean %s            " % (grp, key) + " ".join("%9s" % v[:9] for v in vs))
            for c in conds:
                rr = [r for r in res if r["lvl"] == c and sel(r["name"])]; print("  %-26s" % c + " ".join("%9.2f" % np.nanmean([r["m"][v][key] for r in rr]) for v in vs))
    print("\ngate errors / coarse starts / failures per variant:")
    for v in vs: print("  %-18s gate_err %3d  coarse %3d  fail %3d  errs %s" % (v, sum(r["info"][v].get("rotor") != r["rotor_true"] for r in res if "err" not in r["info"][v]), sum(bool(r["info"][v].get("coarse")) for r in res), sum(not np.isfinite(r["m"][v]["ate"]) for r in res), sorted({r["info"][v]["err"] for r in res if "err" in r["info"][v]})[:2]))
