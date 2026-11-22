#!/usr/bin/env python3
"""E41 - candidates on OFFICIAL missions (organisers' degrade_imu.py, unmodified; both input column orders).
Configs: 'full' = whole sequence, seeds 42 (the script's own), 1, 2, 3, 4; 'crop60' = first 60 % of the sequence degraded as a
file of its own (seeds 42, 7); 'lvl0.5' / 'lvl1.5' = --max-level 0.5 / 1.5 (seed 42). Default --gamma 1.7 throughout.
Variants: v4raw = the Phase 2 pipeline exactly as it would run on these files; v4T = v4 with the trailing zero rows treated as the
blind window; +ing2 = duplicate rows share a slot, every exact repeat is invalid; +rep = gravity-magnitude plausibility windows;
v5 = gated front end; +dsT5 / +dsK4 = joint despiker; +snroll = time-varying accelerometer noise; +abias = body-frame accel bias.
Run: .venv/bin/python research/experiments/e41_official_bench.py"""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p4, fbs6 as f
CFGS = [("full", dict(seed=s)) for s in (42, 1, 2, 3, 4)] + [("crop60", dict(seed=s, crop=0.6)) for s in (42, 7)] + [("lvl0.5", dict(seed=42, max_level=0.5)), ("lvl1.5", dict(seed=42, max_level=1.5))]
DS_T = dict(z=5.0, rule="triple"); DS_K = dict(z=5.0, rule="k4")
VARS = {"v4T": dict(fe="v4", ing=dict(held_run=4, dedup=False), rep=False), "v5": dict(fe="v5", gs=True, both=True), "v5+dsT5": dict(fe="v5", gs=True, ds=DS_T, both=True),
        "v5+dsT5+snroll": dict(fe="v5", gs=True, ds=DS_T, be=dict(sn_roll=2.0), both=True), "v5+dsT5+abias": dict(fe="v5", gs=True, ds=DS_T, be=dict(abias=(0.3, 1e-6)), both=True),
        "v5+dsT5 rotor": dict(fe="v5", gs=True, ds=DS_T, be=dict(force="drag"), both=True), "v5+dsT5 hand": dict(fe="v5", gs=True, ds=DS_T, be=dict(force="matern"), both=True)}
TAG = os.environ.get("E41TAG", "e41_official_bench")
def run(args):
    n, order, cname, ckw = args; m = p4.official(n, order, **ckw); out = {"motionless": p4.score(m, np.zeros((len(m.tn), 3)), np.tile([0, 0, 0, 1.0], (len(m.tn), 1)))}; info = {}
    try: e = f.estimate_v4(m.tn, m.w, m.a, n_blind=0, k_drag=m.kd); out["v4raw"] = p4.score(m, e["p"], e["q"]); info["v4raw"] = dict(rotor=bool(e["rotor"]))
    except Exception as ex: out["v4raw"] = dict.fromkeys(p4.KEYS, float("nan")); info["v4raw"] = dict(err=repr(ex)[:160])
    for v, kw in VARS.items():
        try:
            e = f.estimate_off(m.tn, m.w, m.a, k_drag=m.kd, **kw); out[v] = p4.score(m, e["p"], e["q"]); info[v] = dict(rotor=e["rotor"], lock=e["lock"], pick=e["pick"], **{k: (float(x) if isinstance(x, (int, float, np.floating)) else x) for k, x in e["info"].items()})
            if kw.get("both"):
                for k in ("C", "S"):
                    if "p" + k in e: out[v + "|" + k] = p4.score(m, e["p" + k], e["q" + k])
        except Exception as ex: out[v] = dict.fromkeys(p4.KEYS, float("nan")); info[v] = dict(err=repr(ex)[:160])
    st = m.stats
    return dict(name=n, order=order, cfg=cname, seed=ckw["seed"], m=out, info=info, rotor_true=m.rotor_true, deg=dict(acc_unit=None, spikes=st["spikes"], stuck=st["stuck_samples"], dropped=st["dropped_total"], dups=st["duplicated_rows"], fail=st["failure_frames"], T=st["duration_s"]))
if __name__ == "__main__":
    names = f.available(); jobs = [(n, o, c, kw) for o in ("A", "B") for c, kw in CFGS for n in names]
    with Pool(10) as p: res = p.map(run, jobs, chunksize=1)
    json.dump(res, open(os.path.join(p4.ROOT, "research/results/%s.json" % TAG), "w"))
    for r in res: r["cfgf"] = r["cfg"]
    vs = ["motionless", "v4raw", "v4T"] + [k + x for k in VARS if k != "v4T" for x in ("", "|C", "|S")]
    p4.table(res, vs, by=("order", "cfgf"))
    print("\nerrors:", sorted({(v, r["info"][v]["err"]) for r in res for v in r["info"] if "err" in r["info"][v]})[:6])
    for o in ("A", "B"):
        for v in ("v4raw", "v4T", "v5", "v5+dsT5"):
            if v not in res[0]["info"]: continue
            rr = [r for r in res if r["order"] == o and "err" not in r["info"].get(v, {})]
            print("order %s %-10s gate errors %d/%d  picks %s  coarse %d  despiked rows mean %.1f" % (o, v, sum(r["info"][v].get("rotor") != r["rotor_true"] for r in rr), len(rr), {k: sum(r["info"][v].get("pick") == k for r in rr) for k in ("C", "S", "F")}, sum(bool(r["info"][v].get("coarse")) for r in rr), np.mean([r["info"][v].get("spikes", 0) for r in rr])))
