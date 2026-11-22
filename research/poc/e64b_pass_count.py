#!/usr/bin/env python3
"""E64b - operation counts of the frozen v5 research pipeline (front_final + back_final + blind5.final_blind) on official missions:
how many gravity-smoother passes (and columns), strapdown integrations, gyro bridges, zero-phase low-passes, coarse-bias runs and
smoother runs (rate, steps) one input costs. Feeds the Phase 4 runtime cost model (with the E50/E62/E62b/E64 kernel timings)."""
import sys, os, json, collections, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p4, fbs, fbs3, fbs4, fbs5, fbs6 as f, blind5 as B
LOG = collections.defaultdict(list); MODS = (fbs, fbs3, fbs4, fbs5, f)
def wrap(name, rec):
    orig = getattr(fbs5, name, None) or getattr(fbs, name, None) or getattr(fbs4, name)
    def w(*a, **kw):
        LOG[name].append(rec(*a, **kw)); return orig(*a, **kw)
    for m in MODS:
        if getattr(m, name, None) is orig: setattr(m, name, w)
wrap("gravity_smooth", lambda t, x, valid, *a, **kw: (len(t), x.shape[1]))
wrap("integrate_gyro", lambda t, w, *a, **kw: (len(t),))
wrap("bridge_gyro", lambda t, w, valid, *a, **kw: (len(t), int((valid[1:] & ~valid[:-1]).sum())))
wrap("lowpass", lambda x, fs, fc, *a, **kw: (len(x), 1 if np.ndim(x) == 1 else x.shape[1], fc))
wrap("coarse_bias", lambda *a, **kw: (len(a[0]),))
wrap("run_joint8", lambda t, q, af, model, valid, dec=1, **kw: (len(t), dec, int(model[0].shape[0]), (len(t) + dec - 1) // dec))
if __name__ == "__main__":
    out = {}
    for n, o, sd in (("MH_01_easy", "B", 42), ("MH_05_difficult", "A", 42), ("V1_02_medium", "B", 1), ("room2", "A", 42), ("room4", "B", 42)):
        LOG.clear(); m = p4.official(n, o, seed=sd); F = f.front_final(m.tn, m.w, m.a); r = f.back_final(F, k_drag=m.kd, blind=B.final_blind)
        g = LOG["gravity_smooth"]; lp = LOG["lowpass"]; rj = LOG["run_joint8"]
        rec = dict(rows=len(m.tn), slots=int(len(F["P"]["t"]) + F["nt"]), seconds=float(m.T), rotor=bool(r["rotor"]), mode=r["mode"], coarse=bool(F["coarse"]),
                   gravity_calls=len(g), gravity_columns=int(sum(c for _, c in g)), gravity_N=sorted({N for N, _ in g}), strapdown=len(LOG["integrate_gyro"]), bridges=len(LOG["bridge_gyro"]),
                   gaps=LOG["bridge_gyro"][0][1] if LOG["bridge_gyro"] else 0, lowpass=[(N, c, fc) for N, c, fc in lp], coarse_runs=len(LOG["coarse_bias"]),
                   smoother_runs=[dict(N=N, dec=d, states_per_axis=s, steps=k) for N, d, s, k in rj])
        out["%s_%s_%d" % (n, o, sd)] = rec; print(json.dumps(rec), flush=True)
    json.dump(out, open(os.path.join(p4.ROOT, "research/results/e64b_pass_count.json"), "w"))
