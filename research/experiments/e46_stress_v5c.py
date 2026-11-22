#!/usr/bin/env python3
"""E46 - robustness beyond the official script: OUR stress suite (E19/E25 conditions: long gaps in four encodings, gyro ramps and
random walks up to 0.05 rad/s^2, gain faults, spikes, accelerometer ramps, heavy and exponential noise, combined levels, half
sample rate, timestamp jitter), 17 sequences x 2 seeds, failure at 60 %, 5 blind seconds given as zero rows (official format).
Variants: v4 (Phase 2), v5c gated (thr 2 mrad/s), v5c|C const only, v5c|S spline always (both from the same run)."""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3, fbs6 as f
from e19_stress import COND as C19
COND = dict(C19); COND.update({"L1x": dict(lvl="L1"), "gb05x": dict(lvl="gb05"), "gbrwx": dict(lvl="gbrw")})
CONDS = ["clean", "gap_3x0.25", "gap_3x2", "gap_3x4", "gap_6x2", "gap_removed", "gap_zero", "gap_held", "L1x", "L2", "L3", "L2_removed", "L2+ramp_x_0.002", "gb05x", "gbrwx", "ramp_r_0.0005", "ramp_r_0.002",
         "ramp_r_0.01", "ramp_r_0.05", "ramp_x_0.01", "rampwin_0.05_10s_persist", "spikes", "gain_x2", "gain_x5", "acc_ramp_1", "noise_heavy", "noise_exp", "half_rate", "jitter"]
DS = dict(z=5.0, rule="triple"); BE = dict(gap_ref=2.0)
def run(args):
    c, n, seed = args; kw = dict(COND[c]); post = kw.pop("post", None); m = p3.mission(n, kw.pop("lvl", "clean"), 1, seed, **kw); nb = m.nb; tn, w, a, keep = m.tn, m.w, m.a, m.keep
    if post == "half": tn, w, a, keep = tn[::2], w[::2], a[::2], keep[::2]; nbe = nb // 2; step = 2
    else: nbe = nb; step = 1
    if post == "jitter": tn = tn + np.round(np.random.default_rng(seed).normal(size=len(tn)) * 3e5).astype(np.int64)
    m.keep = keep; dtn = int(np.median(np.diff(m.s.tn))) * step; tb = m.s.tn[m.iT] + dtn * np.arange(1, nbe + 1)
    tn2 = np.concatenate([tn, tb]); w2 = np.vstack([w, np.zeros((nbe, 3))]); a2 = np.vstack([a, np.zeros((nbe, 3))]); out = {}; info = {}
    def fin(p, q):
        if post != "half": return p, q
        nk = len(keep); pb = np.stack([np.interp(np.arange(nb), step * np.arange(nbe) + step - 1, p[nk:, j]) for j in range(3)], 1); return np.vstack([p[:nk], pb]), np.vstack([q[:nk], np.repeat(q[nk - 1:nk], nb, 0)])
    try: e = f.estimate_v4(tn, w, a, n_blind=nbe, k_drag=m.kd); out["v4"] = p3.score(m, *fin(e["p"], e["q"]))
    except Exception as ex: out["v4"] = dict.fromkeys(p3.KEYS, float("nan")); info["v4"] = repr(ex)[:120]
    try:
        e = f.estimate_off(tn2, w2, a2, k_drag=m.kd, fe="v5", gs=True, ds=DS, be=BE, both=True); out["v5c"] = p3.score(m, *fin(e["p"], e["q"]))
        for k in ("C", "S"):
            if "p" + k in e: out["v5c|" + k] = p3.score(m, *fin(e["p" + k], e["q" + k]))
        info["v5c"] = dict(rotor=e["rotor"], pick=e["pick"], **{k: (float(x) if isinstance(x, (int, float, np.floating)) else x) for k, x in e["info"].items()})
    except Exception as ex: out["v5c"] = dict.fromkeys(p3.KEYS, float("nan")); info["v5c"] = dict(err=repr(ex)[:160])
    return dict(name=n, lvl=c, seed=seed, m=out, info=info, rotor_true=m.rotor_true)
if __name__ == "__main__":
    names = f.available()
    with Pool(10) as p: res = p.map(run, [(c, n, sd) for c in CONDS for n in names for sd in range(2)], chunksize=1)
    json.dump(res, open(os.path.join(p3.ROOT, "research/results/e46_stress_v5c.json"), "w"))
    vs = ["v4", "v5c", "v5c|C", "v5c|S"]
    for grp, sel in (("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room"))):
        for key in ("ate", "wp", "b_abs", "ori"):
            print("\n%s  mean %-5s       " % (grp, key) + " ".join("%9s" % v for v in vs) + "   gate_err  spline  coarse")
            for c in CONDS:
                rr = [r for r in res if r["lvl"] == c and sel(r["name"])]
                print("  %-26s" % c + " ".join("%9.2f" % np.nanmean([r["m"][v][key] for r in rr if v in r["m"]]) for v in vs) + "   %3d %3d %3d" % (sum(r["info"]["v5c"].get("rotor") != r["rotor_true"] for r in rr if "err" not in r["info"]["v5c"]), sum(r["info"]["v5c"].get("pick") == "S" for r in rr), sum(bool(r["info"]["v5c"].get("coarse")) for r in rr)))
    print("errors:", sorted({str(r["info"]) for r in res if "err" in str(r["info"])})[:3])
