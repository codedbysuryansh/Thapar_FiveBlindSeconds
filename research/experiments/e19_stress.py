#!/usr/bin/env python3
"""E19 - where does the candidate break?  Severe and unfamiliar faults, beyond the Phase 2 levels.
Fault menu (OUR simulator; magnitudes bracket the public ILLIXR/OpenVINS fault benchmark, arXiv 2602.14413, whose IMU faults are:
gyro ramp 0.01-0.1 rad/s^2 on one axis, gain x2/x5/x10 on both sensors in a window, removed samples for 0.25-4 s):
  gyro ramps (sensor x = the gravity axis on EuRoC, and a random axis), windowed ramps, gain windows, spikes, accel ramps,
  heavy noise, dropout length / count / encoding, combined levels, half sample rate, timestamp jitter.
Each condition: 17 sequences x 2 seeds, failure at 60 %, 5 blind seconds. Variant chosen with --est (default v4).
Run: .venv/bin/python research/experiments/e19_stress.py [--est v4] [--tag name]"""
import sys, os, json, argparse, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p3, fbs4 as f
def ramp(rate, axis):
    def fn(t, w, a, rng):
        d = np.eye(3)[axis] if axis is not None else (lambda x: x / np.linalg.norm(x))(rng.normal(size=3)); return w + rate * t[:, None] * d, a
    return fn
def ramp_win(rate, dur, persist):
    def fn(t, w, a, rng):
        ts = rng.uniform(0.3, 0.6) * t[-1]; b = rate * np.clip(t - ts, 0, dur); b = b if persist else np.where(t < ts + dur, b, 0.0); return w + b[:, None] * np.eye(3)[0], a
    return fn
def windows(t, rng, n, dur):
    out = []
    for _ in range(n):
        for _ in range(50):
            t0 = rng.uniform(0.2, 0.85) * t[-1]
            if all(abs(t0 - x) > dur + 6 for x in out): break
        out.append(t0)
    return out
def gain(fac, dur=2.5, n=3):
    def fn(t, w, a, rng):
        w, a = w.copy(), a.copy()
        for t0 in windows(t, rng, n, dur): m = (t >= t0) & (t < t0 + dur); w[m] *= fac; a[m] *= fac
        return w, a
    return fn
def spikes(frac=0.005):
    def fn(t, w, a, rng):
        m = rng.random(len(t)) < frac; w = w.copy(); a = a.copy(); w[m] += rng.normal(size=(m.sum(), 3)) * 5.0; a[m] += rng.normal(size=(m.sum(), 3)) * 50.0; return w, a
    return fn
G3 = [2.0] * 3
COND = {"clean": {}, "ramp_x_0.0005": dict(fault=ramp(0.0005, 0)), "ramp_x_0.002": dict(fault=ramp(0.002, 0)), "ramp_x_0.01": dict(fault=ramp(0.01, 0)), "ramp_x_0.05": dict(fault=ramp(0.05, 0)),
        "ramp_r_0.0005": dict(fault=ramp(0.0005, None)), "ramp_r_0.002": dict(fault=ramp(0.002, None)), "ramp_r_0.01": dict(fault=ramp(0.01, None)), "ramp_r_0.05": dict(fault=ramp(0.05, None)),
        "rampwin_0.05_10s": dict(fault=ramp_win(0.05, 10.0, False)), "rampwin_0.05_10s_persist": dict(fault=ramp_win(0.05, 10.0, True)),
        "gain_x2": dict(fault=gain(2.0)), "gain_x5": dict(fault=gain(5.0)), "gain_x10": dict(fault=gain(10.0)), "spikes": dict(fault=spikes()),
        "acc_ramp_1": dict(cfg=dict(ab=1.0)), "acc_ramp_3": dict(cfg=dict(ab=3.0)), "noise_heavy": dict(cfg=dict(gn=0.2, an=2.0)), "noise_exp": dict(cfg=dict(gn=0.2, an=2.0, shape="exp")),
        "gap_3x0.25": dict(cfg=dict(gaps=[0.25] * 3)), "gap_3x1": dict(cfg=dict(gaps=[1.0] * 3)), "gap_3x2": dict(cfg=dict(gaps=G3)), "gap_3x4": dict(cfg=dict(gaps=[4.0] * 3)), "gap_6x2": dict(cfg=dict(gaps=[2.0] * 6)),
        "gap_removed": dict(cfg=dict(gaps=G3), enc="removed"), "gap_zero": dict(cfg=dict(gaps=G3), enc="zero"), "gap_held": dict(cfg=dict(gaps=G3), enc="held"),
        "L2": dict(lvl="L2"), "L3": dict(lvl="L3"), "L2+ramp_x_0.002": dict(lvl="L2", fault=ramp(0.002, 0)), "L2_removed": dict(lvl="L2", enc="removed"),
        "half_rate": dict(post="half"), "jitter": dict(post="jitter")}
def get_est(name):
    if name == "v4": return lambda m, nb: f.estimate_v4(m.tn, m.w, m.a, n_blind=nb, k_drag=m.kd)
    import importlib; mod, fn = name.split(":"); return getattr(importlib.import_module(mod), fn)
def run(args):
    c, n, seed, estname = args; kw = dict(COND[c]); post = kw.pop("post", None); m = p3.mission(n, kw.pop("lvl", "clean"), 1, seed, **kw); nb = m.nb; est = get_est(estname)
    try:
        if post == "half":
            keep0 = m.keep; m.tn, m.w, m.a, m.keep = m.tn[::2], m.w[::2], m.a[::2], m.keep[::2]; e = est(m, nb // 2); nk = len(m.keep)
            pb = np.stack([np.interp(np.arange(nb), 2 * np.arange(nb // 2) + 1, e["p"][nk:, j]) for j in range(3)], 1); p = np.vstack([e["p"][:nk], pb]); q = np.vstack([e["q"][:nk], np.repeat(e["q"][nk - 1:nk], nb, 0)])
        else:
            if post == "jitter": m.tn = m.tn + np.round(np.random.default_rng(seed).normal(size=len(m.tn)) * 3e5).astype(np.int64)
            e = est(m, nb); p, q = e["p"], e["q"]
        sc = p3.score(m, p, q); info = dict(rotor=bool(e["rotor"]), lock=float(e["lock"]), D=float(e.get("drift_D", np.nan)), drift=bool(e.get("drift_used", False)), err="")
    except Exception as ex:
        sc = dict.fromkeys(p3.KEYS, float("nan")); info = dict(rotor=False, lock=float("nan"), D=float("nan"), drift=False, err=repr(ex)[:200])
    return dict(name=n, lvl=c, seed=seed, m={estname: sc, "motionless": p3.score(m, np.zeros((len(m.keep) + nb, 3)))}, info=info, rotor_true=m.rotor_true)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--est", default="v4"); ap.add_argument("--tag", default=None); ap.add_argument("--only", default=None); A = ap.parse_args(); names = f.available()
    conds = [c for c in COND if A.only is None or any(c.startswith(x) for x in A.only.split(","))]
    with Pool(10) as p: res = p.map(run, [(c, n, sd, A.est) for c in conds for n in names for sd in range(2)], chunksize=1)
    tag = A.tag or A.est.replace(":", "_"); json.dump(res, open(os.path.join(p3.ROOT, "research/results/e19_stress_%s.json" % tag), "w"))
    print("%-26s | EuRoC: wp  b_abs b_anc gap/rec ate (motionless ate) tilt ori | TUM: wp b_abs b_anc gap/rec ate (motionless) tilt ori | gate errors, drift on, failures" % "condition")
    for c in conds:
        rr = [r for r in res if r["lvl"] == c]; line = "%-26s |" % c
        for sel in (lambda n: not n.startswith("room"), lambda n: n.startswith("room")):
            g = [r for r in rr if sel(r["name"])]; mm = lambda k, v=A.est: np.nanmean([r["m"][v][k] for r in g])
            line += " %5.2f %5.2f %5.2f %5.2f/%5.2f %5.2f (%4.2f) %5.1f %5.1f |" % (mm("wp"), mm("b_abs"), mm("b_anc"), mm("gap"), mm("rec"), mm("ate"), mm("ate", "motionless"), mm("tilt"), mm("ori"))
        line += " gate_err %d  drift %d  fail %d" % (sum(r["info"]["rotor"] != r["rotor_true"] for r in rr if not r["info"]["err"]), sum(r["info"]["drift"] for r in rr), sum(bool(r["info"]["err"]) or not np.isfinite(r["m"][A.est]["ate"]) for r in rr)); print(line)
    errs = sorted({r["info"]["err"] for r in res if r["info"]["err"]})
    if errs: print("errors:", errs[:5])
