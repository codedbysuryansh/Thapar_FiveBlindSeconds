#!/usr/bin/env python3
"""E59 - classical competitor baselines on official missions under the identical protocol (full sequences, seeds 42 and 1, orders A/B).
Every baseline gets the v5 front end (our attitude, gap filling) so that it is as strong as we can make it:
  strapdown        : world-frame specific force minus its mean, integrated twice from rest (what a naive team ships)
  hp_int_<fc>      : the same with a zero-phase 2nd-order Butterworth high-pass (cut-off fc Hz) after each integration - the classic
                     'heave filter' treatment of bounded motion; fc chosen per held-out environment group (LOEO) from the grid
  motionless       : zero position, identity orientation
Blind rows: positions held at the last valid value. Orientation: the v5 attitude (held in the blind window)."""
import sys, os, json, numpy as np
from multiprocessing import Pool
from scipy.signal import butter, filtfilt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p4, fbs6 as f, blind5 as B
FC = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
def one(args):
    n, o, sd = args; m = p4.official(n, o, seed=sd); F = f.front_final(m.tn, m.w, m.a); P = F["P"]; rows, L, nt = F["rows"], F["L"], F["nt"]; fs = P["fs"]; dt = 1.0 / fs
    aw = P["aw"]; q = np.concatenate([P["q"], np.repeat(P["q"][-1:], nt, 0)]); out = {"motionless": p4.score(m, np.zeros((len(m.tn), 3)), np.tile([0, 0, 0, 1.0], (len(m.tn), 1)))}
    def emit(p):
        p = np.concatenate([p, np.repeat(p[-1:], nt, 0)]); return p[rows]
    v = np.cumsum(aw, 0) * dt; p = np.cumsum(v, 0) * dt; out["strapdown"] = p4.score(m, emit(p), q[rows])
    for fc in FC:
        b, a = butter(2, fc / (fs / 2), "high"); v = filtfilt(b, a, np.cumsum(aw, 0) * dt, axis=0); p = filtfilt(b, a, np.cumsum(v, 0) * dt, axis=0); out["hp_int_%g" % fc] = p4.score(m, emit(p), q[rows])
    r = f.back_final(F, k_drag=m.kd, blind=B.final_blind); out["v5"] = p4.score(m, r["p"], r["q"])
    return dict(name=n, order=o, seed=sd, m=out)
if __name__ == "__main__":
    names = f.available(); jobs = [(n, o, sd) for o in ("A", "B") for sd in (42, 1) for n in names]
    with Pool(10) as p: res = p.map(one, jobs, chunksize=1)
    grp = lambda n: "TUM" if n.startswith("room") else n[:2]
    for r in res:   # LOEO choice of the cut-off: the fc with the lowest mean ATE on the OTHER environment groups
        others = [x for x in res if grp(x["name"]) != grp(r["name"])]; best = min(FC, key=lambda fc: np.mean([x["m"]["hp_int_%g" % fc]["ate"] for x in others])); r["m"]["hp_int_LOEO"] = dict(r["m"]["hp_int_%g" % best], fc=best)
    json.dump(res, open(os.path.join(p4.ROOT, "research/results/e59_baselines.json"), "w"))
    for g, sel in (("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room"))):
        rr = [r for r in res if sel(r["name"])]; print("\n[%s | %d missions]" % (g, len(rr)))
        for k in rr[0]["m"]: print("  %-14s " % k + "  ".join("%s %.3f" % (x, np.nanmean([r["m"][k][x] for r in rr])) for x in ("ate", "wp", "b_abs", "drop", "rpe_t")) + ("   fc chosen %s" % sorted({r["m"][k]["fc"] for r in rr}) if k == "hp_int_LOEO" else ""))
