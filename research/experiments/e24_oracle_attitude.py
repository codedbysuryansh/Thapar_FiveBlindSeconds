#!/usr/bin/env python3
"""E24 - ORACLE DIAGNOSTIC (never an inference path): how much of the loss under degradation is attitude?
The v4 back end is run with (a) its own front end, (b) ground-truth orientation in place of the estimated one,
(c) ground-truth TILT with the estimated heading. The gap between (a) and (b)/(c) bounds what any better
gyro-bias / tilt / heading estimator could buy; what remains is noise, accelerometer bias, gaps and the prior."""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3, fbs5 as f
from e19_stress import ramp
CONDS = {"clean": {}, "gaps": dict(lvl="gaps"), "L1": dict(lvl="L1"), "L2": dict(lvl="L2"), "L3": dict(lvl="L3"), "gb05": dict(lvl="gb05"), "noise": dict(lvl="noise"), "ab10": dict(lvl="ab10"),
         "ramp_r_0.0005": dict(fault=ramp(0.0005, None)), "ramp_r_0.002": dict(fault=ramp(0.002, None)), "ramp_r_0.01": dict(fault=ramp(0.01, None))}
def swap(P, q):
    P2 = dict(P); P2["q"] = q; aw = f.qrot(q, P["af"]); P2["aw"] = aw - aw[P["valid"]].mean(0); return P2
def one(args):
    c, n, seed = args; kw = dict(CONDS[c]); m = p3.mission(n, kw.pop("lvl", "clean"), 1, seed, **kw); P = f.prep_v4(m.tn, m.w, m.a); out = {}
    e = f.estimate_v6(m.tn, m.w, m.a, n_blind=m.nb, k_drag=m.kd, P=P); out["v4"] = p3.score(m, e["p"], e["q"])
    qgt = f.qmul(m.gq[:m.iT + 1], f.qinv(f.Rot.from_matrix(m.meta["Rr"]).as_quat())[None]); ok = m.ok[:m.iT + 1]
    if ok.mean() > 0.98:
        idx = np.arange(len(qgt)); nn = np.where(ok)[0]; qg = qgt[nn[np.clip(np.searchsorted(nn, idx), 0, len(nn) - 1)]]; qg = qg / np.linalg.norm(qg, axis=1, keepdims=True)
        e = f.estimate_v6(m.tn, m.w, m.a, n_blind=m.nb, k_drag=m.kd, P=swap(P, qg)); out["oracle_q"] = p3.score(m, e["p"], e["q"])
        # ground-truth tilt, estimated heading: rotate the estimate about the world vertical so that its tilt matches
        ez = np.array([0, 0, 1.0]); zb_gt = f.qrot(f.qinv(qg), np.broadcast_to(ez, (len(qg), 3))); zw = f.qrot(P["q"], zb_gt); qfix = f.shortest_arc(zw / np.linalg.norm(zw, axis=1, keepdims=True), np.broadcast_to(ez, zw.shape)); qt = f.qmul(qfix, P["q"]); qt /= np.linalg.norm(qt, axis=1, keepdims=True)
        e = f.estimate_v6(m.tn, m.w, m.a, n_blind=m.nb, k_drag=m.kd, P=swap(P, qt)); out["oracle_tilt"] = p3.score(m, e["p"], e["q"])
    return dict(name=n, lvl=c, seed=seed, m=out)
if __name__ == "__main__":
    names = f.available()
    with Pool(10) as p: res = p.map(one, [(c, n, sd) for c in CONDS for n in names for sd in range(2)], chunksize=1)
    json.dump(res, open(os.path.join(p3.ROOT, "research/results/e24_oracle_attitude.json"), "w")); res = [r for r in res if "oracle_q" in r["m"]]
    p3.table(res, ["v4", "oracle_tilt", "oracle_q"], list(CONDS), keys=("wp", "b_abs", "b_anc", "gap", "ate", "rpe", "tilt", "ori"), groups=(("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room"))))
