#!/usr/bin/env python3
"""E16b - window/knot scan for the time-varying gyro-bias front end (front4), whole sequences, tilt error and the
no-drag Matern ATE as a common downstream probe. Hybrid 'g+w' = two global (constant) LS passes when there is no rest
anchor, then windowed increments. Run: .venv/bin/python research/experiments/e16b_front_scan.py"""
import sys, os, json, numpy as np
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbs3 as f
LV = {"clean": {}, "gaps": dict(gaps=[2.0] * 3), "L2": dict(gn=0.05, an=0.5, gb=0.02, ab=0.2, gaps=[2.0] * 3, rot=True), "gb02": dict(gb=0.02), "gb05": dict(gb=0.05), "gbrw": dict(gb=0.03, bias_shape="rw")}
CFG = {"f3": None, "w30k10": (30, 10, False), "w60k20": (60, 20, False), "w90k30": (90, 30, False), "g+w30k10": (30, 10, True), "g+w60k20": (60, 20, True)}
def work(args):
    n, lvl = args; s = f.load(n); w, a, vd, meta = f.degrade(s.t, s.w, s.a, LV[lvl], seed=sum(n.encode()) % 997); tu, wu, au, vu, rows = f.ingest(s.tn, w, a); qR = Rot.from_matrix(meta["Rr"]).as_quat(); ok = s.ok; out = {}
    anc = f.rest_anchors(tu, wu, au, vu); bt = f.bias_track(tu, anc); MAT = f.model_matern52(*f.MATERN)
    for k, c in CFG.items():
        if c is None: q, bg = f.attitude3(tu, wu, au, vu, bg_track=bt)
        else:
            b0 = bt
            if c[2] and bt is None: b0 = np.broadcast_to(f.attitude3(tu, wu, au, vu)[1], (len(tu), 3)).copy()
            q, bg = f.attitude4(tu, wu, au, vu, bg_track=b0, win=c[0], knot=c[1])
        qo = f.qmul(q[rows], np.broadcast_to(qR, (len(rows), 4))); p = f.lin_smooth(tu, q, f.fill_gaps(tu, au, vu), vu, MAT, bias_jump=0.5)[0][rows]
        out[k] = [float(np.mean(f.tilt_deg(qo[ok], s.gt_q[ok]))), f.ate(p[ok], s.gt_p[ok])]
    return n, lvl, out
if __name__ == "__main__":
    names = f.available(); jobs = [(n, l) for l in LV for n in names]
    with Pool(10) as p: res = p.map(work, jobs, chunksize=1)
    json.dump(res, open(os.path.join(f.fbs.ROOT, "research/results/e16b_front_scan.json"), "w"))
    for grp, sel in (("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room"))):
        print("\n%s: mean tilt [deg] / Matern ATE [m]" % grp)
        for lvl in LV:
            rr = [r for r in res if r[1] == lvl and sel(r[0])]; print("  %-5s " % lvl + "  ".join("%s %.2f/%.3f" % (k, np.mean([r[2][k][0] for r in rr]), np.mean([r[2][k][1] for r in rr])) for k in CFG))
