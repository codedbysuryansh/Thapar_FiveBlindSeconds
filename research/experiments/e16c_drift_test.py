#!/usr/bin/env python3
"""E16c - can gyro-bias DRIFT be detected from the stream, so that the time-varying correction is applied only when
needed? D = RMS temporal variation of the windowed bias increments (rad/s). Whole sequences.
Run: .venv/bin/python research/experiments/e16c_drift_test.py"""
import sys, os, json, numpy as np
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbs3 as f
LV = {"clean": {}, "gaps": dict(gaps=[2.0] * 3), "L1": dict(gn=0.01, an=0.1, gb=0.005, ab=0.05, gaps=[2.0] * 3, rot=True), "L2": dict(gn=0.05, an=0.5, gb=0.02, ab=0.2, gaps=[2.0] * 3, rot=True),
      "gb01": dict(gb=0.01), "gb02": dict(gb=0.02), "gb05": dict(gb=0.05), "gbrw": dict(gb=0.03, bias_shape="rw"), "ab10": dict(ab=1.0), "noise": dict(gn=0.05, an=0.5)}
def work(args):
    n, lvl = args; s = f.load(n); w, a, vd, meta = f.degrade(s.t, s.w, s.a, LV[lvl], seed=sum(n.encode()) % 997); tu, wu, au, vu, rows = f.ingest(s.tn, w, a); qR = Rot.from_matrix(meta["Rr"]).as_quat(); ok = s.ok
    anc = f.rest_anchors(tu, wu, au, vu); base = f.bias_track(tu, anc)
    if base is None: q3, b3 = f.attitude3(tu, wu, au, vu); base = np.broadcast_to(b3, (len(tu), 3)).copy()
    else: q3, b3 = f.attitude3(tu, wu, au, vu, bg_track=base)
    q4, b4 = f.attitude4(tu, wu, au, vu, bg_track=base, win=60.0, knot=20.0); d = b4 - base; D = float(np.sqrt(((d - d.mean(0)) ** 2).sum(1).mean())); MAT = f.model_matern52(*f.MATERN); af = f.fill_gaps(tu, au, vu); out = {"D": D}
    for k, q in (("const", q3), ("windowed", q4)):
        qo = f.qmul(q[rows], np.broadcast_to(qR, (len(rows), 4))); p = f.lin_smooth(tu, q, af, vu, MAT, bias_jump=0.5)[0][rows]; out[k] = [float(np.mean(f.tilt_deg(qo[ok], s.gt_q[ok]))), f.ate(p[ok], s.gt_p[ok])]
    return n, lvl, out
if __name__ == "__main__":
    names = f.available(); jobs = [(n, l) for l in LV for n in names]
    with Pool(10) as p: res = p.map(work, jobs, chunksize=1)
    json.dump(res, open(os.path.join(f.fbs.ROOT, "research/results/e16c_drift_test.json"), "w"))
    for grp, sel in (("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room"))):
        print("\n%s: D [mrad/s] min/median/max | tilt const -> windowed | Matern ATE const -> windowed | adaptive ATE at thr 1.0 / 1.5 / 2.0 / 3.0 mrad/s (fraction switched)" % grp)
        for lvl in LV:
            rr = [r[2] for r in res if r[1] == lvl and sel(r[0])]; D = np.array([r["D"] for r in rr]) * 1e3
            ad = lambda th: "%.3f (%.2f)" % (np.mean([r["windowed"][1] if r["D"] * 1e3 > th else r["const"][1] for r in rr]), np.mean(D > th))
            print("  %-5s D %.2f/%.2f/%.2f | tilt %.2f -> %.2f | ATE %.3f -> %.3f | %s" % (lvl, D.min(), np.median(D), D.max(), np.mean([r["const"][0] for r in rr]), np.mean([r["windowed"][0] for r in rr]), np.mean([r["const"][1] for r in rr]), np.mean([r["windowed"][1] for r in rr]), "  ".join(ad(th) for th in (1.0, 1.5, 2.0, 3.0))))
