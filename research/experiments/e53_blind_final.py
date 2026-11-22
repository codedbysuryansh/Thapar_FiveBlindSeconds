#!/usr/bin/env python3
"""E53 - the five blind seconds on the FROZEN v5 pipeline (fbs6.estimate_final; blind rows = the trailing invalid rows of the
official input, FAQ: no rows appended). One estimate per mission; every candidate in blind5 fills the blind rows from the state at
the last valid sample. Exact scoring (p4.score: SE(3) Umeyama over all rows incl. blind). Extra outputs for the analysis:
 - ORACLE rows (diagnostic only, never a candidate): true velocity at failure (damped CV), perfect anchored displacement;
 - compact stacking data (candidates and target at 0.1 s marks, expressed in the estimate frame through the PRE-failure alignment)
   so that forecast combinations can be fitted under leave-one-environment-out in the main process."""
import sys, os, json, time, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p4, fbs6 as f, blind5 as B
CFGS = [("full", dict(seed=s)) for s in (42, 1, 2, 3, 4)] + [("crop60", dict(seed=s, crop=0.6)) for s in (42, 7)] + \
       [("crop80", dict(seed=s, crop=0.8)) for s in (42, 7)] + [("crop40", dict(seed=42, crop=0.4))]
STACK = ["hold", "model", "cv_0.5", "cv_1", "cv_2", "ar5", "ar10", "centroid_5"]
def one(args):
    n, order, cname, ckw = args; t0 = time.time(); m = p4.official(n, order, **ckw)
    r = f.estimate_final(m.tn, m.w, m.a, k_drag=m.kd, want=True); e, P, L, rows = r["e"], r["P"], r["L"], r["rows"]; nt = r["n_tail"]
    pos, vel, N0, fs = B.state(e, P); D = B.position_candidates(e, P, nt); pT = pos[N0 - 1]
    slot = rows; pre = m.ok & (slot <= L); bl = np.where(slot > L)[0]; okb = m.ok[bl]; gi = np.where(pre)[0][-1]
    Rm, tt = f.umeyama(pos[slot[pre]], m.gp[pre]); vgt = m.s.gt_v[m.idx[gi]] @ Rm; tau = np.arange(1, nt + 1) / fs
    for td in (1.0, None): D["ORACLE_v_%s" % ("inf" if td is None else "%g" % td)] = B.damp(tau, td)[:, None] * vgt
    tgt_rows = bl[okb]; D_or = np.zeros((nt, 3))
    if len(tgt_rows):   # perfect anchored displacement (in the estimate frame through the pre-failure alignment), interpolated over slots
        sl = slot[tgt_rows] - L - 1; dg = (m.gp[tgt_rows] - m.gp[gi]) @ Rm
        D_or = np.stack([np.interp(np.arange(nt), sl, dg[:, j]) for j in range(3)], 1)
    D["ORACLE_disp"] = D_or
    res = {}; qd = e["q"][slot]
    for k, d in D.items():
        full = np.concatenate([pos[:N0], pT + d])[slot]; res[k] = p4.score(m, full, qd if k == "model" else None)
    O = B.orientation_candidates(e, P, nt); pm = np.concatenate([pos[:N0], pT + D["model"]])[slot]
    for k, qq in O.items(): res["ori:" + k] = p4.score(m, pm, np.concatenate([P["q"], qq])[slot])
    # stacking data at 0.1 s marks (blind rows with ground truth)
    st = None
    if len(tgt_rows) >= 10:
        taus = (slot[tgt_rows] - L) / fs; marks = np.arange(0.1, 5.0001, 0.1); pick = np.unique([int(np.argmin(np.abs(taus - mk))) for mk in marks])
        rr = tgt_rows[pick]; si = slot[rr] - L - 1; tgt = (m.gp[rr] - m.gp[gi]) @ Rm
        st = dict(tau=taus[pick].tolist(), tgt=tgt.tolist(), X={k: D[k][si].tolist() for k in STACK}, end_err=float(np.linalg.norm(pos[slot[gi]] @ Rm.T + tt - m.gp[gi])))
    return dict(name=n, order=order, cfg=cname, seed=ckw["seed"], m=res, rotor=bool(r["rotor"]), mode=r["mode"], guard=bool(r["guard"]),
                disp=float(np.linalg.norm(m.gp[tgt_rows] - m.gp[gi], axis=1).mean()) if len(tgt_rows) else float("nan"),
                speed=float(np.linalg.norm(m.s.gt_v[m.idx[gi]])), vel_err=float(np.linalg.norm(vel[N0 - 1] - vgt)), n_tail=int(nt), stack=st, sec=time.time() - t0)
if __name__ == "__main__":
    names = f.available(); jobs = [(n, o, c, kw) for o in ("A", "B") for c, kw in CFGS for n in names]
    with Pool(10) as p: res = p.map(one, jobs, chunksize=1)
    json.dump(res, open(os.path.join(p4.ROOT, "research/results/e53_blind_final.json"), "w"))
    print("missions %d, mean sec/mission %.1f" % (len(res), np.mean([r["sec"] for r in res])))
