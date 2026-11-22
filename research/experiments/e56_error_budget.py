#!/usr/bin/env python3
"""E56 - ERROR BUDGET of the frozen v5 on official missions (DIAGNOSTIC ONLY: ground truth is injected on purpose to measure how much
each error source costs; nothing here is a candidate). Substitutions after the v5 front end, then the unchanged v5 back end:
  v5            : as frozen
  ORACLE_att    : ground-truth orientation (roll, pitch, yaw)
  ORACLE_yaw    : v5 roll/pitch, ground-truth yaw (removes heading drift and the frame offset)
  ORACLE_tilt   : v5 yaw error kept, ground-truth roll/pitch
  ORACLE_vel1Hz : v5 attitude + ground-truth velocity pseudo-measurements at 1 Hz (sigma 0.05 m/s) - ceiling for velocity aiding
  ORACLE_vel_lo : same at 0.1 Hz (sigma 0.1 m/s)"""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p4, fbs6 as f
from scipy.spatial.transform import Rotation as Rot
def front_parts(tn, w, a, cfg=None):
    c = dict(f.V5, **(cfg or {})); old = f.fbs5.GS_K2; f.fbs5.GS_K2 = c["k2"]
    try:
        F = f.front_final(tn, w, a, cfg); t, wu, au, valid, rows = f.ingest2(tn, w, a); L = F["L"]; t0, wu0, au0, v0 = t[:L + 1], wu[:L + 1], au[:L + 1], valid[:L + 1]
        wu0, au0, v0, ri = f.repair(t0, wu0, au0, v0, g_lo=5.0, g_hi=15.5); cm = f.common_of(t0, wu0, au0, v0); return F, (t0, wu0, au0, v0, cm)
    finally: f.fbs5.GS_K2 = old
def one(args):
    n, o, sd = args; m = p4.official(n, o, seed=sd); F, (t0, wu0, au0, v0, cm) = front_parts(m.tn, m.w, m.a); P = F["P"]; q = P["q"]; bg = P["bg"]
    k = np.clip(np.searchsorted(m.s.t - m.s.t[0], P["t"] - 1e-4), 0, len(m.s.t) - 1); ok = m.s.ok[k]; Rg = Rot.from_quat(m.s.gt_q[k]); Re = Rot.from_quat(q)
    Rerr = (Re * Rg.inv()).as_matrix(); dpsi = np.arctan2(Rerr[:, 1, 0], Rerr[:, 0, 0]); Rz = lambda ang: Rot.from_rotvec(np.outer(ang, [0, 0, 1.0]))
    def fill(qq): qq = qq.copy(); qq[~ok] = q[~ok]; return qq / np.linalg.norm(qq, axis=1, keepdims=True)
    Q = {"ORACLE_att": fill(Rg.as_quat()), "ORACLE_yaw": fill((Rz(-dpsi) * Re).as_quat()), "ORACLE_tilt": fill((Rz(dpsi) * Rg).as_quat())}
    out = {}; r = f.back_final(F, k_drag=m.kd); out["v5"] = p4.score(m, r["p"], r["q"])
    for tag, qq in Q.items():
        F2 = dict(F); F2["P"] = f.prep_from(t0, wu0, au0, v0, qq, bg, True, cm); r = f.back_final(F2, k_drag=m.kd); out[tag] = p4.score(m, r["p"], r["q"])
    vgt = f.qrot(q, f.qrot(f.qinv(m.s.gt_q[k]), m.s.gt_v[k]))   # ground-truth velocity in the estimator's world frame
    for tag, hz, sg in (("ORACLE_vel1Hz", 1.0, 0.05), ("ORACLE_vel_lo", 0.1, 0.1)):
        c = np.arange(0, len(P["t"]), int(round(P["fs"] / hz))); c = c[ok[c]]; r = f.back_final(F, k_drag=m.kd, vel=(c, vgt[c], sg)); out[tag] = p4.score(m, r["p"], r["q"])
    return dict(name=n, order=o, seed=sd, m=out, rotor=F["rotor"])
if __name__ == "__main__":
    names = f.available(); jobs = [(n, o, sd) for o in ("A", "B") for sd in (42, 1) for n in names]
    with Pool(10) as p: res = p.map(one, jobs, chunksize=1)
    json.dump(res, open(os.path.join(p4.ROOT, "research/results/e56_error_budget.json"), "w"))
    for g, sel in (("EuRoC", lambda r: not r["name"].startswith("room")), ("TUM", lambda r: r["name"].startswith("room"))):
        rr = [r for r in res if sel(r)]; print("\n[%s | %d missions]" % (g, len(rr)))
        for v in rr[0]["m"]: print("  %-14s " % v + "  ".join("%s %.3f" % (x, np.nanmean([r["m"][v][x] for r in rr])) for x in ("ate", "wp", "b_abs", "drop", "rpe_t", "ori")))
    print("\nper sequence ATE (v5 | ORACLE_yaw | ORACLE_att | ORACLE_vel1Hz):")
    for n in names:
        rr = [r for r in res if r["name"] == n]; print("  %-16s %.3f | %.3f | %.3f | %.3f" % (n, *[np.mean([r["m"][v]["ate"] for r in rr]) for v in ("v5", "ORACLE_yaw", "ORACLE_att", "ORACLE_vel1Hz")]))
