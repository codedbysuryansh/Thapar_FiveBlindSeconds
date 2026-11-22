#!/usr/bin/env python3
"""E28 - cross-check of our scoring against the organisers' named tool (evo 1.38.1, command line).
For each mission the estimate is written exactly in the required format (one row per IMU row + 5 blind seconds,
'timestamp tx ty tz qx qy qz qw', seconds) and the public ground truth in TUM format; evo_ape / evo_rpe are run on the files.
Reported: evo ATE RMSE without alignment, with --align_origin, with -a (SE(3) Umeyama), with -a -s (Sim(3)); rotation
APE (deg) with -a; RPE over 1 s (200 frames) translation and rotation; and our own numbers computed from the same files.
Usage: e28_evo_check.py [module:function]   (default: v4)"""
import sys, os, json, re, subprocess, tempfile, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p3, fbs4 as f
BIN = os.path.join(p3.ROOT, ".venv/bin"); MISS = [("MH_01_easy", "clean", 0), ("V1_02_medium", "clean", 1), ("room2", "clean", 1), ("V2_02_medium", "gaps", 1), ("room4", "gaps", 2), ("MH_04_difficult", "clean", 2)]
def evo(tool, gt, est, *args):
    r = subprocess.run([os.path.join(BIN, tool), "tum", gt, est, *args], capture_output=True, text=True); m = re.search(r"rmse\s+([0-9.eE+-]+)", r.stdout)
    return float(m.group(1)) if m else float("nan")
def run(est_fn=None):
    rows = []; tmp = tempfile.mkdtemp(prefix="e28_")
    for n, lvl, kf in MISS:
        m = p3.mission(n, lvl, kf, 0); e = f.estimate_v4(m.tn, m.w, m.a, n_blind=m.nb, k_drag=m.kd) if est_fn is None else est_fn(m, m.nb); s = m.s; M = m.M
        ts = s.tn[:M] * 1e-9; est = os.path.join(tmp, "est_%s_%s.txt" % (n, lvl)); gt = os.path.join(tmp, "gt_%s.txt" % n); q = e["q"] / np.linalg.norm(e["q"], axis=1, keepdims=True)
        with open(est, "w") as fh:
            fh.write("# timestamp tx ty tz qx qy qz qw\n")
            for i in range(M): fh.write("%.9f %.6f %.6f %.6f %.7f %.7f %.7f %.7f\n" % (ts[i], *e["p"][i], *q[i]))
        ok = np.where(s.ok)[0]
        with open(gt, "w") as fh:
            for i in ok: fh.write("%.9f %.6f %.6f %.6f %.7f %.7f %.7f %.7f\n" % (s.tn[i] * 1e-9, *s.gt_p[i], *s.gt_q[i]))
        A = m.ok; P = e["p"]; Rm, tt = f.umeyama(P[A], m.gp[A]); ours_full = float(np.sqrt(np.mean(np.sum((P[A] @ Rm.T + tt - m.gp[A]) ** 2, 1)))); sc = p3.score(m, e["p"], e["q"])
        r = dict(name=n, lvl=lvl, kf=kf, rows=M, evo_none=evo("evo_ape", gt, est), evo_origin=evo("evo_ape", gt, est, "--align_origin"), evo_a=evo("evo_ape", gt, est, "-a"), evo_as=evo("evo_ape", gt, est, "-a", "-s"), evo_rot_a=evo("evo_ape", gt, est, "-a", "-r", "angle_deg"),
                 evo_rpe_t=evo("evo_rpe", gt, est, "-a", "-d", "200", "-u", "f"), evo_rpe_r=evo("evo_rpe", gt, est, "-a", "-d", "200", "-u", "f", "-r", "angle_deg"), evo_rpe_t_all=evo("evo_rpe", gt, est, "-a", "-d", "200", "-u", "f", "--all_pairs"),
                 ours_full_file_se3=ours_full, ours_prefail_se3=sc["ate"], ours_sim3=sc["ate_sim3"], ours_rpe=sc["rpe"], ours_ori=sc["ori"]); rows.append(r)
        print("%-16s %-5s kf %d rows %6d | evo ATE: none %7.3f origin %7.3f  -a %6.3f  -a -s %6.3f | ours (same rows, SE3) %6.3f  diff %.1e | ours pre-failure only %6.3f | rot APE -a %6.2f deg (ours yaw-aligned %5.2f) | RPE 1 s: evo %6.3f m, %5.2f deg; all pairs %6.3f; ours %6.3f" % (n, lvl, kf, M, r["evo_none"], r["evo_origin"], r["evo_a"], r["evo_as"], ours_full, abs(ours_full - r["evo_a"]), sc["ate"], r["evo_rot_a"], sc["ori"], r["evo_rpe_t"], r["evo_rpe_r"], r["evo_rpe_t_all"], sc["rpe"]), flush=True)
    return rows
if __name__ == "__main__":
    fn = None; tag = "v4"
    if len(sys.argv) > 1:
        import importlib; mod, name = sys.argv[1].split(":"); fn = getattr(importlib.import_module(mod), name); tag = name
    rows = run(fn); json.dump(rows, open(os.path.join(p3.ROOT, "research/results/e28_evo_check_%s.json" % tag), "w"))
