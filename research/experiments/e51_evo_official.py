#!/usr/bin/env python3
"""E51 - evo cross-check and format audit on OFFICIAL missions. For each mission the estimate is written exactly as the submission
requires (one row per input row incl. duplicated rows and the zero-valued blind rows; 'timestamp tx ty tz qx qy qz qw'; seconds =
ns x 1e-9 printed exactly; unit quaternions) and the public ground truth in TUM format; evo 1.38.1 (CLI) computes APE with SE(3)
Umeyama alignment (-a, as the FAQ states) and RPE; our scorer runs on the same rows. Also checks whether evo accepts the duplicated
timestamps that the official script produces."""
import sys, os, re, json, subprocess, tempfile, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p4, fbs6 as f
BIN = os.path.join(p4.ROOT, ".venv/bin")
MISS = [("MH_01_easy", "A", 42), ("MH_01_easy", "B", 42), ("V1_02_medium", "B", 1), ("V2_03_difficult", "A", 2), ("room2", "A", 42), ("room4", "B", 42)]
def evo(tool, gt, est, *args):
    r = subprocess.run([os.path.join(BIN, tool), "tum", gt, est, *args], capture_output=True, text=True); mm = re.search(r"rmse\s+([0-9.eE+-]+)", r.stdout)
    return (float(mm.group(1)) if mm else float("nan")), (r.stderr.strip().splitlines()[-1][:160] if r.returncode else "")
def write_traj(path, tn, p, q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True); sec, ns = np.divmod(np.asarray(tn, np.int64), 10 ** 9)
    with open(path, "w") as fh:
        fh.write("# timestamp tx ty tz qx qy qz qw\n")
        for a, b, pp, qq in zip(sec.tolist(), ns.tolist(), p.tolist(), q.tolist()): fh.write("%d.%09d %.6f %.6f %.6f %.7f %.7f %.7f %.7f\n" % (a, b, *pp, *qq))
def audit(path, m):
    rows = [l.split() for l in open(path) if l.strip() and not l.startswith("#")]; ok8 = all(len(r) == 8 for r in rows); ts = [r[0] for r in rows]
    exp = ["%d.%09d" % divmod(int(x), 10 ** 9) for x in m.tn]; qn = np.array([[float(x) for x in r[4:]] for r in rows]); fin = all(np.isfinite([float(x) for x in r[1:]]).all() for r in rows)
    return dict(rows=len(rows), input_rows=len(m.tn), eight_columns=ok8, timestamps_identical=ts == exp, max_quat_norm_err=float(np.abs(np.linalg.norm(qn, axis=1) - 1).max()), finite=bool(fin), dup_timestamps=int(len(ts) - len(set(ts))))
if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="e51_"); out = []
    for n, o, sd in MISS:
        m = p4.official(n, o, sd); e = f.estimate_final(m.tn, m.w, m.a, k_drag=m.kd)
        est = os.path.join(tmp, "est_%s_%s_%d.txt" % (n, o, sd)); gt = os.path.join(tmp, "gt_%s.txt" % n); write_traj(est, m.tn, e["p"], e["q"]); s = m.s; ok = np.where(s.ok)[0]
        with open(gt, "w") as fh:
            for i in ok: fh.write("%.9f %.6f %.6f %.6f %.7f %.7f %.7f %.7f\n" % (s.tn[i] * 1e-9, *s.gt_p[i], *s.gt_q[i]))
        # a de-duplicated copy, in case evo rejects repeated timestamps
        lines = open(est).read().splitlines(); seen = set(); dd = [lines[0]] + [l for l in lines[1:] if not (l.split()[0] in seen or seen.add(l.split()[0]))]; estd = est.replace(".txt", "_dedup.txt"); open(estd, "w").write("\n".join(dd) + "\n")
        sc = p4.score(m, e["p"], e["q"]); r = dict(name=n, order=o, seed=sd, audit=audit(est, m), ours_ate=sc["ate"], ours_rpe_t=sc["rpe_t"], ours_rpe_r=sc["rpe_r"], ours_rpe1_t=sc["rpe1_t"])
        for tag, path in (("as_is", est), ("dedup", estd)):
            for k, args in (("ate_a", ("-a",)), ("ate_none", ()), ("ate_as", ("-a", "-s")), ("rot_a", ("-a", "-r", "angle_deg")), ("rpe200_t", ("-a", "-d", "200", "-u", "f")), ("rpe200_r", ("-a", "-d", "200", "-u", "f", "-r", "angle_deg")), ("rpe1_t", ("-a",))):
                tool = "evo_rpe" if k.startswith("rpe") else "evo_ape"; v, err = evo(tool, gt, path, *args); r["%s_%s" % (tag, k)] = v
                if err: r["%s_%s_err" % (tag, k)] = err
        out.append(r); print(json.dumps(r)[:900], flush=True)
    json.dump(out, open(os.path.join(p4.ROOT, "research/results/e51_evo_official.json"), "w"))
