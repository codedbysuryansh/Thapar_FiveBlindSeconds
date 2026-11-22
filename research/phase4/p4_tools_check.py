#!/usr/bin/env python3
"""Phase 4: checks the two release tools against the research harness.
  1. tools/evaluate.py (NumPy only) on the 34 T2 outputs vs p4.score (research scorer, SciPy rotations) and evo_ape (from T2).
  2. tools/evaluate.py --protocol (inputs from tools/make_samples.py, main.py in-process, scored from the written file) vs T3
     (research protocol, in-process positions) on the 340 missions: group means and per-mission ATE.
Usage: .venv/bin/python research/phase4/p4_tools_check.py [--protocol PROCS]"""
import os, sys, json
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_k] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..")); REL = os.path.join(ROOT, "release")
sys.path.insert(0, os.path.join(REL, "tools"))
import numpy as np                                                       # noqa: E402
import evaluate as EV                                                    # noqa: E402
OUT = os.path.join(HERE, "results"); LOG = os.path.join(ROOT, "logs", "phase4", "p4_tools_check.log")
def seqdir(n): return os.path.join(ROOT, "data/tumvi/dataset-%s_512_16/mav0" % n) if n.startswith("room") else os.path.join(ROOT, "data/euroc", n, "mav0")
def gtfile(n): return os.path.join(seqdir(n), "mocap0" if n.startswith("room") else "state_groundtruth_estimate0", "data.csv")
def main():
    lines = []
    def L(s): print(s, flush=True); lines.append(s)
    T2 = json.load(open(os.path.join(OUT, "p4_t2_regression.json")))["files"]; res = dict(files=[])
    L("1. tools/evaluate.py vs p4.score and evo_ape on the 34 T2 outputs (data/phase4/out_t2)")
    worst = {k: 0.0 for k in ("ate", "wp", "b_abs", "b_anc", "drop", "rpe_t", "rpe_r", "ori")}; wevo = 0.0
    for r in T2:
        tag, n = r["tag"], r["name"]; s = EV.score_files(os.path.join(ROOT, "data/phase4/out_t2", tag + ".txt"), os.path.join(ROOT, "data/phase4/inputs", tag + ".zip"), gtfile(n))
        d = {k: abs(s["score"][k] - r["score"][k]) for k in worst}; wevo = max(wevo, abs(s["score"]["ate"] - r["evo_rmse"]))
        for k in worst: worst[k] = max(worst[k], d[k])
        res["files"].append(dict(tag=tag, evaluate=s["score"], p4_score=r["score"], evo=r["evo_rmse"], diff=d))
    L("   largest |evaluate.py - p4.score| over 34 files: " + ", ".join("%s %.1e" % (k, v) for k, v in worst.items()) + "  (m; deg for ori, rpe_r)")
    L("   largest |evaluate.py ATE - evo_ape| over 34 files: %.1e m" % wevo); res["worst"] = worst; res["worst_evo"] = wevo
    if "--protocol" in sys.argv:
        import argparse, time
        procs = int(sys.argv[sys.argv.index("--protocol") + 1])
        a = argparse.Namespace(sequences="", euroc=os.path.join(ROOT, "data/euroc"), tumvi=os.path.join(ROOT, "data/tumvi"), degrade=os.path.join(ROOT, "research/official/degrade_imu.py"),
                               orders="A,B", seeds="42,5,6,7,8,9,10,11,12,13", k_drag="loso", procs=procs, json=os.path.join(OUT, "p4_tools_protocol.json"))
        L("2. tools/evaluate.py --protocol (340 missions) vs T3"); t0 = time.time(); rc = EV.protocol(a); L("   protocol exit %d, %.0f s" % (rc, time.time() - t0))
        P = json.load(open(a.json)); T3 = json.load(open(os.path.join(OUT, "p4_t3_protocol.json")))["missions"]
        t3 = {(m["name"], m["order"], m["seed"]): m for m in T3}; dd = []
        for m in P:
            o = t3.get((m["name"], m["order"], m["seed"]))
            if o is not None and "score" in m: dd.append(abs(m["score"]["ate"] - o["score"]["ate"]))
        for g, sel in (("EuRoC", lambda m: not m["name"].startswith("room")), ("TUM-VI", lambda m: m["name"].startswith("room"))):
            pp = [m for m in P if sel(m)]; mm = lambda k, w="score": np.nanmean([m[w][k] for m in pp])
            L("   %-6s n=%d  ATE %.4f (motionless %.4f)  wp %.4f (%.4f)  b_abs %.4f (%.4f)  b_anc %.4f  drop %.4f (%.4f)  rpe_t %.4f  ori %.3f" % (g, len(pp), mm("ate"), mm("ate", "motionless"),
              mm("wp"), mm("wp", "motionless"), mm("b_abs"), mm("b_abs", "motionless"), mm("b_anc"), mm("drop"), mm("drop", "motionless"), mm("rpe_t"), mm("ori")))
        L("   per-mission |ATE(evaluate.py, written file) - ATE(T3, in-process)|: matched %d/340, max %.1e m" % (len(dd), max(dd) if dd else float("nan")))
        res["protocol_matched"] = len(dd); res["protocol_max_dATE"] = max(dd) if dd else None
    json.dump(res, open(os.path.join(OUT, "p4_tools_check.json"), "w"), indent=1); open(LOG, "w").write("\n".join(lines) + "\n")
if __name__ == "__main__":
    main()
