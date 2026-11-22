#!/usr/bin/env python3
"""Phase 4 tests T1 (output contract, evo_ape) and T2 (regression against the frozen research reference) for release/main.py
on the 34 official-format files (17 sequences x column orders A/B, degrade_imu.py seed 42, data/phase4/inputs).

Per file: the real command line (`python release/main.py INPUT OUTPUT`) in a fresh process; tools/check_output.py; evo_ape
(SE(3) Umeyama alignment over all poses) against ground truth; p4.score (the research scoring) of the written file; decisions
from an in-process main.estimate; comparison with research/phase4/results/p4_ref_freeze.json and data/phase4/ref/<tag>.npz
(= fbs6.estimate_final(..., cfg=fbs6.V5P4, blind=blind5.final_blind), k_drag 0.1873).
Pass criteria (freeze report section 12.8): identical gate / guard / bias-mode decisions on 34/34; per-file ATE within
max(1 %, 1 cm); mean ATE within 0.5 %; blind-window rows within 1 cm; T1 contract on 34/34 and evo_ape runs.
Ground truth is used for scoring only. Writes research/phase4/results/p4_t2_regression.json and logs/phase4/p4_t2_regression.log."""
import os, sys, json, time, subprocess, importlib.util, numpy as np
from multiprocessing import Pool
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "research", "harness")); import p4                                   # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "release", "tools")); import check_output as CO                       # noqa: E402
MAIN = os.path.join(ROOT, "release", "main.py"); INP = os.path.join(ROOT, "data", "phase4", "inputs"); REFD = os.path.join(ROOT, "data", "phase4", "ref")
OUTD = os.path.join(ROOT, "data", "phase4", "out_t2"); GTD = os.path.join(ROOT, "data", "phase4", "gt_tum"); RES = os.path.join(ROOT, "research", "phase4", "results")
EVO = os.path.join(os.path.dirname(sys.executable), "evo_ape")
SEQ = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult", "V1_01_easy", "V1_02_medium", "V1_03_difficult",
       "V2_01_easy", "V2_02_medium", "V2_03_difficult", "room1", "room2", "room3", "room4", "room5", "room6"]
DEC = ("rotor_gate", "rotor", "guard", "mode", "n_tail", "L", "implausible")
REF = json.load(open(os.path.join(RES, "p4_ref_freeze.json")))
def load_main():
    s = importlib.util.spec_from_file_location("fbs_main", MAIN); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def read_tum(path):
    ts = []; P = []; Q = []
    for ln in open(path):
        f = ln.split(); ts.append(f[0]); P.append([float(x) for x in f[1:4]]); Q.append([float(x) for x in f[4:8]])
    return ts, np.array(P), np.array(Q)
def evo_rmse(gt, est):
    r = subprocess.run([EVO, "tum", gt, est, "-a", "--no_warnings"], capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        p = ln.split()
        if len(p) == 2 and p[0] == "rmse": return float(p[1]), r.returncode
    return float("nan"), r.returncode
def one(args):
    n, order = args; tag = "%s_%s42" % (n, order); zp = os.path.join(INP, tag + ".zip"); out = os.path.join(OUTD, tag + ".txt")
    t0 = time.perf_counter(); pr = subprocess.run([sys.executable, MAIN, zp, out], capture_output=True, text=True); wall = time.perf_counter() - t0
    co = CO.check(zp, out); ts, P, Q = read_tum(out); m = p4.official(n, order, seed=42)
    ns = np.array([int(s.replace(".", "")) for s in ts], np.int64); same_t = bool(np.array_equal(ns, m.tn))
    sc = p4.score(m, P, Q); ref = [r for r in REF if r["tag"] == tag][0]; R = np.load(os.path.join(REFD, tag + ".npz"))
    M = load_main(); D, sol, info = M.estimate(M.Source(zp), M.K_DRAG); dec = {k: info[k] for k in DEC}
    blind = R["rows"] > ref["dec"]["L"]; dp = np.abs(P - R["p"]); dq = np.abs(np.abs((Q / np.linalg.norm(Q, axis=1, keepdims=True) * R["q"]).sum(1)) - 1)
    gt = os.path.join(GTD, tag + ".tum")
    with open(gt, "w") as f:
        for i in np.flatnonzero(m.ok): f.write("%d.%09d %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n" % ((m.tn[i] // 10 ** 9, m.tn[i] % 10 ** 9) + tuple(m.gp[i]) + tuple(m.gq[i])))
    ev, evc = evo_rmse(gt, out)
    return dict(tag=tag, name=n, order=order, exit=pr.returncode, wall=wall, log=pr.stderr.strip().splitlines()[-1] if pr.stderr.strip() else "", t1=co, t_equal=same_t,
                score=sc, ref_score=ref["score"], dec=dec, ref_dec={k: ref["dec"][k] for k in DEC}, dp_max=float(dp.max()), dp_blind=float(dp[blind].max()) if blind.any() else 0.0,
                n_blind=int(blind.sum()), dq_max=float(dq.max()), evo_rmse=ev, evo_exit=evc, lock=info["lock"], ref_lock=ref["dec"]["lock"], h=info["h"], D=info["D"])
if __name__ == "__main__":
    os.makedirs(OUTD, exist_ok=True); os.makedirs(GTD, exist_ok=True); os.makedirs(RES, exist_ok=True); os.makedirs(os.path.join(ROOT, "logs", "phase4"), exist_ok=True)
    jobs = [(n, o) for n in SEQ for o in ("A", "B")]
    with Pool(int(os.environ.get("P4_PROCS", "6"))) as pool: res = pool.map(one, jobs, chunksize=1)
    lines = []; L = lines.append; ok_all = True
    L("T1 + T2: release/main.py vs research reference (V5P4, k_drag 0.1873), 34 official-format files")
    L("%-20s %4s %5s %8s %8s %8s %9s %9s %8s %8s %s" % ("file", "T1", "dec", "ATE", "ATE_ref", "dATE", "max|dp|", "blind|dp|", "evo", "evo-ATE", "log"))
    for r in res:
        a, b = r["score"]["ate"], r["ref_score"]["ate"]; tol = max(0.01 * b, 0.01); r["ate_ok"] = abs(a - b) <= tol; r["dec_ok"] = r["dec"] == r["ref_dec"]; r["blind_ok"] = r["dp_blind"] <= 0.01
        r["t1_ok"] = bool(r["t1"]["ok"] and r["exit"] == 0 and r["t_equal"] and r["evo_exit"] == 0 and np.isfinite(r["evo_rmse"]))
        ok_all &= r["ate_ok"] and r["dec_ok"] and r["blind_ok"] and r["t1_ok"]
        L("%-20s %4s %5s %8.4f %8.4f %+8.1e %9.2e %9.2e %8.4f %+8.1e %s" % (r["tag"], "ok" if r["t1_ok"] else "FAIL", "ok" if r["dec_ok"] else "DIFF", a, b, a - b, r["dp_max"], r["dp_blind"], r["evo_rmse"], r["evo_rmse"] - a, r["log"]))
    summ = {}
    for g, sel in (("EuRoC", lambda n: not n.startswith("room")), ("TUM-VI", lambda n: n.startswith("room")), ("all", lambda n: True)):
        rr = [r for r in res if sel(r["name"])]; summ[g] = {}
        for k in ("ate", "wp", "b_abs", "b_anc", "drop", "ori"):
            v = float(np.mean([r["score"][k] for r in rr])); w = float(np.mean([r["ref_score"][k] for r in rr])); summ[g][k] = dict(main=v, ref=w, rel=(v / w - 1) if w else 0.0)
        ok_all &= abs(summ[g]["ate"]["rel"]) <= 0.005
        L("%-7s n=%2d  ATE %.4f vs %.4f (%+.2e %%)  wp %.4f vs %.4f  b_abs %.4f vs %.4f  ori %.3f vs %.3f deg" % (g, len(rr), summ[g]["ate"]["main"], summ[g]["ate"]["ref"], 100 * summ[g]["ate"]["rel"],
          summ[g]["wp"]["main"], summ[g]["wp"]["ref"], summ[g]["b_abs"]["main"], summ[g]["b_abs"]["ref"], summ[g]["ori"]["main"], summ[g]["ori"]["ref"]))
    L("decisions identical %d/34 | per-file ATE within max(1%%, 1 cm) %d/34 | blind rows within 1 cm %d/34 | T1 contract + evo %d/34 | worst |dp| %.2e m | worst blind |dp| %.2e m"
      % (sum(r["dec_ok"] for r in res), sum(r["ate_ok"] for r in res), sum(r["blind_ok"] for r in res), sum(r["t1_ok"] for r in res), max(r["dp_max"] for r in res), max(r["dp_blind"] for r in res)))
    L("T1 + T2: %s" % ("PASS" if ok_all else "FAIL"))
    txt = "\n".join(lines); print(txt); open(os.path.join(ROOT, "logs", "phase4", "p4_t2_regression.log"), "w").write(txt + "\n")
    json.dump(dict(files=res, summary=summ, pass_=bool(ok_all), python=sys.version, numpy=np.__version__), open(os.path.join(RES, "p4_t2_regression.json"), "w"), indent=1, default=float)
