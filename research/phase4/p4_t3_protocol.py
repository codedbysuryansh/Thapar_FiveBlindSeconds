#!/usr/bin/env python3
"""Phase 4 test T3: the official protocol of E65 (degrade_imu.py with seed 42 and seeds 5-13, both column orders, the 17 public
sequences = 340 full missions) run through release/main.py, paired with E65's V5P4 results (research/results/e65_p4_config.json,
condition "full", entry "v5_p4"). As in E65, k_drag is the leave-one-sequence-out value p4.k_loso(n). Each mission is written
in the official format (EuRoC header and column order) to a temporary file and estimated by main.estimate in-process; scoring
with p4.score against ground truth. Pass: group means within 1 % of E65 (EuRoC 2.670 m, TUM-VI 0.451 m).
Writes research/phase4/results/p4_t3_protocol.json and logs/phase4/p4_t3_protocol.log."""
import os, sys, json, time, tempfile, importlib.util, numpy as np, pandas as pd
from multiprocessing import Pool
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "research", "harness")); import p4                                   # noqa: E402
MAIN = os.path.join(ROOT, "release", "main.py"); RES = os.path.join(ROOT, "research", "phase4", "results")
SEEDS = [42, 5, 6, 7, 8, 9, 10, 11, 12, 13]
SEQ = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult", "V1_01_easy", "V1_02_medium", "V1_03_difficult",
       "V2_01_easy", "V2_02_medium", "V2_03_difficult", "room1", "room2", "room3", "room4", "room5", "room6"]
E65 = {(r["name"], r["order"], r["seed"]): r for r in json.load(open(os.path.join(ROOT, "research", "results", "e65_p4_config.json"))) if r["cfg"] == "full"}
def load_main():
    s = importlib.util.spec_from_file_location("fbs_main", MAIN); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def one(args):
    n, order, seed = args; m = p4.official(n, order, seed=seed); M = load_main()
    fd, path = tempfile.mkstemp(suffix=".csv", prefix="fbs_t3_"); os.close(fd)
    try:
        pd.DataFrame(dict(zip(p4.EUROC_COLS, [m.tn, m.w[:, 0], m.w[:, 1], m.w[:, 2], m.a[:, 0], m.a[:, 1], m.a[:, 2]]))).to_csv(path, index=False)
        t0 = time.perf_counter(); D, sol, info = M.estimate(M.Source(path), m.kd); s = np.concatenate([sl for _ts, sl, _v in D.rows(False)]); p, q = sol.poses(s); dt = time.perf_counter() - t0
    finally: os.remove(path)
    sc = p4.score(m, p, q); e = E65[(n, order, seed)]
    return dict(name=n, order=order, seed=seed, score=sc, e65=e["m"]["v5_p4"], rotor=bool(info["rotor"]), e65_rotor=bool(e["rotor"][1]), sec=dt, mode=info["mode"], guard=info["guard"])
if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True); os.makedirs(os.path.join(ROOT, "logs", "phase4"), exist_ok=True)
    jobs = [(n, o, s) for s in SEEDS for o in ("A", "B") for n in SEQ]
    with Pool(int(os.environ.get("P4_PROCS", "8"))) as pool: res = pool.map(one, jobs, chunksize=2)
    lines = []; L = lines.append; ok = True
    L("T3: release/main.py on the E65 'full' protocol (seeds %s, orders A/B, 17 sequences = %d missions), k_drag = leave-one-sequence-out" % (SEEDS, len(res)))
    for g, sel, target in (("EuRoC", lambda n: not n.startswith("room"), 2.670), ("TUM-VI", lambda n: n.startswith("room"), 0.451)):
        rr = [r for r in res if sel(r["name"])]; out = {}
        for k in ("ate", "wp", "b_abs", "b_anc", "drop", "ori"):
            a = float(np.mean([r["score"][k] for r in rr])); b = float(np.mean([r["e65"][k] for r in rr])); out[k] = (a, b)
        d = np.array([r["score"]["ate"] - r["e65"]["ate"] for r in rr]); rel = out["ate"][0] / out["ate"][1] - 1; ok &= abs(rel) <= 0.01 and abs(out["ate"][0] / target - 1) <= 0.01
        L("%-7s n=%3d  ATE %.4f (E65 %.4f, frozen value %.3f; %+.2e %%)  per-mission |dATE| max %.2e m  wp %.4f/%.4f  b_abs %.4f/%.4f  b_anc %.4f/%.4f  ori %.3f/%.3f"
          % (g, len(rr), out["ate"][0], out["ate"][1], target, 100 * rel, np.abs(d).max(), out["wp"][0], out["wp"][1], out["b_abs"][0], out["b_abs"][1], out["b_anc"][0], out["b_anc"][1], out["ori"][0], out["ori"][1]))
    L("gate decisions equal to E65: %d/%d; mean time per mission %.2f s; T3: %s" % (sum(r["rotor"] == r["e65_rotor"] for r in res), len(res), np.mean([r["sec"] for r in res]), "PASS" if ok else "FAIL"))
    txt = "\n".join(lines); print(txt); open(os.path.join(ROOT, "logs", "phase4", "p4_t3_protocol.log"), "w").write(txt + "\n")
    json.dump(dict(missions=res, pass_=bool(ok)), open(os.path.join(RES, "p4_t3_protocol.json"), "w"), indent=1, default=float)
