#!/usr/bin/env python3
"""E17b - whole-sequence clean ATE of the final v4 pipeline on all 17 sequences, with BOTH skill definitions and a
bootstrap CI (percentile, 20,000 resamples over sequences, seed 0). k_drag leave-one-flight-out.
Run: .venv/bin/python research/experiments/e17b_clean_table.py"""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbs3 as f
from e10_mission import KD
def work(n):
    s = f.load(n); ok = s.ok; kd = float(np.mean([v["fits"]["2.0"]["k"] for m, v in KD.items() if m != n])); A = lambda p: f.ate(p[ok], s.gt_p[ok])
    e4 = f.estimate_v4(s.tn, s.w, s.a, k_drag=kd); e3 = f.estimate_final(s.tn, s.w, s.a, k_drag=kd); e1 = f.estimate3(s.tn, s.w, s.a)
    return dict(name=n, ml=A(np.zeros((len(s.t), 3))), p1b=A(e1["p"]), e13=A(e3["p"]), v4=A(e4["p"]), rotor=e4["rotor"], lock=e4["lock"], D=e4["drift_D"] * 1e3, zupt=e4["zupt_rows"] / 200.0, dur=float(s.t[-1]))
if __name__ == "__main__":
    with Pool(9) as p: res = p.map(work, f.available())
    json.dump(res, open(os.path.join(f.fbs.ROOT, "research/results/e17b_clean_table.json"), "w"), indent=1); rng = np.random.default_rng(0)
    print("%-16s %6s %6s %6s %6s %6s | rotor lock   D[mrad/s] rest[s]" % ("sequence", "dur[s]", "motnl", "P1B", "e13", "v4"))
    for r in res: print("%-16s %6.1f %6.3f %6.3f %6.3f %6.3f | %d   %.4f %5.2f   %5.1f" % (r["name"], r["dur"], r["ml"], r["p1b"], r["e13"], r["v4"], r["rotor"], r["lock"], r["D"], r["zupt"]))
    for grp, sel in (("ALL", lambda n: True), ("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room"))):
        rr = [r for r in res if sel(r["name"])]; ml = np.array([r["ml"] for r in rr])
        for k in ("p1b", "e13", "v4"):
            x = np.array([r[k] for r in rr]); sk = 1 - x / ml; bs = sk[rng.integers(0, len(sk), (20000, len(sk)))].mean(1); bx = rng.integers(0, len(x), (20000, len(x))); ba = 1 - x[bx].mean(1) / ml[bx].mean(1)
            print("%-5s %-4s mean ATE %.3f (motionless %.3f) | aggregate reduction %.1f%% [95%% CI %.1f, %.1f] | mean per-sequence skill %.1f%% [%.1f, %.1f] | wins %d/%d" % (grp, k, x.mean(), ml.mean(), 100 * (1 - x.mean() / ml.mean()), *np.percentile(100 * ba, [2.5, 97.5]), 100 * sk.mean(), *np.percentile(100 * bs, [2.5, 97.5]), (x < ml).sum(), len(x)))
