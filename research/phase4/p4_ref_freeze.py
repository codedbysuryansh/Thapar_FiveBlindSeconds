#!/usr/bin/env python3
"""Phase 4 - freeze the research reference on the 34 official files (17 public sequences x column orders A/B, seed 42).
Reference = fbs6.estimate_final(tn, w, a, k_drag=K_DRAG, cfg=fbs6.V5P4, blind=blind5.final_blind) - the submission configuration.
The research harness is imported, never modified. Ground truth is used only for the stored scores.
Writes
  data/phase4/inputs/<seq>_<order>42.zip   official-format input: imu0/data.csv, EuRoC header and column order, the organisers'
                                           degraded values written the way degrade_imu.py writes them (pandas to_csv)
  data/phase4/ref/<seq>_<order>42.npz      reference output rows + front-end and smoother traces (T2 and stage-by-stage checks)
  research/phase4/results/p4_ref_freeze.json   decisions, scores, timing per file
Usage: .venv/bin/python research/phase4/p4_ref_freeze.py [processes]"""
import os, sys, json, time, zipfile
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ.setdefault(_k, "1")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "research", "harness"))
import numpy as np, pandas as pd
import fbs6, blind5, p4
from multiprocessing import Pool

OUT_IN = os.path.join(ROOT, "data", "phase4", "inputs"); OUT_REF = os.path.join(ROOT, "data", "phase4", "ref"); OUT_RES = os.path.join(HERE, "results")
SEQ = fbs6.fbs.ALL

def official_csv(m):
    """The official evaluation format (slide 15): EuRoC header and column order (timestamp, gyro, accel)."""
    cols = [m.tn, m.w[:, 0], m.w[:, 1], m.w[:, 2], m.a[:, 0], m.a[:, 1], m.a[:, 2]]
    return pd.DataFrame(dict(zip(p4.EUROC_COLS, cols))).to_csv(index=False)

def run(job):
    n, order, seed = job; tag = "%s_%s%d" % (n, order, seed); m = p4.official(n, order, seed=seed)
    with zipfile.ZipFile(os.path.join(OUT_IN, tag + ".zip"), "w", zipfile.ZIP_DEFLATED) as z: z.writestr("imu0/data.csv", official_csv(m))
    t0 = time.perf_counter(); F = fbs6.front_final(m.tn, m.w, m.a, cfg=fbs6.V5P4)
    e = fbs6.back_final(F, k_drag=fbs6.K_DRAG, blind=blind5.final_blind, want=True); sec = time.perf_counter() - t0
    P = F["P"]; E = e["e"]; sc = p4.score(m, e["p"], e["q"])
    np.savez_compressed(os.path.join(OUT_REF, tag + ".npz"), p=e["p"], q=e["q"], rows=F["rows"], t=P["t"], qa=P["q"], bg=P["bg"], valid=P["valid"],
                        zmask=P["zmask"], wl=P["wl"], wc=P["wc"], af=P["af"], aw=P["aw"], xs=E["xs"], tk=E["tk"], tn=m.tn)
    dec = dict(rotor_gate=bool(F["rotor"]), rotor=bool(e["rotor"]), guard=bool(e["guard"]), lock=float(F["lock"]), mode=F["mode"], D=float(F["D"]), h=float(F["h"]),
               coarse=bool(F["coarse"]), n_tail=int(F["nt"]), L=int(F["L"]), implausible=int(F["implausible"]), spikes_legacy=int(F["spikes_legacy"]))
    return dict(tag=tag, name=n, order=order, seed=seed, rows=int(len(m.tn)), seconds=sec, dec=dec, score={k: float(v) for k, v in sc.items()})

if __name__ == "__main__":
    for d in (OUT_IN, OUT_REF, OUT_RES): os.makedirs(d, exist_ok=True)
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    jobs = [(n, o, 42) for n in SEQ for o in ("A", "B")]
    with Pool(nproc) as pool: res = pool.map(run, jobs, chunksize=1)
    json.dump(res, open(os.path.join(OUT_RES, "p4_ref_freeze.json"), "w"), indent=1)
    for r in res: print("%-18s rows %6d  rotor %d guard %d mode %-6s lock %.4f  ate %.3f  b_abs %.3f  %.1f s" % (r["tag"], r["rows"], r["dec"]["rotor"], r["dec"]["guard"], r["dec"]["mode"], r["dec"]["lock"], r["score"]["ate"], r["score"]["b_abs"], r["seconds"]))
    for g, sel in (("EuRoC", lambda r: not r["name"].startswith("room")), ("TUM-VI", lambda r: r["name"].startswith("room"))):
        rr = [r for r in res if sel(r)]; print("%s mean ATE %.4f  wp %.4f  b_abs %.4f  (n=%d)" % (g, np.mean([r["score"]["ate"] for r in rr]), np.mean([r["score"]["wp"] for r in rr]), np.mean([r["score"]["b_abs"] for r in rr]), len(rr)))
