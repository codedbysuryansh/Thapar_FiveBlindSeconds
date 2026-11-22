#!/usr/bin/env python3
"""Phase 4 test T4 (robustness): release/main.py against the research reference (fbs6.front_final / back_final with
cfg=fbs6.V5P4, blind5.final_blind, k_drag 0.1873) on three families of inputs beyond the 34 official files:
  E46  our stress suite (29 conditions: long gaps in four encodings, gyro ramps / random walks, gain faults, spikes,
       accelerometer ramps, heavy / exponential noise, combined levels, half sample rate, timestamp jitter), 17 sequences, seed 0,
       failure at 60 %, 5 blind seconds as zero rows
  E55  windows of the official degradation (prefixes 15-60 %, suffixes, middles, full), orders A/B, seed 42
  E60  the official degradation re-expressed in four rotated sensor frames, orders A/B, seed 42
Every input is rounded to 6 decimals (the official format) and given identically to both: main.py through a CSV file (EuRoC header
and column order), the reference as arrays. Pass: no exception in main.estimate, decisions (gate, guard, bias mode, blind length,
implausible-window count) identical to the reference on every case where the reference runs. Max position difference reported.
Writes research/phase4/results/p4_t4_robustness.json and logs/phase4/p4_t4_robustness.log."""
import os, sys, json, time, tempfile, importlib.util, numpy as np, pandas as pd
from multiprocessing import Pool
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "research", "harness")); sys.path.insert(0, os.path.join(ROOT, "research", "experiments"))
import p3, p4, fbs6, blind5                                                                                     # noqa: E402
MAIN = os.path.join(ROOT, "release", "main.py"); RES = os.path.join(ROOT, "research", "phase4", "results")
DEC = ("rotor_gate", "rotor", "guard", "mode", "n_tail", "L", "implausible")
def load_main():
    s = importlib.util.spec_from_file_location("fbs_main", MAIN); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def e46_input(c, n, seed):
    from e46_stress_v5c import COND
    kw = dict(COND[c]); post = kw.pop("post", None); m = p3.mission(n, kw.pop("lvl", "clean"), 1, seed, **kw); nb = m.nb; tn, w, a = m.tn, m.w, m.a
    if post == "half": tn, w, a = tn[::2], w[::2], a[::2]; nbe = nb // 2; step = 2
    else: nbe = nb; step = 1
    if post == "jitter": tn = tn + np.round(np.random.default_rng(seed).normal(size=len(tn)) * 3e5).astype(np.int64)
    dtn = int(np.median(np.diff(m.s.tn))) * step; tb = m.s.tn[m.iT] + dtn * np.arange(1, nbe + 1)
    return np.concatenate([tn, tb]), np.vstack([w, np.zeros((nbe, 3))]), np.vstack([a, np.zeros((nbe, 3))])
def case_input(case):
    fam = case[0]
    if fam == "E46": return e46_input(*case[1:])
    if fam == "E55":
        _f, n, o, wn, a0, b0 = case; m = p4.official(n, o, seed=42, crop=b0, start=a0); return m.tn, m.w, m.a
    _f, n, o, R = case; m = p4.official(n, o, seed=42); R = np.array(R); return m.tn, m.w @ R.T, m.a @ R.T
def one(case):
    tn, w, a = case_input(case); w = np.round(w, 6); a = np.round(a, 6); out = dict(case=[c if not isinstance(c, list) else "R" for c in case], rows=int(len(tn)))
    try:
        F = fbs6.front_final(tn, w, a, cfg=fbs6.V5P4); e = fbs6.back_final(F, k_drag=fbs6.K_DRAG, blind=blind5.final_blind)
        ref = dict(rotor_gate=bool(F["rotor"]), rotor=bool(e["rotor"]), guard=bool(e["guard"]), mode=F["mode"], n_tail=int(F["nt"]), L=int(F["L"]), implausible=int(F["implausible"]))
        out["ref"] = ref; rp = e["p"]
    except Exception as ex: out["ref_error"] = repr(ex)[:200]; rp = None                               # noqa: BLE001
    fd, path = tempfile.mkstemp(suffix=".csv", prefix="fbs_t4_"); os.close(fd); M = load_main()
    try:
        pd.DataFrame(dict(zip(p4.EUROC_COLS, [tn, w[:, 0], w[:, 1], w[:, 2], a[:, 0], a[:, 1], a[:, 2]]))).to_csv(path, index=False)
        t0 = time.perf_counter(); D, sol, info = M.estimate(M.Source(path), M.K_DRAG); s = np.concatenate([sl for _ts, sl, _v in D.rows(False)]); p, q = sol.poses(s)
        out["sec"] = time.perf_counter() - t0; out["main"] = {k: info[k] for k in DEC}
        if rp is not None: out["dp_max"] = float(np.nanmax(np.abs(p - rp))); out["dec_ok"] = out["main"] == out["ref"]
    except Exception as ex: out["main_error"] = repr(ex)[:200]                                             # noqa: BLE001
    finally: os.remove(path)
    return out
if __name__ == "__main__":
    from e46_stress_v5c import CONDS
    from e55_gate_windows import WIN
    from e60_sensor_rotation import ROTS
    names = fbs6.fbs.ALL
    cases = [("E46", c, n, 0) for c in CONDS for n in names] + [("E55", n, o, wn, a, b) for o in ("A", "B") for wn, a, b in WIN for n in names] \
        + [("E60", n, o, R.tolist()) for k, R in ROTS.items() if k != "none" for o in ("A", "B") for n in names]
    os.makedirs(RES, exist_ok=True); os.makedirs(os.path.join(ROOT, "logs", "phase4"), exist_ok=True)
    with Pool(int(os.environ.get("P4_PROCS", "10"))) as pool: res = pool.map(one, cases, chunksize=2)
    lines = []; L = lines.append; ok = True
    for fam in ("E46", "E55", "E60"):
        rr = [r for r in res if r["case"][0] == fam]; me = [r for r in rr if "main_error" in r]; re_ = [r for r in rr if "ref_error" in r]; cmp_ = [r for r in rr if "dec_ok" in r]
        bad = [r for r in cmp_ if not r["dec_ok"]]; ok &= not me and not bad
        L("%s: %d cases | main errors %d | reference errors %d | decisions identical %d/%d | max |dp| %.2e m | mean time %.2f s"
          % (fam, len(rr), len(me), len(re_), len(cmp_) - len(bad), len(cmp_), max([r["dp_max"] for r in cmp_] or [0.0]), np.mean([r["sec"] for r in rr if "sec" in r])))
        for r in me[:5] + bad[:5]: L("   %s %s" % (r["case"], r.get("main_error", "") or ("main %s ref %s" % (r["main"], r["ref"]))))
        for r in re_[:3]: L("   reference failed on %s: %s" % (r["case"], r["ref_error"]))
    L("T4: %s" % ("PASS" if ok else "FAIL")); txt = "\n".join(lines); print(txt)
    open(os.path.join(ROOT, "logs", "phase4", "p4_t4_robustness.log"), "w").write(txt + "\n")
    json.dump(dict(cases=res, pass_=bool(ok)), open(os.path.join(RES, "p4_t4_robustness.json"), "w"), indent=1, default=float)
