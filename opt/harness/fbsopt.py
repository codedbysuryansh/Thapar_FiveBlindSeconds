#!/usr/bin/env python3
"""fbsopt.py - Phase 5 optimisation harness (research only: never shipped, never published).

Every candidate is a folder opt/candidates/<ID>/ that holds a complete main.py snapshot, config.json (module overrides applied
after import) and meta.json (SHA-256 of both, parent, hypothesis). Mission sets are the organisers' degrade_imu.py (SHA-256
cc7398db...2f31, unmodified) applied to the 17 public sequences, cached once as the zip files main.py reads. Ground truth is
used only to score, never by an estimator. Results, logs and comparisons are written next to the candidate; registry.json and
REGISTRY.md record every candidate with its status and the reason.

  fbsopt.py build [--sets S42,DEV,CONF,STRESS] [--procs 10]
  fbsopt.py new ID --src FILE [--config JSON] --title T --hypothesis H [--parent P]
  fbsopt.py run ID --set DEV [--procs 11] [--keep]
  fbsopt.py compare ID --set DEV [--ref CHAMP-1.0.0]
  fbsopt.py decide ID --status accepted|rejected|pending|inconclusive|promoted --reason TEXT
  fbsopt.py registry
"""
import os, sys, io, json, time, math, zipfile, hashlib, argparse, importlib.util, tempfile, shutil, platform, datetime
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"
import numpy as np

ROOT = "~/SamsungPRISM_FiveBlindSeconds"
OPT = os.path.join(ROOT, "opt"); CACHE = os.path.join(OPT, "cache"); CANDS = os.path.join(OPT, "candidates")
CHAMP_TOOLS = os.path.join(ROOT, "champion/v1.0.0/release/tools")      # read-only copies of evaluate.py / make_samples.py
DEGRADE = os.path.join(ROOT, "research/official/degrade_imu.py"); DEGRADE_SHA256 = "cc7398db47efe8cefb581c13501f43687db2a5a13e7ed89697dc98afcd4b2f31"
EUROC_DIR = os.path.join(ROOT, "data/euroc"); TUMVI_DIR = os.path.join(ROOT, "data/tumvi")
EUROC = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult", "V1_01_easy", "V1_02_medium", "V1_03_difficult",
         "V2_01_easy", "V2_02_medium", "V2_03_difficult"]
TUMVI = ["room%d" % i for i in range(1, 7)]
SEQS = EUROC + TUMVI
def env_group(n): return "TUM" if n.startswith("room") else n[:2]           # MH, V1, V2, TUM (leave-one-environment-out groups)
def dataset(n): return "TUM-VI" if n.startswith("room") else "EuRoC"
sys.path.insert(0, CHAMP_TOOLS)
import evaluate as EV                                                      # noqa: E402  (NumPy-only scorer of Phase 4, = p4.score)

def _ms(seeds, orders="AB", **kw): return [dict(seed=int(s), order=o, **kw) for s in seeds for o in orders]
SETS = {
    "S42": _ms([42]),                                                       # the organisers' fixed seed: reported, never selected on
    "DEV": _ms([1001, 1002, 1003, 1004]),                                   # exploration and tuning
    "DEV2": _ms([2001, 2002, 2003, 2004]),                                  # second exploration set (replication before CONF)
    "CONF": _ms(range(3001, 3011)),                                         # promotion decisions only
    "TRAIN": _ms([5001, 5002, 5003, 5004]),                                 # training data of the learned-model retest only
    "STRESS": _ms([4001, 4002], crop=0.6) + _ms([4003, 4004], crop=0.8) + _ms([4011, 4012], max_level=0.5) + _ms([4013, 4014], max_level=1.5)
              + _ms([4021], gamma=1.0) + _ms([4022], gamma=3.0),
}
def tag_of(name, sp):
    t = "%s_%s%d" % (name, sp["order"], sp["seed"])
    if sp.get("crop", 1.0) < 1.0: t += "_c%g" % sp["crop"]
    if sp.get("max_level", 1.0) != 1.0: t += "_ml%g" % sp["max_level"]
    if sp.get("gamma", 1.7) != 1.7: t += "_g%g" % sp["gamma"]
    return t
def missions(setname, seqs=None):
    return [(n, sp, tag_of(n, sp)) for sp in SETS[setname] for n in (seqs or SEQS)]
def mav0(n): return os.path.join(TUMVI_DIR, "dataset-%s_512_16" % n, "mav0") if n.startswith("room") else os.path.join(EUROC_DIR, n, "mav0")
def gt_file(n): return os.path.join(mav0(n), "mocap0" if n.startswith("room") else "state_groundtruth_estimate0", "data.csv")
K_FIT = EV.K_FIT
def k_loso(n): return float(np.mean([v for k, v in K_FIT.items() if k != n]))

# ---------------------------------------------------------------- cache: degraded inputs and ground truth
_DEG = None
def _degrade_module():
    global _DEG
    if _DEG is None:
        h = hashlib.sha256(open(DEGRADE, "rb").read()).hexdigest(); assert h == DEGRADE_SHA256, "degrade_imu.py changed: %s" % h
        spec = importlib.util.spec_from_file_location("degrade_imu", DEGRADE); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); _DEG = m
    return _DEG
def degraded_csv(n, order, seed, max_level=1.0, gamma=1.7, crop=1.0):
    """= tools/make_samples.degraded_csv, plus an optional crop of the clean recording before degradation (failure earlier)."""
    import pandas as pd
    deg = _degrade_module(); imu = pd.read_csv(os.path.join(mav0(n), "imu0", "data.csv"), comment="#", header=None)
    tn = imu[0].values.astype(np.int64); w = imu.iloc[:, 1:4].values.astype(float); a = imu.iloc[:, 4:7].values.astype(float)
    if crop < 1.0:
        end = int(np.searchsorted(tn, tn[0] + int(crop * (tn[-1] - tn[0])))); tn, w, a = tn[:end], w[:end], a[:end]
    EC = ["#timestamp [ns]", "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]", "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"]
    DC = ["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
    if order == "A": df = pd.DataFrame(dict(zip(DC, [tn, a[:, 0], a[:, 1], a[:, 2], w[:, 0], w[:, 1], w[:, 2]])))
    else: df = pd.DataFrame(dict(zip(EC, [tn, w[:, 0], w[:, 1], w[:, 2], a[:, 0], a[:, 1], a[:, 2]])))
    cfg = dict(deg.CONFIG); cfg["max_level"] = max_level; cfg["gamma"] = gamma
    out, st = deg.degrade(df, cfg, seed); tno = out.iloc[:, 0].to_numpy(np.int64)
    if order == "A": a_o = out.iloc[:, 1:4].to_numpy(float); w_o = out.iloc[:, 4:7].to_numpy(float)
    else: w_o = out.iloc[:, 1:4].to_numpy(float); a_o = out.iloc[:, 4:7].to_numpy(float)
    return pd.DataFrame(dict(zip(EC, [tno, w_o[:, 0], w_o[:, 1], w_o[:, 2], a_o[:, 0], a_o[:, 1], a_o[:, 2]]))).to_csv(index=False), st, tno
def gt_cached(n):
    f = os.path.join(CACHE, "gt", n + ".npz")
    if not os.path.exists(f):
        ns, p, q = EV.read_gt(gt_file(n)); np.savez(f, ns=ns, p=p, q=q)
    z = np.load(f); return z["ns"], z["p"], z["q"]
def _build_one(job):
    n, sp, tag = job; zf = os.path.join(CACHE, "missions", tag + ".zip"); nf = os.path.join(CACHE, "missions", tag + ".npz")
    if os.path.exists(zf) and os.path.exists(nf): return tag, "cached"
    txt, st, tno = degraded_csv(n, sp["order"], sp["seed"], sp.get("max_level", 1.0), sp.get("gamma", 1.7), sp.get("crop", 1.0))
    tmp = zf + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z: z.writestr("imu0/data.csv", txt)
    m = EV.mission(tno, gt_cached(n)); s0 = EV.score(m, np.zeros((len(tno), 3)), np.tile([0.0, 0.0, 0.0, 1.0], (len(tno), 1)))
    np.savez(nf, tno=tno, motionless=json.dumps(s0), stats=json.dumps(st), csv_sha256=hashlib.sha256(txt.encode()).hexdigest())
    os.replace(tmp, zf); return tag, "built"
def build(a):
    for n in SEQS: gt_cached(n)
    jobs = [j for s in a.sets.split(",") for j in missions(s)]
    from multiprocessing import get_context
    with get_context("spawn").Pool(a.procs) as pool:
        done = 0
        for tag, how in pool.imap_unordered(_build_one, jobs, chunksize=2):
            done += 1
            if how == "built" and done % 20 == 0: print("built %d / %d" % (done, len(jobs)), flush=True)
    print("cache ready: %d missions" % len(jobs))

# ---------------------------------------------------------------- candidates
def cdir(cid): return os.path.join(CANDS, cid)
def sha256f(path): return hashlib.sha256(open(path, "rb").read()).hexdigest()
import contextlib, fcntl
@contextlib.contextmanager
def reg_txn():
    """Exclusive read-modify-write of registry.json (several harness processes may run at once)."""
    with open(os.path.join(OPT, "registry.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            reg = load_reg(); yield reg; save_reg(reg)
        finally: fcntl.flock(lk, fcntl.LOCK_UN)
def load_reg():
    f = os.path.join(OPT, "registry.json")
    return json.load(open(f)) if os.path.exists(f) else dict(candidates={}, created=datetime.datetime.now().isoformat(timespec="seconds"))
def save_reg(reg):
    f = os.path.join(OPT, "registry.json"); tmp = f + ".tmp"; json.dump(reg, open(tmp, "w"), indent=1); os.replace(tmp, f); write_registry_md(reg)
def new(a):
    d = cdir(a.id)
    if os.path.exists(d): sys.exit("candidate %s exists; candidates are never overwritten" % a.id)
    os.makedirs(d); shutil.copy2(a.src, os.path.join(d, "main.py")); cfg = json.loads(a.config) if a.config else {}
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1, sort_keys=True)
    meta = dict(id=a.id, title=a.title, hypothesis=a.hypothesis, parent=a.parent, created=datetime.datetime.now().isoformat(timespec="seconds"),
                src_from=os.path.abspath(a.src), main_sha256=sha256f(os.path.join(d, "main.py")), config=cfg,
                config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest())
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    with reg_txn() as reg: reg["candidates"][a.id] = dict(meta, status="pending", reason="", results={})
    print("candidate %s created (main.py %s)" % (a.id, meta["main_sha256"][:16]))
_MODS = {}
def load_candidate(cid):
    if cid in _MODS: return _MODS[cid]
    d = cdir(cid); meta = json.load(open(os.path.join(d, "meta.json"))); path = os.path.join(d, "main.py")
    assert sha256f(path) == meta["main_sha256"], "main.py of %s changed after registration" % cid
    spec = importlib.util.spec_from_file_location("cand_" + cid.replace("-", "_").replace(".", "_"), path); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    cfg = json.load(open(os.path.join(d, "config.json")))
    for k, v in cfg.get("set", {}).items():
        if isinstance(getattr(M, k, None), dict) and isinstance(v, dict): getattr(M, k).update(v)
        elif isinstance(getattr(M, k, None), tuple): setattr(M, k, tuple(v))
        else: setattr(M, k, v)
    _MODS[cid] = (M, cfg); return _MODS[cid]

# ---------------------------------------------------------------- run and score
def extra_metrics(m, p):
    out = {}; fin = np.isfinite(p).all(1); A = m.ok & fin
    if A.sum() < 10: return out
    Rm, tt = EV.umeyama(p[A], m.gp[A]); e = p @ Rm.T + tt - m.gp
    out["ate_h"] = float(np.sqrt(np.mean(e[A, 0] ** 2 + e[A, 1] ** 2))); out["ate_v"] = float(np.sqrt(np.mean(e[A, 2] ** 2)))
    pf = p[fin]; out["est_radius"] = float(np.sqrt(np.mean(np.sum((pf - pf.mean(0)) ** 2, 1)))); out["gt_radius"] = float(np.sqrt(np.mean(np.sum((m.gp[A] - m.gp[A].mean(0)) ** 2, 1))))
    pre = A & ~m.blind; Tp = m.T - 5.0; ci = [int(np.argmin(np.abs(np.where(pre, m.t, 1e9) - Tp * (k + 1) / 9))) for k in range(8)]
    er = np.linalg.norm(e, axis=1); out["wp_max"] = float(er[ci].max())
    for j, (lo, hi) in enumerate(((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0))):
        sel = pre & (m.t >= lo * Tp) & (m.t < hi * Tp); out["ate_t%d" % j] = float(np.sqrt(np.mean(er[sel] ** 2))) if sel.any() else float("nan")
    return out
def _run_one(job):
    cid, n, tag, keep = job; M, cfg = load_candidate(cid); zf = os.path.join(CACHE, "missions", tag + ".zip"); z = np.load(os.path.join(CACHE, "missions", tag + ".npz"))
    tno = z["tno"]; kdm = cfg.get("k_drag", "loso")
    kd = k_loso(n) if kdm == "loso" else (float(kdm) if not isinstance(kdm, dict) else float(kdm.get(n, kdm.get("default"))))
    od = os.path.join(cdir(cid), "outputs", tag) if keep else tempfile.mkdtemp(prefix="fbs_")
    os.makedirs(od, exist_ok=True); est = os.path.join(od, "estimated_trajectory.txt"); err = ""
    t0 = time.perf_counter(); c0 = time.process_time()
    try: _p, info = M.run(zf, est, k_drag=kd)
    except Exception as ex: info = dict(fallback="exception"); err = repr(ex)                # noqa: BLE001
    wall = time.perf_counter() - t0; cpu = time.process_time() - c0
    r = dict(tag=tag, name=n, group=env_group(n), dataset=dataset(n), rows=int(len(tno)), imu_s=float((tno[-1] - tno[0]) * 1e-9), wall=wall, cpu=cpu, k_drag=kd,
             info={k: (v if isinstance(v, (bool, int, str)) or v is None else float(v)) for k, v in info.items()}, error=err, motionless=json.loads(str(z["motionless"])))
    try:
        r["out_sha256"] = sha256f(est); ens, p, q = EV.read_estimate(est)
        if len(ens) != len(tno) or not (ens == tno).all(): r["error"] = (err + " contract").strip()
        else:
            m = EV.mission(tno, gt_cached(n)); r["score"] = EV.score(m, p, q); r["score"].update(extra_metrics(m, p))
    except Exception as ex: r["error"] = (err + " score:" + repr(ex)).strip()                 # noqa: BLE001
    if not keep: shutil.rmtree(od, ignore_errors=True)
    return r
def run(a):
    d = cdir(a.id); assert os.path.exists(d), "no candidate " + a.id
    ms = missions(a.set, a.seqs.split(",") if a.seqs else None)
    for _n, _sp, tag in ms: assert os.path.exists(os.path.join(CACHE, "missions", tag + ".zip")), "set %s not built (%s)" % (a.set, tag)
    jobs = [(a.id, n, tag, a.keep) for n, _sp, tag in ms]; t0 = time.time(); res = []
    logf = open(os.path.join(d, "run_%s.log" % a.set), "w"); logf.write("run %s on %s: %d missions, %d processes, %s, %s\n" % (a.id, a.set, len(jobs), a.procs, platform.platform(), datetime.datetime.now().isoformat(timespec="seconds")))
    from multiprocessing import get_context
    with get_context("spawn").Pool(a.procs) as pool:
        for r in pool.imap_unordered(_run_one, jobs, chunksize=1):
            res.append(r); s = r.get("score", {})
            logf.write("%-28s wall %.3f s  ate %s  info %s  %s\n" % (r["tag"], r["wall"], "%.4f" % s["ate"] if s else "-", json.dumps(r["info"], sort_keys=True), r["error"])); logf.flush()
    res.sort(key=lambda r: r["tag"]); meta = json.load(open(os.path.join(d, "meta.json")))
    out = dict(candidate=a.id, set=a.set, main_sha256=meta["main_sha256"], config=meta["config"], machine=platform.platform(), python=sys.version.split()[0], numpy=np.__version__,
               started=datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds"), seconds=time.time() - t0, missions=res)
    json.dump(out, open(os.path.join(d, "results_%s.json" % a.set), "w"), indent=1)
    bad = [r["tag"] for r in res if r["error"] or "score" not in r]; logf.write("done in %.1f s; errors %d %s\n" % (time.time() - t0, len(bad), bad[:10])); logf.close()
    print("%s on %s: %d missions in %.1f s, %d errors" % (a.id, a.set, len(res), time.time() - t0, len(bad)))
    summary(res)

# ---------------------------------------------------------------- summaries and paired comparison
COMPONENTS = (("wp", 30.0), ("b_abs", 20.0), ("drop", 10.0))
SHOW = ("ate", "wp", "b_abs", "drop", "b_anc", "rpe_t", "rpe_r", "ate_h", "ate_v", "wp_max")
def points(r):
    """Accuracy points proxy (out of 60): rubric weight x max(0, 1 - error / motionless error) per component (no official map)."""
    s, s0 = r["score"], r["motionless"]; tot = 0.0; wsum = 0.0
    for c, w in COMPONENTS:
        if np.isfinite(s.get(c, np.nan)) and np.isfinite(s0.get(c, np.nan)) and s0[c] > 0: tot += w * max(0.0, 1.0 - s[c] / s0[c]); wsum += w
    return tot * 60.0 / wsum if wsum else float("nan")
def summary(res):
    for dsn in ("EuRoC", "TUM-VI", "ALL"):
        rr = [r for r in res if "score" in r and (dsn == "ALL" or r["dataset"] == dsn)]
        if not rr: continue
        print("  %-6s n=%-4d " % (dsn, len(rr)) + "  ".join("%s %.4f" % (k, np.nanmean([r["score"].get(k, np.nan) for r in rr])) for k in SHOW[:6])
              + "  pts %.2f  ms/s %.2f" % (np.nanmean([points(r) for r in rr]), 1e3 * np.median([r["wall"] / r["imu_s"] for r in rr])))
def load_results(cid, setname):
    f = os.path.join(cdir(cid), "results_%s.json" % setname)
    if not os.path.exists(f): sys.exit("no results for %s on %s" % (cid, setname))
    return {r["tag"]: r for r in json.load(open(f))["missions"]}
def _boot(seq_vals, groups, stat, B=4000, seed=20261122):
    """Percentile CI of stat(selected sequences) under sequence resampling stratified by dataset."""
    rng = np.random.default_rng(seed); keys = list(seq_vals); idx = {g: [i for i, k in enumerate(keys) if groups[k] == g] for g in set(groups.values())}; out = []
    for _ in range(B):
        pick = [keys[i] for g in sorted(idx) for i in rng.choice(idx[g], len(idx[g]), replace=True)]; out.append(stat(pick))
    out = np.array(out); return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))
def compare_core(A, B):
    tags = sorted(set(A) & set(B)); seqs = sorted({A[t]["name"] for t in tags}, key=SEQS.index); rep = dict(n=len(tags), groups={})
    fails = [t for t in tags if ("score" not in A[t] or A[t]["error"] or A[t]["info"].get("fallback")) and not ("score" not in B[t] or B[t]["error"] or B[t]["info"].get("fallback"))]
    rep["new_failures"] = fails; ok = [t for t in tags if "score" in A[t] and "score" in B[t]]
    rep["byte_identical"] = int(sum(A[t].get("out_sha256") == B[t].get("out_sha256") for t in tags))
    rep["decision_changes"] = [t for t in ok if (A[t]["info"].get("rotor"), A[t]["info"].get("guard"), A[t]["info"].get("mode")) != (B[t]["info"].get("rotor"), B[t]["info"].get("guard"), B[t]["info"].get("mode"))]
    metrics = list(SHOW) + ["pre_ate", "b_end", "ori"]
    def val(r, k): return points(r) if k == "pts" else r["score"].get(k, np.nan)
    per = {}                                                                   # per-sequence means over its missions
    for s in seqs:
        ts = [t for t in ok if A[t]["name"] == s]
        per[s] = {k: (np.nanmean([val(A[t], k) for t in ts]), np.nanmean([val(B[t], k) for t in ts])) for k in metrics + ["pts"]}
        per[s]["ms"] = (np.median([A[t]["wall"] / A[t]["imu_s"] for t in ts]) * 1e3, np.median([B[t]["wall"] / B[t]["imu_s"] for t in ts]) * 1e3)
    groups = {s: dataset(s) for s in seqs}
    for gname in ("EuRoC", "TUM-VI", "ALL"):
        gs = [s for s in seqs if gname == "ALL" or groups[s] == gname]
        if not gs: continue
        gg = {s: groups[s] for s in gs}; G = {}
        def rel(pick, k):
            a = np.nanmean([per[s][k][0] for s in pick]); b = np.nanmean([per[s][k][1] for s in pick]); return 100.0 * (a - b) / b
        def comp(pick):
            return sum(w * rel(pick, c) for c, w in COMPONENTS) / sum(w for _c, w in COMPONENTS)
        for k in metrics:
            a = np.nanmean([per[s][k][0] for s in gs]); b = np.nanmean([per[s][k][1] for s in gs]); lo, hi = _boot({s: 0 for s in gs}, gg, lambda p, k=k: rel(p, k))
            G[k] = dict(cand=a, ref=b, rel=100 * (a - b) / b, ci=(lo, hi), wins=int(sum(per[s][k][0] < per[s][k][1] - 1e-12 for s in gs)), losses=int(sum(per[s][k][0] > per[s][k][1] + 1e-12 for s in gs)))
        lo, hi = _boot({s: 0 for s in gs}, gg, comp); G["composite"] = dict(rel=comp(gs), ci=(lo, hi))
        dp = lambda p: np.nanmean([per[s]["pts"][0] - per[s]["pts"][1] for s in p]); lo, hi = _boot({s: 0 for s in gs}, gg, dp)
        G["points"] = dict(cand=np.nanmean([per[s]["pts"][0] for s in gs]), ref=np.nanmean([per[s]["pts"][1] for s in gs]), diff=dp(gs), ci=(lo, hi))
        G["ms_per_imu_s"] = dict(cand=float(np.median([per[s]["ms"][0] for s in gs])), ref=float(np.median([per[s]["ms"][1] for s in gs])))
        rep["groups"][gname] = G
    worst = []
    for t in ok:                                                               # mission-level composite change, against the group means
        g = rep["groups"][A[t]["dataset"]]; c = sum(w * (val(A[t], k) - val(B[t], k)) / g[k]["ref"] for k, w in COMPONENTS if np.isfinite(val(A[t], k)) and np.isfinite(val(B[t], k))) / 60.0
        worst.append((100 * c, t))
    worst.sort(); rep["best_missions"] = worst[:6]; rep["worst_missions"] = worst[::-1][:6]
    rep["per_sequence"] = {s: {k: per[s][k] for k in ("ate", "wp", "b_abs", "drop", "pts", "ms")} for s in seqs}
    return rep
def fmt_report(rep, title):
    L = [title, "missions %d, byte-identical outputs %d, new failures %d %s, decision changes %d %s" % (rep["n"], rep["byte_identical"], len(rep["new_failures"]), rep["new_failures"][:5],
         len(rep["decision_changes"]), rep["decision_changes"][:5])]
    for gname, G in rep["groups"].items():
        c = G["composite"]; p = G["points"]
        L.append("  [%s] composite %+.2f %% [%+.2f, %+.2f]   points %.3f -> %.3f (%+.3f [%+.3f, %+.3f])   ms/IMU-s %.2f -> %.2f" % (gname, c["rel"], c["ci"][0], c["ci"][1], p["ref"], p["cand"], p["diff"], p["ci"][0], p["ci"][1], G["ms_per_imu_s"]["ref"], G["ms_per_imu_s"]["cand"]))
        for k in ("ate", "wp", "b_abs", "drop", "b_anc", "rpe_t", "ate_h", "ate_v", "wp_max", "ori"):
            g = G[k]; L.append("      %-7s %8.4f -> %8.4f  %+6.2f %% [%+6.2f, %+6.2f]  wins %2d losses %2d" % (k, g["ref"], g["cand"], g["rel"], g["ci"][0], g["ci"][1], g["wins"], g["losses"]))
    L.append("  best missions (composite %%): " + ", ".join("%s %+.1f" % (t, c) for c, t in rep["best_missions"]))
    L.append("  worst missions (composite %%): " + ", ".join("%s %+.1f" % (t, c) for c, t in rep["worst_missions"]))
    return "\n".join(L)
def compare(a):
    A = load_results(a.id, a.set); B = load_results(a.ref, a.set); rep = compare_core(A, B); title = "%s vs %s on %s" % (a.id, a.ref, a.set); txt = fmt_report(rep, title); print(txt)
    base = os.path.join(cdir(a.id), "compare_%s_vs_%s" % (a.set, a.ref)); open(base + ".txt", "w").write(txt + "\n")
    json.dump(rep, open(base + ".json", "w"), indent=1, default=float)
    G = rep["groups"]
    with reg_txn() as reg:
      e = reg["candidates"].setdefault(a.id, dict(status="pending", results={}))
      e.setdefault("results", {})["%s_vs_%s" % (a.set, a.ref)] = dict(
        n=rep["n"], byte_identical=rep["byte_identical"], new_failures=len(rep["new_failures"]), decision_changes=len(rep["decision_changes"]),
        **{gn: dict(composite=G[gn]["composite"]["rel"], ci=G[gn]["composite"]["ci"], points=G[gn]["points"]["diff"], ate=G[gn]["ate"]["rel"], wp=G[gn]["wp"]["rel"],
                    b_abs=G[gn]["b_abs"]["rel"], drop=G[gn]["drop"]["rel"], ms=(G[gn]["ms_per_imu_s"]["ref"], G[gn]["ms_per_imu_s"]["cand"])) for gn in G})
def decide(a):
    with reg_txn() as reg: e = reg["candidates"][a.id]; e["status"] = a.status; e["reason"] = a.reason; e["decided"] = datetime.datetime.now().isoformat(timespec="seconds")
    print("%s -> %s" % (a.id, a.status))
def write_registry_md(reg):
    L = ["# Phase 5 experiment registry", "", "Generated by `opt/harness/fbsopt.py` from `registry.json`. Every candidate is kept, whatever its status.",
         "Composite = rubric-weighted relative change of the mean waypoint (30), blind (20) and after-dropout (10) errors against the reference; negative is better.",
         "95 % intervals: sequence bootstrap stratified by dataset. Sets: DEV (seeds 1001-1004), DEV2 (2001-2004), CONF (3001-3010), STRESS (crop, severity, ramp shape), S42 (the organisers' seed, never selected on).", ""]
    order = {"promoted": 0, "accepted": 1, "pending": 2, "inconclusive": 3, "rejected": 4}
    for cid, e in sorted(reg["candidates"].items(), key=lambda kv: (order.get(kv[1].get("status"), 9), kv[0])):
        L.append("## %s - %s" % (cid, e.get("title", ""))); L.append("")
        L.append("- Status: **%s**%s" % (e.get("status"), (" - " + e["reason"]) if e.get("reason") else ""))
        L.append("- Hypothesis: %s" % e.get("hypothesis", "")); L.append("- Parent: %s; main.py SHA-256 `%s`; config `%s`" % (e.get("parent"), e.get("main_sha256", "")[:16], json.dumps(e.get("config", {}), sort_keys=True)))
        for k, r in sorted(e.get("results", {}).items()):
            parts = []
            for gn in ("EuRoC", "TUM-VI", "ALL"):
                if gn in r: g = r[gn]; parts.append("%s %+.2f %% [%+.2f, %+.2f] (pts %+.3f; ate %+.1f, wp %+.1f, blind %+.1f, drop %+.1f %%; ms %.2f->%.2f)" % (gn, g["composite"], g["ci"][0], g["ci"][1], g["points"], g["ate"], g["wp"], g["b_abs"], g["drop"], g["ms"][0], g["ms"][1]))
            L.append("- %s: n %d, identical %d, new failures %d, decision changes %d; %s" % (k, r["n"], r["byte_identical"], r["new_failures"], r["decision_changes"], "; ".join(parts)))
        L.append("")
    if reg.get("analyses"):
        L += ["# Analyses and diagnostic experiments (not candidates)", ""]
        for aid, e in sorted(reg["analyses"].items()):
            L += ["## %s - %s" % (aid, e.get("title", "")), "", "- Status: **%s**" % e.get("status", ""), "- Question: %s" % e.get("question", ""), "- Method: %s" % e.get("method", ""),
                  "- Result: %s" % e.get("result", ""), "- Decision: %s" % e.get("decision", ""), "- Files: %s" % e.get("files", ""), ""]
    open(os.path.join(OPT, "REGISTRY.md"), "w").write("\n".join(L) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--sets", default="S42,DEV,CONF,STRESS"); b.add_argument("--procs", type=int, default=10)
    n = sub.add_parser("new"); n.add_argument("id"); n.add_argument("--src", required=True); n.add_argument("--config", default=""); n.add_argument("--title", required=True)
    n.add_argument("--hypothesis", required=True); n.add_argument("--parent", default="CHAMP-1.0.0")
    r = sub.add_parser("run"); r.add_argument("id"); r.add_argument("--set", required=True); r.add_argument("--procs", type=int, default=11); r.add_argument("--keep", action="store_true"); r.add_argument("--seqs", default="")
    c = sub.add_parser("compare"); c.add_argument("id"); c.add_argument("--set", required=True); c.add_argument("--ref", default="CHAMP-1.0.0")
    d = sub.add_parser("decide"); d.add_argument("id"); d.add_argument("--status", required=True); d.add_argument("--reason", required=True)
    sub.add_parser("registry")
    an = sub.add_parser("analysis"); an.add_argument("id"); an.add_argument("--title", required=True); an.add_argument("--status", required=True); an.add_argument("--question", required=True)
    an.add_argument("--method", required=True); an.add_argument("--result", required=True); an.add_argument("--decision", required=True); an.add_argument("--files", default="")
    a = ap.parse_args()
    if a.cmd == "build": build(a)
    elif a.cmd == "new": new(a)
    elif a.cmd == "run": run(a)
    elif a.cmd == "compare": compare(a)
    elif a.cmd == "decide": decide(a)
    elif a.cmd == "analysis":
        with reg_txn() as reg: reg.setdefault("analyses", {})[a.id] = dict(title=a.title, status=a.status, question=a.question, method=a.method, result=a.result, decision=a.decision, files=a.files, date=datetime.datetime.now().isoformat(timespec="seconds"))
    else: write_registry_md(load_reg())
