"""p4.py - OFFICIAL missions: a public sequence degraded by the organisers' degrade_imu.py (research/official, unmodified),
fed to the estimator in EuRoC column order, scored as the Track 3 FAQ describes (evo-style SE(3) Umeyama alignment over ALL
poses, the blind rows included). Column order of the script's INPUT: 'A' = as the script's docstring says (timestamp, accel,
gyro); 'B' = a EuRoC imu0/data.csv passed as is (timestamp, gyro, accel), in which case the script treats the gyro columns as
accelerometer columns. Which one the organisers used is unknown, so both are evaluated. Ground truth: scoring only."""
import os, sys, json, importlib.util, numpy as np, pandas as pd
import fbs4 as f
ROOT = f.fbs.ROOT
_spec = importlib.util.spec_from_file_location("degrade_imu", os.path.join(ROOT, "research/official/degrade_imu.py")); DEG = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(DEG)
sys.path.insert(0, os.path.join(ROOT, "research/official"))
from degrade_masks import degrade_masks  # noqa: E402
EUROC_COLS = ["#timestamp [ns]", "w_RS_S_x [rad s^-1]", "w_RS_S_y [rad s^-1]", "w_RS_S_z [rad s^-1]", "a_RS_S_x [m s^-2]", "a_RS_S_y [m s^-2]", "a_RS_S_z [m s^-2]"]
DOC_COLS = ["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
KD = json.load(open(os.path.join(ROOT, "research/results/e05_drag.json")))
def k_loso(n): return float(np.mean([v["fits"]["2.0"]["k"] for m, v in KD.items() if m != n]))
class OM: pass
def official(n, order="B", seed=42, max_level=1.0, gamma=1.7, crop=1.0, masks=False, start=0.0):
    s = f.load(n); end = len(s.tn) if crop >= 1.0 else int(np.searchsorted(s.t, s.t[0] + crop * (s.t[-1] - s.t[0])))
    beg = 0 if start <= 0.0 else int(np.searchsorted(s.t, s.t[0] + start * (s.t[-1] - s.t[0])))
    tn, w, a = s.tn[beg:end], s.w[beg:end], s.a[beg:end]
    if order == "A": df = pd.DataFrame(dict(zip(DOC_COLS, [tn, a[:, 0], a[:, 1], a[:, 2], w[:, 0], w[:, 1], w[:, 2]])))
    else: df = pd.DataFrame(dict(zip(EUROC_COLS, [tn, w[:, 0], w[:, 1], w[:, 2], a[:, 0], a[:, 1], a[:, 2]])))
    cfg = dict(DEG.CONFIG); cfg["max_level"] = max_level; cfg["gamma"] = gamma
    out, st = DEG.degrade(df, cfg, seed)
    mk = degrade_masks(df, cfg, seed, orig=DEG)[1] if masks else None
    tno = out.iloc[:, 0].to_numpy(np.int64)
    if order == "A": a_o = out.iloc[:, 1:4].to_numpy(float); w_o = out.iloc[:, 4:7].to_numpy(float)
    else: w_o = out.iloc[:, 1:4].to_numpy(float); a_o = out.iloc[:, 4:7].to_numpy(float)
    idx = np.searchsorted(s.tn, tno); assert (s.tn[idx] == tno).all()
    T = (tn[-1] - tn[0]) * 1e-9; tsec = (tno - tn[0]) * 1e-9; m = OM()
    m.n, m.order, m.seed, m.max_level, m.gamma, m.crop, m.s, m.tn, m.w, m.a, m.idx, m.stats, m.masks, m.out = n, order, seed, max_level, gamma, crop, s, tno, w_o, a_o, idx, st, mk, out
    m.t = tsec; m.T = T; m.blind = tsec >= T - 5.0; m.gp = s.gt_p[idx]; m.gq = s.gt_q[idx]; m.ok = s.ok[idx].copy(); m.kd = k_loso(n); m.rotor_true = not n.startswith("room")
    d = np.diff(tno); med = np.median(d[d > 0]); miss = np.round(d / med).astype(int) - 1; m.gap_after = np.zeros(len(tno), int); m.gap_after[1:] = np.maximum(miss, 0)   # samples missing just before each row
    m.fs = 1e9 / med; m.df = df; m.cfg = cfg
    return m
KEYS = ["ate", "wp", "b_abs", "b_end", "b_anc", "pre_ate", "drop", "rpe1_t", "rpe1_r", "rpe_t", "rpe_r", "ori", "ori_b", "ate_prealign", "wp_prealign", "b_abs_prealign"]
def score(m, p, q=None, ncp=8):
    out = dict.fromkeys(KEYS, float("nan")); p = np.asarray(p, float); fin = np.isfinite(p).all(1); A = m.ok & fin
    if A.sum() < 10: return out
    Rm, tt = f.umeyama(p[A], m.gp[A]); er = np.linalg.norm(p @ Rm.T + tt - m.gp, axis=1); out["ate"] = float(np.sqrt(np.mean(er[A] ** 2)))
    pre = A & ~m.blind; b = A & m.blind; Tp = m.T - 5.0; ci = [int(np.argmin(np.abs(np.where(pre, m.t, 1e9) - Tp * (k + 1) / (ncp + 1)))) for k in range(ncp)]
    out["wp"] = float(er[ci].mean()); out["pre_ate"] = float(np.sqrt(np.mean(er[pre] ** 2)))
    if b.any():
        bi = np.where(b)[0]; L = int(np.where(pre)[0][-1]); out["b_abs"] = float(er[bi].mean()); out["b_end"] = float(er[bi[-1]])
        out["b_anc"] = float(np.linalg.norm((p[bi] - p[L]) @ Rm.T - (m.gp[bi] - m.gp[L]), axis=1).mean())
    g = np.where(m.gap_after >= 5)[0]; near = np.zeros(len(p), bool)
    for i in g: near[i:np.searchsorted(m.t, m.t[i] + 1.0)] = True
    near &= pre; out["drop"] = float(er[near].mean()) if near.any() else float("nan")
    R2, t2 = f.umeyama(p[pre], m.gp[pre]); e2 = np.linalg.norm(p @ R2.T + t2 - m.gp, axis=1); out["ate_prealign"] = float(np.sqrt(np.mean(e2[pre] ** 2))); out["wp_prealign"] = float(e2[ci].mean())
    out["b_abs_prealign"] = float(e2[b].mean()) if b.any() else float("nan")
    if q is not None:
        q = np.asarray(q, float); q = q / np.linalg.norm(q, axis=1, keepdims=True); idx = np.where(A)[0]
        Re = f.Rot.from_quat(q[idx]); Rg = f.Rot.from_quat(m.gq[idx]); Ra = f.Rot.from_matrix(Rm); E = (Ra * Re).inv() * Rg; ang = np.degrees(E.magnitude())
        out["ori"] = float(np.sqrt(np.mean(ang ** 2))); bb = m.blind[idx]; out["ori_b"] = float(np.sqrt(np.mean(ang[bb] ** 2))) if bb.any() else float("nan")
        for d, kt, kr in ((1, "rpe1_t", "rpe1_r"), (int(round(m.fs)), "rpe_t", "rpe_r")):
            k0 = np.arange(0, len(idx) - d, d); k1 = k0 + d; pe, pg = p[idx], m.gp[idx]; te = Re[k0].inv().apply(pe[k1] - pe[k0]); tg = Rg[k0].inv().apply(pg[k1] - pg[k0])
            dRg = Rg[k0].inv() * Rg[k1]; dRe = Re[k0].inv() * Re[k1]; out[kt] = float(np.sqrt(np.mean(np.sum(dRg.inv().apply(te - tg) ** 2, 1)))); out[kr] = float(np.sqrt(np.mean(np.degrees((dRg.inv() * dRe).magnitude()) ** 2)))
    return out
def table(res, variants, keys=("ate", "wp", "b_abs", "b_anc", "drop", "rpe_t", "ori"), by=("order",), groups=(("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room")))):
    import itertools
    vals = {k: sorted({r[k] for r in res}, key=str) for k in by}
    for combo in itertools.product(*[vals[k] for k in by]):
        for gname, sel in groups:
            rr = [r for r in res if sel(r["name"]) and all(r[k] == v for k, v in zip(by, combo))]
            if not rr: continue
            print("\n[%s | %s | %d missions]" % (" ".join("%s=%s" % (k, v) for k, v in zip(by, combo)), gname, len(rr)))
            for v in variants:
                if v not in rr[0]["m"]: continue
                print("  %-26s " % v + "  ".join("%s %6.3f" % (k, np.nanmean([r["m"][v][k] for r in rr])) for k in keys) + "  fail %d" % sum(not np.isfinite(r["m"][v]["ate"]) for r in rr))
