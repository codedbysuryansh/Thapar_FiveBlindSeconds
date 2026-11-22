"""p3.py - Phase 3 mission construction and scoring shared by e19+ (RESEARCH INSTRUMENT, not the submission).
A mission = one public sequence cut at a failure index, corrupted by OUR simulator (the organiser's script is not public),
optionally re-encoded (NaN / zero / held / removed rows), followed by 5 blind seconds. Ground truth is used for scoring only."""
import os, json, numpy as np
import fbs4 as f
ROOT = f.fbs.ROOT
KD = json.load(open(os.path.join(ROOT, "research/results/e05_drag.json")))
def k_loso(n): return float(np.mean([v["fits"]["2.0"]["k"] for m, v in KD.items() if m != n]))
LEVELS = {"clean": {}, "gaps": dict(gaps=[2.0] * 3), "L1": dict(gn=0.01, an=0.1, gb=0.005, ab=0.05, gaps=[2.0] * 3, rot=True),
          "L2": dict(gn=0.05, an=0.5, gb=0.02, ab=0.2, gaps=[2.0] * 3, rot=True),
          "L3": dict(gn=0.1, an=1.0, gb=0.05, ab=0.5, gaps=[2.0] * 3, rot=True, shape="exp"),
          "gb05": dict(gb=0.05), "gbrw": dict(gb=0.05, bias_shape="rw"), "ab10": dict(ab=1.0), "noise": dict(gn=0.05, an=0.5)}
class M: pass
def seed_of(n, kf, seed): return 1000 * seed + 7 * kf + sum(n.encode()) % 997
def mission(n, lvl="clean", kf=1, seed=0, fault=None, enc="nan", cfg=None):
    """kf: 0 = failure 5 s before the end of ground truth, 1 = at 60 %, 2 = at 80 %, float in (0,1) = at that fraction.
    fault(t, w, a, rng) -> (w, a[, extra_invalid_mask]) is applied to the clean stream BEFORE the level's corruption."""
    m = M(); s = f.load(n); fs = 1 / np.median(np.diff(s.t)); nb = int(round(5 * fs)); oi = np.where(s.ok)[0]; t0, t1 = s.t[oi[0]], s.t[oi[-1]]
    tf = [t1 - 5.05, t0 + 0.6 * (t1 - t0), t0 + 0.8 * (t1 - t0)][kf] if isinstance(kf, int) else t0 + kf * (t1 - t0); iT = int(np.searchsorted(s.t, tf))
    while not s.ok[iT:iT + nb + 1].all() and iT > 2000: iT -= 40
    sl = slice(0, iT + 1); sd = seed_of(n, kf if isinstance(kf, int) else int(100 * kf), seed); w0, a0 = s.w[sl].copy(), s.a[sl].copy(); extra = None
    if fault is not None:
        r = fault(s.t[sl] - s.t[0], w0, a0, np.random.default_rng(sd + 77)); w0, a0 = r[0], r[1]; extra = r[2] if len(r) > 2 else None
    w, a, valid, meta = f.degrade(s.t[sl], w0, a0, LEVELS[lvl] if cfg is None else cfg, seed=sd); tn = s.tn[sl]; keep = np.arange(iT + 1)
    if extra is not None: valid = valid & ~extra; w[~valid] = np.nan; a[~valid] = np.nan
    if enc == "zero": w = np.where(valid[:, None], w, 0.0); a = np.where(valid[:, None], a, 0.0)
    elif enc == "held":
        last = np.maximum.accumulate(np.where(valid, np.arange(len(valid)), 0)); w, a = w[last], a[last]
    elif enc == "removed": keep = np.where(valid)[0]; tn, w, a = tn[keep], w[keep], a[keep]
    Mtot = iT + 1 + nb; m.n, m.s, m.iT, m.nb, m.M, m.tn, m.w, m.a, m.valid, m.meta, m.keep, m.fs, m.lvl, m.enc = n, s, iT, nb, Mtot, tn, w, a, valid, meta, keep, fs, lvl, enc
    m.t = s.t[:Mtot]; m.gp = s.gt_p[:Mtot]; m.gq = s.gt_q[:Mtot]; m.ok = s.ok[:Mtot]; m.okp = m.ok & (np.arange(Mtot) <= iT); m.b = np.arange(iT + 1, Mtot); m.kd = k_loso(n)
    kept = np.zeros(Mtot, bool); kept[keep] = True; cand = np.where(m.okp & kept)[0]; cp = np.searchsorted(s.t, np.linspace(t0 + 10, s.t[iT], 6)).clip(oi[0], iT)
    m.cp = np.array([cand[np.argmin(np.abs(cand - c))] for c in cp]); m.kept = kept; m.rotor_true = not n.startswith("room"); return m
KEYS = ["wp", "b_abs", "b_anc", "b_end", "gap", "rec", "ate", "rpe", "ate_sim3", "scale", "tilt", "ori", "ori_b", "ate_all", "rpe_t", "rpe_r", "rpe_t_pre", "rpe_r_pre"]
def qyaw(psi): return np.array([0.0, 0.0, np.sin(psi / 2), np.cos(psi / 2)])
def score(m, p, q=None):
    """p (and q): one row per INPUT row followed by nb blind rows. SE(3) Umeyama alignment on the pre-failure rows with
    valid ground truth (the blind rows never enter the alignment). Returns a dict over KEYS (NaN where undefined)."""
    P = np.full((m.M, 3), np.nan); nk = len(m.keep); P[m.keep] = p[:nk]; P[m.iT + 1:] = p[nk:nk + m.nb]; fin = np.isfinite(P).all(1); A = m.okp & fin; out = dict.fromkeys(KEYS, float("nan"))
    if A.sum() < 10 or not np.isfinite(p).all(): return out
    Rm, tt = f.umeyama(P[A], m.gp[A]); er = np.linalg.norm(P @ Rm.T + tt - m.gp, axis=1); t = m.t; b = m.b
    out["wp"] = float(er[m.cp].mean()); out["b_abs"] = float(er[b].mean()); out["b_end"] = float(er[b[-1]]); out["ate"] = float(np.sqrt(np.mean(er[A] ** 2)))
    out["b_anc"] = float(np.linalg.norm((P[b] - P[m.iT]) @ Rm.T - (m.gp[b] - m.gp[m.iT]), axis=1).mean()) if fin[m.iT] else float("nan")
    ga = [er[(t >= g0) & (t < g0 + d) & A].mean() for g0, d in m.meta["gaps"] if ((t >= g0) & (t < g0 + d) & A).sum() > 10]; out["gap"] = float(np.mean(ga)) if ga else float("nan")
    rc = [er[(t >= g0 + d) & (t < g0 + d + 1.0) & A].mean() for g0, d in m.meta["gaps"] if ((t >= g0 + d) & (t < g0 + d + 1.0) & A).sum() > 10]; out["rec"] = float(np.mean(rc)) if rc else float("nan")
    out["rpe"] = float(f.rpe_disp(t[A], P[A], m.gp[A], Rm)); x = P[A] - P[A].mean(0); y = m.gp[A] - m.gp[A].mean(0); sc = float(np.sum((x @ Rm.T) * y) / max(np.sum(x * x), 1e-12)); out["scale"] = sc
    out["ate_sim3"] = float(np.sqrt(np.mean(np.sum((sc * x @ Rm.T - y) ** 2, 1))))
    if q is not None:
        Q = np.full((m.M, 4), np.nan); Q[m.keep] = q[:nk]; Q[m.iT + 1:] = q[nk:nk + m.nb]; qo = f.qmul(Q, f.Rot.from_matrix(m.meta["Rr"]).as_quat()[None]); B = A & np.isfinite(Q).all(1)
        out["tilt"] = float(np.mean(f.tilt_deg(qo[B], m.gq[B]))); dq = f.qmul(m.gq[B], f.qinv(qo[B])); xr = f.qrot(dq, np.array([1.0, 0, 0])); psi = np.arctan2(xr[:, 1], xr[:, 0]); psi0 = float(np.arctan2(np.sin(psi).mean(), np.cos(psi).mean()))
        out["ori"] = float(np.mean(f.ang_deg(f.qmul(qyaw(psi0)[None], qo[B]), m.gq[B]))); out["ori_b"] = float(np.mean(f.ang_deg(f.qmul(qyaw(psi0)[None], qo[b]), m.gq[b])))
        # evo-style quantities on the WHOLE file (pre-failure + blind rows with ground truth): ATE after SE(3) alignment on all of them,
        # RPE over 1 s of frames (consecutive pairs), translation part [m] and rotation angle [deg], as evo_rpe -d <1 s of frames> -u f
        C = m.ok & fin & np.isfinite(Q).all(1); idx = np.where(C)[0]; R2, t2 = f.umeyama(P[C], m.gp[C]); out["ate_all"] = float(np.sqrt(np.mean(np.sum((P[C] @ R2.T + t2 - m.gp[C]) ** 2, 1)))); d = int(round(m.fs)); i0 = idx[:-d:d]; i1 = idx[d::d]
        if len(i0) > 2:
            Re = f.Rot.from_quat(qo[C] / np.linalg.norm(qo[C], axis=1, keepdims=True)); Rg = f.Rot.from_quat(m.gq[C]); k0 = np.arange(0, len(idx) - d, d); k1 = k0 + d; pe, pg = P[C], m.gp[C]
            te = Re[k0].inv().apply(pe[k1] - pe[k0]); tg = Rg[k0].inv().apply(pg[k1] - pg[k0]); dRe = Re[k0].inv() * Re[k1]; dRg = Rg[k0].inv() * Rg[k1]; E = dRg.inv() * dRe; et = np.linalg.norm(dRg.inv().apply(te - tg), axis=1); er_ = np.degrees(E.magnitude()); pre = idx[k1] <= m.iT
            out["rpe_t"] = float(np.sqrt(np.mean(et ** 2))); out["rpe_r"] = float(np.sqrt(np.mean(er_ ** 2))); out["rpe_t_pre"] = float(np.sqrt(np.mean(et[pre] ** 2))) if pre.any() else float("nan"); out["rpe_r_pre"] = float(np.sqrt(np.mean(er_[pre] ** 2))) if pre.any() else float("nan")
    return out
def v4(m, **kw):
    e = f.estimate_v4(m.tn, m.w, m.a, n_blind=m.nb, k_drag=m.kd, **kw); return e["p"], e["q"], e
def table(res, variants, levels, keys=("wp", "b_abs", "b_anc", "gap", "ate", "rpe"), groups=(("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room")), ("ALL", lambda n: True))):
    """res: list of dict(name, lvl, m={variant: score dict}). Prints mean of each key per group / level / variant."""
    for lvl in levels:
        for gname, sel in groups:
            rr = [r for r in res if r["lvl"] == lvl and sel(r["name"])]
            if not rr: continue
            print("\n[%s | %s | %d missions]" % (lvl, gname, len(rr)))
            for v in variants:
                if v not in rr[0]["m"]: continue
                print("  %-22s " % v + "  ".join("%s %6.2f" % (k, np.nanmean([r["m"][v][k] for r in rr])) for k in keys) + "  fail %d" % sum(not np.isfinite(r["m"][v]["ate"]) for r in rr))
