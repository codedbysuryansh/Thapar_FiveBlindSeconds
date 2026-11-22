"""fbs6.py - pipeline pieces for the OFFICIAL input (Track 3 FAQ + degrade_imu.py, received 21 Nov 2026). RESEARCH INSTRUMENT.
Facts used: dropouts are removed rows; the 5 blind seconds are rows of exact zeros at the end; stuck samples repeat the previous
row with an advancing timestamp; duplicated rows repeat the previous row with the SAME timestamp; spikes hit accelerometer and
gyro on the same row. Clean EuRoC / TUM-VI never repeat a 6-channel row exactly and never repeat a timestamp (E40)."""
import numpy as np
from fbs5 import *  # noqa: F401,F403
import fbs5, fbs4, fbs3

def ingest2(tn, w, a, held_run=2, dedup=True):
    """Uniform timeline. dedup: rows with the same timestamp share one slot. held_run=2: every exact repeat of the previous slot
    (advancing timestamp) is invalid, the first sample of the run is kept. Zero rows and NaN rows are invalid."""
    tn = np.asarray(tn, np.int64); X = np.hstack([w, a]).astype(float); d = np.diff(tn); med = float(np.median(d[d > 0]))
    st = np.maximum(np.round(d / med).astype(np.int64), 0 if dedup else 1); rows = np.concatenate([[0], np.cumsum(st)]); N = int(rows[-1]) + 1
    ur, first = np.unique(rows, return_index=True); t = np.interp(np.arange(N), ur, (tn[first] - tn[0]) * 1e-9)
    Xu = np.full((N, 6), np.nan); Xu[rows] = X; bad = np.isnan(Xu).any(1) | (Xu == 0).all(1); same = np.zeros(N, bool); same[1:] = (Xu[1:] == Xu[:-1]).all(1) & ~bad[1:]
    run = np.zeros(N, np.int64)
    for i in np.where(same)[0]: run[i] = run[i - 1] + 1
    held = np.zeros(N, bool); i = N - 1
    while i > 0:
        if run[i] > 0 and run[i] >= held_run - 1: j = i - run[i]; held[j + 1:i + 1] = True; i = j
        else: i -= 1
    valid = ~(bad | held); Xu[~valid] = np.nan
    return t, Xu[:, :3], Xu[:, 3:], valid, rows
def despike2(t, w, a, valid, z=5.0, rule="triple", floor=(2e-3, 2e-2)):
    """Single-sample outliers against a 5-sample median. Scale per channel and 1 s block = max(1.4826 median|r|, 0.5 mean|r|, floor).
    rule 'triple': all three axes of either sensor exceed z; 'k4': at least 4 of the 6 channels exceed z. Flagged samples are
    replaced by linear interpolation of their valid neighbours and stay valid."""
    wf = fill_gaps(t, w, valid); af = fill_gaps(t, a, valid); X = np.hstack([wf, af]); N = len(t); pad = np.pad(X, ((2, 2), (0, 0)), mode="edge")
    med = np.median(np.stack([pad[i:i + N] for i in range(5)]), 0); r = np.abs(X - med); fs = 1.0 / np.median(np.diff(t)); n1 = max(5, int(fs)); m = max(1, N // n1)
    rb = r[:m * n1].reshape(m, n1, 6); sig = np.maximum(np.maximum(1.4826 * np.median(rb, 1), 0.5 * rb.mean(1)), np.array([floor[0]] * 3 + [floor[1]] * 3)); sig = np.repeat(sig, n1, 0); sig = np.vstack([sig, np.repeat(sig[-1:], N - len(sig), 0)])
    zz = r / sig; sp = ((zz[:, :3] > z).all(1) | (zz[:, 3:] > z).all(1)) if rule == "triple" else ((zz > z).sum(1) >= 4); sp &= valid; ns = int(sp.sum())
    if ns:
        good = valid & ~sp; w = w.copy(); a = a.copy()
        for j in range(3): w[sp, j] = np.interp(t[sp], t[good], w[good, j]); a[sp, j] = np.interp(t[sp], t[good], a[good, j])
    return w, a, ns

# ------------------------------------------------------------------ gated front end (v5): constant / anchored bias unless the drift test or the health test fires
HEALTH_AT = "final"
def attitude5b(t, w, a, valid, tau=3.0, bg_track=None, gsmooth=True, bridge=True, fit="const", ridge=1e-1, win=60.0, knot=20.0, iters=2):
    """attitude5 with the OU gyro bridge; also returns the health value: 1st percentile of |g_I|/g, on the first pass (HEALTH_AT='first',
    E41) or on the final pass after the bias fit (HEALTH_AT='final': a constant bias that the fit removes no longer looks unhealthy)."""
    fs = 1.0 / np.median(np.diff(t)); ez = np.array([0, 0, 1.0]); N = len(t); af = fill_gaps(t, a, valid); wf = bridge_gyro(t, w, valid) if bridge else fill_gaps(t, w, valid); w_rms = float(np.sqrt(np.mean(np.sum(np.nan_to_num(w[valid]) ** 2, 1))))
    seg = np.cumsum(np.concatenate([[0], np.diff(valid.astype(np.int8)) != 0])); bg = np.zeros((N, 3)) if bg_track is None else bg_track.copy(); n_it = iters if fit else 0; hmin = 1.0
    for it in range(n_it + 1):
        qg = integrate_gyro(t, wf - bg); x = qrot(qg, af); gI = gravity_smooth(t, x, valid, tau, w_rms) if gsmooth else lowpass_seg(x, fs, 1.0 / (2 * np.pi * tau), valid)
        gn = np.linalg.norm(gI, axis=1, keepdims=True)
        if (it == 0 and HEALTH_AT == "first") or it == n_it: hmin = float(np.percentile(gn[valid], 1) / 9.81)
        gI = gI / gn; qc = shortest_arc(gI, np.broadcast_to(ez, gI.shape)); q = qmul(qc, qg); q /= np.linalg.norm(q, axis=1, keepdims=True)
        if it == n_it: break
        step = max(1, int(fs * 1.0)); wcr = qlog(qmul(qc[step:], qinv(qc[:-step]))) / (t[step:] - t[:-step])[:, None]; ii = np.where(valid[:-step] & valid[step:] & (seg[:-step] == seg[step:]))[0][::10]
        if len(ii) < 50: break
        A = Rot.from_quat(q[ii]).as_matrix()[:, :2, :]; y = -wcr[ii, :2]
        if fit == "const":
            A2 = A.reshape(-1, 3); bg = bg + np.linalg.solve(A2.T @ A2 + len(A2) * ridge * np.eye(3), A2.T @ y.reshape(-1))
        else:
            tm = t[ii] + 0.5 * step / fs; knots = np.arange(t[0], t[-1] + knot, knot); db = np.zeros((len(knots), 3))
            for j, tk in enumerate(knots):
                wg = np.clip(1 - np.abs(tm - tk) / win, 0, None); sel = wg > 0
                if sel.sum() < 20: continue
                Aw = A[sel] * wg[sel][:, None, None]; db[j] = np.linalg.solve(np.einsum("nij,nik->jk", Aw, A[sel]) + 2 * wg[sel].sum() * ridge * np.eye(3), np.einsum("nij,ni->j", Aw, y[sel]))
            bg = bg + np.stack([np.interp(t, knots, db[:, i]) for i in range(3)], 1)
    return q, bg, hmin
def fe_const(t, wu, au, valid, gsm, br, tau=3.0):
    anc = rest_anchors(t, wu, au, valid); base = bias_track(t, anc)
    if base is None: q3, b3, h = attitude5b(t, wu, au, valid, tau, gsmooth=gsm, bridge=br, fit="const"); base = np.broadcast_to(b3, (len(t), 3)).copy()
    else: q3, b3, h = attitude5b(t, wu, au, valid, tau, bg_track=base, gsmooth=gsm, bridge=br, fit=None)
    return q3, base, h, bool(anc)
def fe_drift(t, wu, au, valid, gsm, br, base, tau=3.0):
    q4, b4, _ = attitude5b(t, wu, au, valid, tau, bg_track=base, gsmooth=gsm, bridge=br, fit="knots", win=60.0, knot=20.0); d = b4 - base; return float(np.sqrt(((d - d.mean(0)) ** 2).sum(1).mean()))
def fe_spline(t, wu, au, valid, gsm, br, lam2, base, h, tau=3.0, knot=20.0, health_thr=0.9, coarse=True):
    if coarse and h < health_thr:
        wb = bridge_gyro(t, wu, valid) if br else fill_gaps(t, wu, valid); bc, _, _ = coarse_bias(t, wb, fill_gaps(t, au, valid), valid)
        qs, bs = attitude6(t, wu, au, valid, tau, bg0=bc, knot=knot, lam2=lam2, iters=10, gsmooth=gsm, bridge=br); return qs, bs, True
    qs, bs = attitude6(t, wu, au, valid, tau, bg0=base, knot=knot, lam2=lam2, gsmooth=gsm, bridge=br); return qs, bs, False
def thrust_lock_fill(t, af, valid, win=10.0):
    """thrust_lock_local with gap filling before ONE zero-phase low-pass over the whole stream. The per-segment low-pass of v4
    breaks when single-sample dropouts are frequent (official data): EuRoC values rose from <=0.027 to 0.045-0.063 (E41/E42)."""
    fs = 1.0 / np.median(np.diff(t)); n = int(win * fs); al = lowpass(fill_gaps(t, af, valid), fs, 1.0); out = []
    for i in range(0, len(t) - n + 1, n // 2):
        v = valid[i:i + n]
        if v.sum() < n // 2: continue
        x = al[i:i + n][v]; zb = x.mean(0); zb = zb / np.linalg.norm(zb); perp = x - np.outer(x @ zb, zb); out.append(np.sqrt((perp ** 2).sum(1).mean()) / G)
    return float(np.median(out)) if out else 1.0
LOCK = "fill"
def common_of(t, wu, au, valid):
    af = fill_gaps(t, au, valid); lk = thrust_lock_fill(t, af, valid) if LOCK == "fill" else thrust_lock_local(t, af, valid)
    return dict(lock=lk, zmask=rest_info(t, wu, au, valid)[1])
def prep_from(t, wu, au, valid, q, bg, br, common):
    fs = 1.0 / np.median(np.diff(t)); af = fill_gaps(t, au, valid); wc = (bridge_gyro(t, wu, valid) if br else fill_gaps(t, wu, valid)) - bg; wl = lowpass(wc, fs, 15.0); aw = qrot(q, af); aw = aw - aw[valid].mean(0)
    return dict(t=t, q=q, bg=bg, af=af, valid=valid, rows=np.arange(len(t)), lock=common["lock"], zmask=common["zmask"], wc=wc, wl=wl, aw=aw, ww=qrot(q, wc), fs=fs, dtn=1.0 / fs, D=float("nan"), drift=False, N0=len(t))
def split_tail(t, wu, au, valid):
    """The trailing run of invalid slots is the blind window: it is cut off before the front end and predicted by the back end."""
    L = int(np.where(valid)[0][-1]); return L, len(t) - 1 - L
def estimate_off(tn, w, a, k_drag=None, ing=None, ds=None, rep=True, fe="v5", thr=DRIFT_THR, gs="auto", br=True, lam2=3.0, coarse=True, be=None, both=False):
    """End-to-end on one official input (one output row per input row). fe: 'v5' (gated), 'v4' (front5 of Phase 2), 'const', 'spline'.
    both=True also returns the const and spline solutions for threshold sweeps (keys pC, qC, pS, qS)."""
    ing = dict(held_run=2, dedup=True) if ing is None else ing; t, wu, au, valid, rows = ingest2(tn, w, a, **ing); L, nt = split_tail(t, wu, au, valid)
    t0, wu0, au0, v0 = t[:L + 1], wu[:L + 1], au[:L + 1], valid[:L + 1]; info = dict(n_tail=nt, spikes=0, implausible=0)
    if ds is not None: wu0, au0, info["spikes"] = despike2(t0, wu0, au0, v0, **ds)
    if rep: wu0, au0, v0, ri = repair(t0, wu0, au0, v0, g_lo=5.0, g_hi=15.5); info["implausible"] = ri["implausible"]; info["spikes_legacy"] = ri["spikes"]
    cm = common_of(t0, wu0, au0, v0); gsm = (not v0.all()) if gs == "auto" else bool(gs); sols = {}
    if fe == "v4":
        q, bg, quiet, af, D, used = fbs3.front5(t0, wu0, au0, v0); sols["F"] = (q, bg); info.update(D=D, drift=bool(used), coarse=False); pick = "F"; br_ = False
    else:
        br_ = br; q3, base, h, anc = fe_const(t0, wu0, au0, v0, gsm, br); info.update(h=h, anchors=anc, gs=gsm); sols["C"] = (q3, base)
        if fe != "const":
            if coarse and h < 0.9: D = float("inf")
            else: D = fe_drift(t0, wu0, au0, v0, gsm, br, base)
            if fe == "spline" or both or D > thr: qs, bs, co = fe_spline(t0, wu0, au0, v0, gsm, br, lam2, base, h, coarse=coarse); sols["S"] = (qs, bs); info["coarse"] = co
            info["D"] = D
        pick = "C" if fe == "const" else ("S" if fe == "spline" else ("S" if info.get("D", 0) > thr else "C"))
    out = {}
    for k, (q, bg) in sols.items():
        P = prep_from(t0, wu0, au0, v0, q, bg, br_, cm); e = fbs4.estimate_v6(None, None, None, n_blind=nt, k_drag=k_drag, P=P, **(be or {})); out[k] = (e["p"][rows], e["q"][rows], e)
    p, q, e = out[pick]; res = dict(p=p, q=q, rotor=bool(e["rotor"]), lock=float(e["lock"]), pick=pick, info=info)
    if both:
        for k in out: res["p" + k], res["q" + k] = out[k][0], out[k][1]
    return res

# ------------------------------------------------------------------ FROZEN Phase 3 candidate (v5): research reference for the Phase 4 Python implementation
V5 = dict(k2=0.25, gate=0.035, thr_hand=2.0e-3, health=0.9, lam2=3.0, knot=20.0, gap_ref=2.0, dec_rotor=4, dec_hand=2, guard=12.0, k_drag=K_DRAG)
def radius(p): return float(np.sqrt(np.mean(np.sum((p - p.mean(0)) ** 2, 1))))
V5P4 = dict(V5, dec_rotor=8, gs_dec=16)   # Phase 4 SUBMISSION configuration: gravity smoother 12.5 Hz, rotor smoother 25 Hz (E64 cost model, E65 accuracy)
# Phase 4 rate configuration (E63/E63b/E65): the IRW gravity smoother on block means of d samples (12.5 Hz for d = 16)
def gravity_smooth_dec(t, x, valid, tau=3.0, w_rms=None, sig_m=1.5, c_rot=0.5, d=8):
    N = len(t); dt = float(np.median(np.diff(t))); K = (N + d - 1) // d; pad = K * d - N
    jump = np.zeros(N); st = np.where(valid[1:] & ~valid[:-1])[0] + 1
    if len(st):
        en = np.where(~valid[1:] & valid[:-1])[0] + 1
        for s in st:
            prev = en[en < s]; T = (t[s] - t[prev[-1] - 1]) if len(prev) else 0.0
            th = c_rot * (w_rms if w_rms is not None else 0.5) * T if fbs5.GS_JUMP == "linear" else min(0.5 * fbs5.GS_K2 * T * T, 1.0); jump[s] = (9.81 * th / sig_m) ** 2
    xv = np.nan_to_num(x) * valid[:, None]; vv = valid.astype(float)
    if pad: xv = np.vstack([xv, np.zeros((pad, x.shape[1]))]); vv = np.concatenate([vv, np.zeros(pad)]); jump = np.concatenate([jump, np.zeros(pad)])
    cnt = vv.reshape(K, d).sum(1); xb = xv.reshape(K, d, -1).sum(1) / np.maximum(cnt, 1)[:, None]; jb = jump.reshape(K, d).sum(1); okb = cnt > 0
    tb = (np.arange(K) * d + (d - 1) / 2.0) * dt; q = dt / tau ** 4; dtb = d * dt; q11, q12, q22 = q * dtb ** 3 / 3, q * dtb ** 2 / 2, q * dtb
    out = np.empty((K, x.shape[1]))
    for c in range(x.shape[1]):
        m0 = m1 = 0.0; p00, p01, p11 = 1e6, 0.0, 1e2; first = True; XF = np.empty((K, 2)); PF = np.empty((K, 3)); PP = np.empty((K, 3))
        for i in range(K):
            if not first: m0 += dtb * m1; p00, p01, p11 = p00 + 2 * dtb * p01 + dtb * dtb * p11 + q11, p01 + dtb * p11 + q12, p11 + q22
            p00 += jb[i]; p11 += 0.01 * jb[i]; PP[i] = (p00, p01, p11); XP0, XP1 = m0, m1
            if okb[i]:
                if first: m0 = xb[i, c]; first = False
                s = p00 + 1.0 / cnt[i]; k0, k1 = p00 / s, p01 / s; r = xb[i, c] - m0; m0 += k0 * r; m1 += k1 * r; p00, p01, p11 = p00 - k0 * p00, p01 - k0 * p01, p11 - k1 * p01
            XF[i] = (m0, m1); PF[i] = (p00, p01, p11)
        s0, s1 = XF[-1]; out[-1, c] = s0
        for i in range(K - 2, -1, -1):
            f00, f01, f11 = PF[i]; b00, b01, b10, b11 = f00 + dtb * f01, f01, f01 + dtb * f11, f11; d00, d01, d11 = PP[i + 1]; det = d00 * d11 - d01 * d01
            c00 = (b00 * d11 - b01 * d01) / det; c01 = (-b00 * d01 + b01 * d00) / det; c10 = (b10 * d11 - b11 * d01) / det; c11 = (-b10 * d01 + b11 * d00) / det
            xp0 = XF[i, 0] + dtb * XF[i, 1]; xp1 = XF[i, 1]; e0 = s0 - xp0; e1 = s1 - xp1; s0 = XF[i, 0] + c00 * e0 + c01 * e1; s1 = XF[i, 1] + c10 * e0 + c11 * e1; out[i, c] = s0
    tf = np.arange(N) * dt; return np.stack([np.interp(tf, tb, out[:, c]) for c in range(x.shape[1])], 1)
def front_final(tn, w, a, cfg=None):
    """v5 front end on one official input: ingest2 -> blind tail split -> plausibility + legacy despike -> fill-lock platform gate ->
    attitude (rotor: spline bias; hand-held: constant / anchored bias unless the drift or health test fires) -> back-end inputs P.
    Ablation switches (defaults = frozen v5): dedup, rep (repair), gsm (IRW gravity smoother), br (OU gyro bridge), rotor_spline,
    drift_test, jump ('quad' = T^2 rotation-uncertainty law, 'linear' = Phase 2 law)."""
    c = dict(V5, **(cfg or {})); old = fbs5.GS_K2, fbs5.GS_JUMP, fbs5.gravity_smooth, globals()["gravity_smooth"]; fbs5.GS_K2 = c["k2"]; fbs5.GS_JUMP = c.get("jump", "quad")
    if c.get("gs_dec"):
        _g = lambda *a_, **kw_: gravity_smooth_dec(*a_, d=c["gs_dec"], **kw_); fbs5.gravity_smooth = _g; globals()["gravity_smooth"] = _g
    gsm, br = c.get("gsm", True), c.get("br", True)
    try:
        t, wu, au, valid, rows = ingest2(tn, w, a, dedup=c.get("dedup", True)); L, nt = split_tail(t, wu, au, valid); t0, wu0, au0, v0 = t[:L + 1], wu[:L + 1], au[:L + 1], valid[:L + 1]
        if c.get("rep", True): wu0, au0, v0, ri = repair(t0, wu0, au0, v0, g_lo=5.0, g_hi=15.5)
        else: ri = dict(implausible=0, spikes=0)
        cm = common_of(t0, wu0, au0, v0); rotor = cm["lock"] < c["gate"]
        q3, base, h, anc = fe_const(t0, wu0, au0, v0, gsm, br); D = float("nan"); co = False
        if rotor and c.get("rotor_spline", True): q, bg, co = fe_spline(t0, wu0, au0, v0, gsm, br, c["lam2"], base, h, knot=c["knot"], health_thr=c["health"]); mode = "spline"
        elif rotor: q, bg, mode = q3, base, "const"
        else:
            D = (float("inf") if h < c["health"] else fe_drift(t0, wu0, au0, v0, gsm, br, base)) if c.get("drift_test", True) else 0.0
            if D > c["thr_hand"]: q, bg, co = fe_spline(t0, wu0, au0, v0, gsm, br, c["lam2"], base, h, knot=c["knot"], health_thr=c["health"]); mode = "spline"
            else: q, bg, mode = q3, base, "const"
        P = prep_from(t0, wu0, au0, v0, q, bg, br, cm)
        return dict(P=P, rows=rows, L=L, nt=nt, rotor=bool(rotor), lock=cm["lock"], mode=mode, D=D, h=h, coarse=bool(co), implausible=ri["implausible"], spikes_legacy=ri["spikes"], cfg=c)
    finally: fbs5.GS_K2, fbs5.GS_JUMP, fbs5.gravity_smooth, globals()["gravity_smooth"] = old
def back_final(F, k_drag=None, blind=None, want=False, vel=None):
    """v5 back end: rotor 17-state spring + drag + lever + ZUPT at dec 4; hand-held per-axis Matern + ZUPT at dec 2; accel-bias variance at
    gaps scaled by T / 2 s; guard (rotor radius > 12 x hand-held radius -> hand-held); blind rows (blind=None: smoother prediction with the
    orientation held; or a callable(e, P, nt) -> (dp (nt,3), q (nt,4))). vel = optional (centres, velocities, sigma) pseudo-measurements."""
    c = F["cfg"]; P, rows, L, nt, rotor = F["P"], F["rows"], F["L"], F["nt"], F["rotor"]; kd = c["k_drag"] if k_drag is None else k_drag; be = dict(gap_ref=c["gap_ref"])
    if not c.get("zupt", True): be["zmask"] = None
    if not c.get("lever", True): be["wl"] = None
    if not c.get("drag", True): kd = 0.0
    fl = want or blind is not None
    eh = fbs4.estimate_v6(None, None, None, n_blind=nt, k_drag=kd, P=P, force="matern", dec=c["dec_hand"], full=fl, vel=vel, **{k: v for k, v in be.items() if k != "wl"}); e = eh; guard = False
    if rotor:
        er = fbs4.estimate_v6(None, None, None, n_blind=nt, k_drag=kd, P=P, force="drag", dec=c["dec_rotor"], full=fl, vel=vel, **be)
        if radius(er["p"]) > c["guard"] * max(radius(eh["p"]), 1e-6): guard = True
        else: e = er
    p, qq = e["p"].copy(), e["q"].copy()
    if blind is not None and nt > 0: dp, qb = blind(e, P, nt); p[L + 1:] = p[L] + dp; qq[L + 1:] = qb
    return dict(p=p[rows], q=qq[rows], rotor=bool(rotor and not guard), lock=F["lock"], guard=guard, mode=F["mode"], D=F["D"], h=F["h"], coarse=F["coarse"], n_tail=nt, implausible=F["implausible"], spikes_legacy=F["spikes_legacy"], e=e if want else None, P=P if want else None, L=L, rows=rows if want else None)
def estimate_final(tn, w, a, k_drag=None, cfg=None, blind=None, want=False, vel=None):
    """v5 end to end (front_final -> back_final); one output row per input row (duplicated and blind rows included)."""
    return back_final(front_final(tn, w, a, cfg), k_drag=k_drag, blind=blind, want=want, vel=vel)
