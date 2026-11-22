"""fbs4.py - Phase 3 research harness. Builds on fbs3 (frozen at the Phase 2 state). RESEARCH INSTRUMENT, not the submission.
Adds: v4 internals exposed (prep_v4), a joint smoother with optional velocity pseudo-measurement rows (run_joint7) so that
learned or other velocity sources can be fused under the identical pipeline, and estimate_v5 (v4 + optional rows)."""
import numpy as np
from fbs3 import *  # noqa: F401,F403
import fbs3, fbs

def prep_v4(tn, w, a):
    """Everything v4 computes before the position smoother, for one raw stream (no blind rows yet)."""
    t, wu, au, valid, rows = ingest(tn, w, a); q, bg, quiet, af, D, drift = front5(t, wu, au, valid); lock = thrust_lock_local(t, af, valid)
    dtn = float(np.median(np.diff(t))); fs = 1.0 / dtn; zmask = rest_info(t, wu, au, valid)[1]; wc = fill_gaps(t, wu, valid) - bg; wl = lowpass(wc, fs, 15.0)
    aw = qrot(q, af); aw = aw - aw[valid].mean(0)
    return dict(t=t, q=q, bg=bg, af=af, valid=valid, rows=rows, lock=lock, zmask=zmask, wc=wc, wl=wl, aw=aw, ww=qrot(q, wc), fs=fs, dtn=dtn, D=D, drift=drift, N0=len(t))
def run_joint7(t, q, af, model, valid, k_drag=0.0, sd=SD_DRAG, zmask=None, sz=0.02, wl=None, vel=None, lever_prior=0.15, bias_jump=0.5, tau_c=100.0, sc=0.15):
    """run_joint6 plus velocity pseudo-measurement rows: vel = (row indices, v (M,3) in the estimator's world frame, sigma scalar or (M,))."""
    F4, Qc4, P04, H4 = model; lev = wl is not None and k_drag > 0; n = 17 if lev else 14; N = len(t); F = np.zeros((n, n)); Qc = np.zeros((n, n)); P0 = np.zeros((n, n))
    for j in range(3):
        sl = slice(4 * j, 4 * j + 4); F[sl, sl] = F4; Qc[sl, sl] = Qc4; P0[sl, sl] = P04
    for j in (12, 13): F[j, j] = -1.0 / tau_c; Qc[j, j] = 2 * sc ** 2 / tau_c; P0[j, j] = sc ** 2
    if lev:
        for j in (14, 15, 16): P0[j, j] = lever_prior ** 2
    A, Q, idx = discretise(F, Qc, t); m = 11 if vel is not None else 8; H = np.zeros((N, m, n)); y = np.zeros((N, m)); R = np.ones((N, m)); mask = np.zeros((N, m), np.bool_)
    aw = qrot(q, af); aw = aw - aw[valid].mean(0); sn2 = hf_std(aw[valid]) ** 2
    for j in range(3): H[:, j, 4 * j:4 * j + 4] = H4; y[:, j] = aw[:, j]; R[:, j] = sn2; mask[:, j] = valid
    if k_drag > 0:
        zb, e1, e2 = thrust_frame(af[valid]); fl = flight_mask(t, af)
        if lev:
            fs = 1.0 / np.median(np.diff(t)); wd = np.gradient(wl, axis=0) * fs; L = [np.cross(wd, u) + np.cross(wl, np.cross(wl, u)) for u in np.eye(3)]
        for i, e in enumerate((e1, e2)):
            ew = qrot(q, np.broadcast_to(e, (N, 3)))
            for j in range(3): H[:, 3 + i, 4 * j + 1] = -k_drag * ew[:, j]
            H[:, 3 + i, 12 + i] = 1; y[:, 3 + i] = af @ e; R[:, 3 + i] = sd ** 2; mask[:, 3 + i] = valid & fl
            if lev:
                for j in range(3): H[:, 3 + i, 14 + j] = L[j] @ e
    if zmask is not None:
        for j in range(3): H[:, 5 + j, 4 * j + 1] = 1; R[:, 5 + j] = sz ** 2; mask[:, 5 + j] = zmask
    if vel is not None:
        ii, vv, sg = vel; sg2 = np.broadcast_to(np.asarray(sg, float) ** 2, (len(ii),))
        for j in range(3): H[ii, 8 + j, 4 * j + 1] = 1; y[ii, 8 + j] = vv[:, j]; R[ii, 8 + j] = sg2; mask[ii, 8 + j] = True
    binf = np.zeros(N); binf[1:][valid[1:] & ~valid[:-1]] = bias_jump ** 2
    xs, Ps, xf, Pf = kf_rts_seq2(A, Q, idx, H, y, R, mask, P0, binf); return xs[:, [0, 4, 8]], xs, Ps
def estimate_v5(tn, w, a, n_blind=0, k_drag=None, vel_fn=None, sv=0.5, gate=0.04, force=None, P=None):
    """v4 with optional velocity rows. vel_fn(P) -> (row indices on the uniform timeline, v (M,3)) or None. With vel_fn=None this
    is v4 computed through the joint smoother (hand-held: Matern model with the drag rows switched off)."""
    P = prep_v4(tn, w, a) if P is None else P; t, q, af, valid, zmask, wl = P["t"], P["q"], P["af"], P["valid"], P["zmask"], P["wl"]; N0 = P["N0"]; vel = None
    if vel_fn is not None:
        r = vel_fn(P)
        if r is not None and len(r[0]): vel = (r[0], r[1], sv if len(r) < 3 else r[2])
    if n_blind:
        ext = lambda x: np.concatenate([x, np.repeat(x[-1:], n_blind, 0)]); t = np.concatenate([t, t[-1] + P["dtn"] * np.arange(1, n_blind + 1)]); valid = np.concatenate([valid, np.zeros(n_blind, bool)])
        zmask = np.concatenate([zmask, np.zeros(n_blind, bool)]); q, af, wl = ext(q), ext(af), ext(wl)
    rotor = (P["lock"] < gate) if force is None else (force == "drag")
    if rotor: p, xs, Ps = run_joint7(t, q, af, model_spring(*ROTOR), valid, k_drag=K_DRAG if k_drag is None else k_drag, zmask=zmask, wl=wl, vel=vel)
    else: p, xs, Ps = run_joint7(t, q, af, model_matern52(*MATERN), valid, k_drag=0.0, zmask=zmask, vel=vel)
    io = np.concatenate([P["rows"], np.arange(N0, N0 + n_blind)]).astype(np.int64)
    return {"t": t[io], "p": p[io], "q": q[io], "v": xs[:, [1, 5, 9]][io], "sig": np.sqrt(Ps[:, 0, 0] + Ps[:, 4, 4] + Ps[:, 8, 8])[io], "rotor": bool(rotor), "lock": P["lock"], "n_vel": 0 if vel is None else len(vel[0])}

# ---------------------------------------------------------------- Phase 3: decimated smoother (rate governor) + full-state access
def build_joint(t, q, af, model, valid, k_drag=0.0, sd=SD_DRAG, zmask=None, sz=0.02, wl=None, vel=None, lever_prior=0.15, bias_jump=0.5, tau_c=100.0, sc=0.15, sn_scale=1.0, gap_ref=None, sn_roll=None, abias=None, no_wbias=False):
    """The measurement system of run_joint7 at the full IMU rate: returns F, Qc, P0, H, y, R, mask, binf.
    Phase 3 options: sn_roll = window [s] of a time-varying accelerometer noise level; abias = (prior sigma, random-walk
    intensity) adds three BODY-frame accelerometer-bias states; no_wbias switches the per-world-axis bias states off."""
    F4, Qc4, P04, H4 = model; lev = wl is not None and k_drag > 0; n0 = 17 if lev else 14; n = n0 + (3 if abias is not None else 0); N = len(t); F = np.zeros((n, n)); Qc = np.zeros((n, n)); P0 = np.zeros((n, n))
    for j in range(3):
        sl = slice(4 * j, 4 * j + 4); F[sl, sl] = F4; Qc[sl, sl] = Qc4; P0[sl, sl] = P04
        if no_wbias: Qc[4 * j + 3, 4 * j + 3] = 0.0; P0[4 * j + 3, :] = 0.0; P0[:, 4 * j + 3] = 0.0; P0[4 * j + 3, 4 * j + 3] = 1e-10
    for j in (12, 13): F[j, j] = -1.0 / tau_c; Qc[j, j] = 2 * sc ** 2 / tau_c; P0[j, j] = sc ** 2
    if lev:
        for j in (14, 15, 16): P0[j, j] = lever_prior ** 2
    m = 11 if vel is not None else 8; H = np.zeros((N, m, n)); y = np.zeros((N, m)); R = np.ones((N, m)); mask = np.zeros((N, m), np.bool_)
    aw = qrot(q, af); aw = aw - aw[valid].mean(0); sn2 = (sn_scale * hf_std(aw[valid])) ** 2
    if sn_roll is not None:
        fs_ = 1.0 / np.median(np.diff(t)); d2 = np.zeros(N); d2[1:] = 0.5 * np.mean(np.diff(aw, axis=0) ** 2, 1); d2[1:][~(valid[1:] & valid[:-1])] = np.nan; ok_ = ~np.isnan(d2); ok_[0] = False
        d2 = np.interp(np.arange(N), np.where(ok_)[0], d2[ok_]) if ok_.sum() > 10 else np.full(N, sn2); sn2 = sn_scale ** 2 * np.maximum(lowpass(d2, fs_, 1.0 / (2 * np.pi * sn_roll)), 0.05 * np.median(d2))
    for j in range(3): H[:, j, 4 * j:4 * j + 4] = H4; y[:, j] = aw[:, j]; R[:, j] = sn2; mask[:, j] = valid
    if abias is not None:
        Rm_ = Rot.from_quat(q).as_matrix()
        for k in range(3): P0[n0 + k, n0 + k] = abias[0] ** 2; Qc[n0 + k, n0 + k] = abias[1]; H[:, 0, n0 + k] = Rm_[:, 0, k]; H[:, 1, n0 + k] = Rm_[:, 1, k]; H[:, 2, n0 + k] = Rm_[:, 2, k]
    if k_drag > 0:
        zb, e1, e2 = thrust_frame(af[valid]); fl = flight_mask(t, af)
        if lev:
            fs = 1.0 / np.median(np.diff(t)); wd = np.gradient(wl, axis=0) * fs; L = [np.cross(wd, u) + np.cross(wl, np.cross(wl, u)) for u in np.eye(3)]
        for i, e in enumerate((e1, e2)):
            ew = qrot(q, np.broadcast_to(e, (N, 3)))
            for j in range(3): H[:, 3 + i, 4 * j + 1] = -k_drag * ew[:, j]
            H[:, 3 + i, 12 + i] = 1; y[:, 3 + i] = af @ e; R[:, 3 + i] = sd ** 2; mask[:, 3 + i] = valid & fl
            if lev:
                for j in range(3): H[:, 3 + i, 14 + j] = L[j] @ e
    if zmask is not None:
        for j in range(3): H[:, 5 + j, 4 * j + 1] = 1; R[:, 5 + j] = sz ** 2; mask[:, 5 + j] = zmask
    if vel is not None:
        ii, vv, sg = vel; sg2 = np.broadcast_to(np.asarray(sg, float) ** 2, (len(ii),))
        for j in range(3): H[ii, 8 + j, 4 * j + 1] = 1; y[ii, 8 + j] = vv[:, j]; R[ii, 8 + j] = sg2; mask[ii, 8 + j] = True
    binf = np.zeros(N); binf[1:][valid[1:] & ~valid[:-1]] = bias_jump ** 2
    if gap_ref is not None:
        st = np.where(valid[1:] & ~valid[:-1])[0] + 1; en = np.where(~valid[1:] & valid[:-1])[0] + 1
        for i in st:
            pr = en[en < i]; binf[i] *= min(1.0, ((t[i] - t[pr[-1] - 1]) if len(pr) else gap_ref) / gap_ref)
    return F, Qc, P0, H, y, R, mask, binf
def block_average(t, H, y, R, mask, binf, d):
    """Average each measurement row over blocks of d samples (information-preserving for white noise: R/count)."""
    N, m, n = H.shape; K = (N + d - 1) // d; pad = K * d - N
    if pad:
        dt = t[-1] - t[-2]; t = np.concatenate([t, t[-1] + dt * np.arange(1, pad + 1)]); H = np.concatenate([H, np.zeros((pad, m, n))]); y = np.concatenate([y, np.zeros((pad, m))]); R = np.concatenate([R, np.ones((pad, m))])
        mask = np.concatenate([mask, np.zeros((pad, m), np.bool_)]); binf = np.concatenate([binf, np.zeros(pad)])
    mk = mask.reshape(K, d, m); cnt = mk.sum(1); c1 = np.maximum(cnt, 1)
    yd = (y * mask).reshape(K, d, m).sum(1) / c1; Hd = (H * mask[:, :, None]).reshape(K, d, m, n).sum(1) / c1[:, :, None]; Rd = (R * mask).reshape(K, d, m).sum(1) / c1 / c1; Rd[cnt == 0] = 1.0
    return t.reshape(K, d).mean(1), Hd, yd, Rd, cnt > 0, binf.reshape(K, d).max(1)
def hermite(tk, pk, vk, t):
    """Cubic Hermite interpolation of (pk, vk) at nodes tk onto t (constant-velocity extrapolation outside)."""
    i = np.clip(np.searchsorted(tk, t) - 1, 0, len(tk) - 2); h = (tk[i + 1] - tk[i])[:, None]; s = ((t - tk[i]) / (tk[i + 1] - tk[i]))[:, None]; out = np.empty((len(t), pk.shape[1]))
    s2, s3 = s * s, s * s * s; out = (2 * s3 - 3 * s2 + 1) * pk[i] + (s3 - 2 * s2 + s) * h * vk[i] + (-2 * s3 + 3 * s2) * pk[i + 1] + (s3 - s2) * h * vk[i + 1]
    lo = t < tk[0]; hi = t > tk[-1]; out[lo] = pk[0] + (t[lo] - tk[0])[:, None] * vk[0]; out[hi] = pk[-1] + (t[hi] - tk[-1])[:, None] * vk[-1]; return out
def run_joint8(t, q, af, model, valid, dec=1, **kw):
    """run_joint7 with the smoother running on blocks of `dec` samples. Returns p (full rate), xs, Ps (at the smoother rate), tk."""
    F, Qc, P0, H, y, R, mask, binf = build_joint(t, q, af, model, valid, **kw)
    if dec > 1: tk, H, y, R, mask, binf = block_average(t, H, y, R, mask, binf, dec)
    else: tk = t
    A, Q, idx = discretise(F, Qc, tk); xs, Ps, xf, Pf = kf_rts_seq2(A, Q, idx, H, y, R, mask, P0, binf)
    p = xs[:, [0, 4, 8]] if dec == 1 else hermite(tk, xs[:, [0, 4, 8]], xs[:, [1, 5, 9]], t); return p, xs, Ps, tk
def estimate_v6(tn, w, a, n_blind=0, k_drag=None, dec=1, gate=0.04, force=None, P=None, model=None, vel=None, full=False, **kw):
    """v4 through the joint smoother with a rate governor (dec) and optional overrides (model, extra build_joint keywords)."""
    P = prep_v4(tn, w, a) if P is None else P; t, q, af, valid, zmask, wl = P["t"], P["q"], P["af"], P["valid"], P["zmask"], P["wl"]; N0 = P["N0"]
    if n_blind:
        ext = lambda x: np.concatenate([x, np.repeat(x[-1:], n_blind, 0)]); t = np.concatenate([t, t[-1] + P["dtn"] * np.arange(1, n_blind + 1)]); valid = np.concatenate([valid, np.zeros(n_blind, bool)])
        zmask = np.concatenate([zmask, np.zeros(n_blind, bool)]); q, af, wl = ext(q), ext(af), ext(wl)
    rotor = (P["lock"] < gate) if force is None else (force == "drag"); kw.setdefault("zmask", zmask)
    if rotor: kw.setdefault("wl", wl); p, xs, Ps, tk = run_joint8(t, q, af, model_spring(*ROTOR) if model is None else model, valid, dec=dec, k_drag=K_DRAG if k_drag is None else k_drag, vel=vel, **kw)
    else: kw.pop("wl", None); p, xs, Ps, tk = run_joint8(t, q, af, model_matern52(*MATERN) if model is None else model, valid, dec=dec, k_drag=0.0, vel=vel, **kw)
    io = np.concatenate([P["rows"], np.arange(N0, N0 + n_blind)]).astype(np.int64); out = {"t": t[io], "p": p[io], "q": q[io], "rotor": bool(rotor), "lock": P["lock"], "drift_D": P["D"], "drift_used": bool(P["drift"])}
    if full: out.update(xs=xs, Ps=Ps, tk=tk, N0=N0, t_full=t, P=P)
    return out
