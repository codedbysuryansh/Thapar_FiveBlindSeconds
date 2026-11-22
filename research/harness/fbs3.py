"""fbs3.py - Phase 2 front end (v3) and unified pipeline. RESEARCH INSTRUMENT, NOT THE SUBMISSION.
Builds on the frozen Phase 1B harness (fbs.py, left untouched for reproducibility). Ground truth never enters here."""
import numpy as np
from numba import njit
from fbs import *  # noqa: F401,F403
import fbs

SPRING = (20, 0.5, 0.3, 1.0, 20, 0.1)      # T_p, tau_v, tau_a, sa, tau_b, sb  (frozen from Phase 1B e07, index 1)
MATERN = (1, 2, 20, 0.1)                   # ell, sp, tau_b, sb                (frozen from Phase 1B e02 LOSO majority)

def ingest(tn, w, a, held_run=4):
    """Map every dropout encoding onto one uniform timeline with a validity mask.
    NaN rows, all-zero rows, runs of >= held_run identical rows (held values; the first is kept) and removed rows
    (timestamp steps > 1.5 nominal intervals; the missing stamps are re-inserted) all become valid=False.
    Returns t [s from first stamp], w, a (NaN where invalid), valid, rows (position of each input row on the timeline)."""
    tn = np.asarray(tn, np.int64); X = np.hstack([w, a]).astype(float); d = np.diff(tn); med = np.median(d)
    steps = np.maximum(1, np.round(d / med).astype(np.int64)); rows = np.concatenate([[0], np.cumsum(steps)]); N = int(rows[-1]) + 1
    t = np.interp(np.arange(N), rows, (tn - tn[0]) * 1e-9); Xu = np.full((N, 6), np.nan); Xu[rows] = X
    bad = np.isnan(Xu).any(1) | (Xu == 0).all(1); same = np.zeros(N, bool); same[1:] = (Xu[1:] == Xu[:-1]).all(1)
    run = np.zeros(N, np.int64)
    for i in range(1, N): run[i] = run[i - 1] + 1 if same[i] else 0
    held = np.zeros(N, bool); i = N - 1
    while i > 0:
        if run[i] >= held_run - 1:
            j = i - run[i]; held[j + 1:i + 1] = True; i = j
        else: i -= 1
    valid = ~(bad | held); Xu[~valid] = np.nan
    return t, Xu[:, :3], Xu[:, 3:], valid, rows

def rest_anchors(t, w, a, valid, win=0.5, frac=0.05, gate=0.035, min_windows=3):
    """Genuine rest intervals -> (time, gyro bias) anchors. A cluster of quiet windows is accepted only if it has
    >= min_windows windows whose mean gyro readings agree to within the noise of a window mean (+1 mrad/s).
    Slow hand-held motion (TUM-VI room6) fails that agreement test; a drone on the ground passes it."""
    fs = 1.0 / np.median(np.diff(t)); n = max(8, int(win * fs)); m = len(t) // n; idx = np.arange(m * n).reshape(m, n); okw = valid[idx].all(1)
    if okw.sum() < 10: return []
    Wn = np.nan_to_num(w[:m * n]).reshape(m, n, 3); An = np.nan_to_num(a[:m * n]).reshape(m, n, 3)
    gs = Wn.std(1).max(1); as_ = An.std(1).max(1); gs[~okw] = np.inf; as_[~okw] = np.inf
    sc = gs / np.median(gs[okw]) + as_ / np.median(as_[okw]); k = max(3, int(frac * okw.sum())); cand = np.argsort(sc)[:k]; cand = np.sort(cand[gs[cand] < gate])
    out = []
    if len(cand) == 0: return out
    for g in np.split(cand, np.where(np.diff(cand) > 4)[0] + 1):
        if len(g) < min_windows: continue
        means = Wn[g].mean(1); spread = means.std(0).max(); allow = 1.0e-3 + 3.0 * gs[g].mean() / np.sqrt(n)
        if spread < allow: out.append((float(t[idx[g]].mean()), np.median(means, 0)))
    return out
def bias_track(t, anchors):
    if not anchors: return None
    ta = np.array([x[0] for x in anchors]); B = np.array([x[1] for x in anchors])
    return np.stack([np.interp(t, ta, B[:, i]) for i in range(3)], 1)

def attitude3(t, w, a, valid, tau=3.0, bridge="lin", bg_track=None, drift=False, ridge=1e-1, ridge_d=1e-1, true_gap_rot=None):
    """Gap-aware offline attitude. bg_track (N,3): rest-anchored bias (no further fitting). Otherwise the horizontal
    bias components are fitted from tilt drift (constant, or constant + linear-in-time when drift=True).
    true_gap_rot is an ORACLE DIAGNOSTIC ONLY (the clean gyro inside gaps) used to size the heading loss."""
    fs = 1.0 / np.median(np.diff(t)); ez = np.array([0, 0, 1.0]); N = len(t); af = fill_gaps(t, a, valid)
    seg = np.cumsum(np.concatenate([[0], np.diff(valid.astype(np.int8)) != 0])); s = (t - t.mean()) / max(t[-1] - t[0], 1e-9)
    bg = np.zeros((N, 3)) if bg_track is None else bg_track.copy(); iters = 2 if bg_track is None else 0
    for it in range(iters + 1):
        wc = fill_gaps(t, w, valid)
        if true_gap_rot is not None: wc = np.where(valid[:, None], wc, true_gap_rot)
        wc = wc - bg
        if bridge == "zero": wc = np.where(valid[:, None], wc, 0.0)
        qg = integrate_gyro(t, wc); gI = lowpass_seg(qrot(qg, af), fs, 1.0 / (2 * np.pi * tau), valid); gI /= np.linalg.norm(gI, axis=1, keepdims=True)
        qc = shortest_arc(gI, np.broadcast_to(ez, gI.shape)); q = qmul(qc, qg); q /= np.linalg.norm(q, axis=1, keepdims=True)
        if it == iters: break
        step = max(1, int(fs * 1.0)); wcr = qlog(qmul(qc[step:], qinv(qc[:-step]))) / (t[step:] - t[:-step])[:, None]
        good = valid[:-step] & valid[step:] & (seg[:-step] == seg[step:]); Rm = Rot.from_quat(q[:-step][good][::10]).as_matrix()
        A = Rm[:, :2, :].reshape(-1, 3); y = -wcr[good][::10, :2].reshape(-1); reg = [ridge] * 3
        if drift: A = np.hstack([A, A * np.repeat(s[:-step][good][::10], 2)[:, None]]); reg = reg + [ridge_d] * 3
        sol = np.linalg.solve(A.T @ A + len(y) * np.diag(reg), A.T @ y); bg = bg + sol[:3] + (sol[3:] * s[:, None] if drift else 0.0)
    return q, bg
def front3(t, w, a, valid, tau=3.0, drift=False, two_sided=True, bridge="lin", true_gap_rot=None):
    anc = rest_anchors(t, w, a, valid)
    if anc and not two_sided: anc = [(anc[0][0], np.mean([x[1] for x in anc], 0))]
    q, bg = attitude3(t, w, a, valid, tau=tau, bridge=bridge, bg_track=bias_track(t, anc), drift=drift, true_gap_rot=true_gap_rot)
    return q, bg, bool(anc), fill_gaps(t, a, valid)
def thrust_lock(t, af, valid):
    """Geometry test for a multirotor, no labels: RMS of the low-passed (1 Hz) specific force perpendicular to its
    mean direction, in g. Small when thrust stays on one body axis; large when a hand tilts the device."""
    zb = af[valid].mean(0); zb = zb / np.linalg.norm(zb); fs = 1.0 / np.median(np.diff(t))
    al = lowpass_seg(af, fs, 1.0, valid)[valid]; perp = al - np.outer(al @ zb, zb); return float(np.sqrt((perp ** 2).sum(1).mean()) / G)

@njit(cache=True)
def kf_rts_seq2(A, Q, idx, H, y, R, mask, P0, binf):
    """kf_rts_seq with variance injection binf[k] on the three bias states (3, 7, 11) at prediction."""
    N = H.shape[0]; m = H.shape[1]; n = H.shape[2]
    xf = np.zeros((N, n)); Pf = np.zeros((N, n, n)); xp = np.zeros((N, n)); Pp = np.zeros((N, n, n)); x = np.zeros(n); P = P0.copy()
    for k in range(N):
        if k > 0:
            Ak = A[idx[k]]; x = Ak @ x; P = Ak @ P @ Ak.T + Q[idx[k]]
            if binf[k] > 0:
                P[3, 3] += binf[k]; P[7, 7] += binf[k]; P[11, 11] += binf[k]
        xp[k] = x; Pp[k] = P
        for i in range(m):
            if mask[k, i]:
                h = H[k, i]; PH = P @ h; S = h @ PH + R[k, i]; K = PH / S; x = x + K * (y[k, i] - h @ x); P = P - np.outer(K, PH)
        P = 0.5 * (P + P.T); xf[k] = x; Pf[k] = P
    xs = xf.copy(); Ps = Pf.copy()
    for k in range(N - 2, -1, -1):
        Ak = A[idx[k + 1]]; C = Pf[k] @ Ak.T @ np.linalg.inv(Pp[k + 1]); xs[k] = xf[k] + C @ (xs[k + 1] - xp[k + 1]); Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return xs, Ps, xf, Pf
def run_joint3(t, q, af, model, valid, k_drag=0.0, sd=2.0, pos_fix=None, bias_jump=0.5, tau_c=100.0, sc=0.15):
    """14-state joint smoother for any per-axis (F, Qc, P0, H) model: world-acceleration measurements, optional
    rotor-drag relation (k_drag > 0, in flight), optional position fixes pos_fix=(indices, positions (M,3), sigma)."""
    F4, Qc4, P04, H4 = model; n = 14; N = len(t); F = np.zeros((n, n)); Qc = np.zeros((n, n)); P0 = np.zeros((n, n))
    for j in range(3):
        sl = slice(4 * j, 4 * j + 4); F[sl, sl] = F4; Qc[sl, sl] = Qc4; P0[sl, sl] = P04
    for j in (12, 13): F[j, j] = -1.0 / tau_c; Qc[j, j] = 2 * sc ** 2 / tau_c; P0[j, j] = sc ** 2
    A, Q, idx = discretise(F, Qc, t); m = 5 + (3 if pos_fix is not None else 0)
    H = np.zeros((N, m, n)); y = np.zeros((N, m)); R = np.ones((N, m)); mask = np.zeros((N, m), np.bool_)
    aw = qrot(q, af); aw = aw - aw[valid].mean(0); sn2 = hf_std(aw[valid]) ** 2
    for j in range(3): H[:, j, 4 * j:4 * j + 4] = H4; y[:, j] = aw[:, j]; R[:, j] = sn2; mask[:, j] = valid
    if k_drag > 0:
        zb, e1, e2 = thrust_frame(af[valid]); fl = flight_mask(t, af)
        for i, e in enumerate((e1, e2)):
            ew = qrot(q, np.broadcast_to(e, (N, 3)))
            for j in range(3): H[:, 3 + i, 4 * j + 1] = -k_drag * ew[:, j]
            H[:, 3 + i, 12 + i] = 1; y[:, 3 + i] = af @ e; R[:, 3 + i] = sd ** 2; mask[:, 3 + i] = valid & fl
    if pos_fix is not None:
        ii, pp, sg = pos_fix
        for j in range(3): H[ii, 5 + j, 4 * j] = 1; y[ii, 5 + j] = pp[:, j]; R[ii, 5 + j] = sg ** 2; mask[ii, 5 + j] = True
    binf = np.zeros(N); binf[1:][valid[1:] & ~valid[:-1]] = bias_jump ** 2
    xs, Ps, xf, Pf = kf_rts_seq2(A, Q, idx, H, y, R, mask, P0, binf)
    return xs[:, [0, 4, 8]], xs, Ps

def estimate3(tn, w, a, model=None, n_blind=0, drag=None, pos_fix=None, tau=3.0, drift=False, two_sided=True, gate=0.04, bias_jump=0.5):
    """Unified v3 pipeline on a raw (possibly degraded) stream: ingest -> rest-anchored gap-aware attitude ->
    stationary-prior smoother (-> optional gated drag / position fixes) -> one pose per input row, plus n_blind
    appended rows predicted by the model itself with orientation held. drag = (k, sd) or None; it is applied only
    if the thrust-lock geometry test is below `gate`."""
    t, wu, au, valid, rows = ingest(tn, w, a); N0 = len(t); q, bg, quiet, af = front3(t, wu, au, valid, tau=tau, drift=drift, two_sided=two_sided)
    lock = thrust_lock_local(t, af, valid); dtn = np.median(np.diff(t))
    if n_blind:
        t = np.concatenate([t, t[-1] + dtn * np.arange(1, n_blind + 1)]); valid = np.concatenate([valid, np.zeros(n_blind, bool)])
        q = np.vstack([q, np.repeat(q[-1:], n_blind, 0)]); af = np.vstack([af, np.repeat(af[-1:], n_blind, 0)])
    model = model_spring(*SPRING) if model is None else model; use_drag = drag is not None and lock < gate
    if use_drag or pos_fix is not None:
        pf = None if pos_fix is None else (rows[pos_fix[0]], pos_fix[1], pos_fix[2])
        p, xs, Ps = run_joint3(t, q, af, model, valid, k_drag=drag[0] if use_drag else 0.0, sd=drag[1] if use_drag else 2.0, pos_fix=pf, bias_jump=bias_jump)
        v = xs[:, [1, 5, 9]]; sig = np.sqrt(Ps[:, 0, 0] + Ps[:, 4, 4] + Ps[:, 8, 8])
    else:
        p, xs, Ps = lin_smooth(t, q, af, valid, model, bias_jump=bias_jump)[:3]; v = xs[:, :, 1]; sig = np.sqrt(3 * Ps[:, 0, 0])
    io = np.concatenate([rows, np.arange(N0, N0 + n_blind)]).astype(np.int64)
    return {"t": t[io], "p": p[io], "q": q[io], "v": v[io], "sig": sig[io], "valid": valid[io], "quiet": quiet, "bg": bg, "lock": lock, "drag_used": bool(use_drag), "n_rows": len(rows),
            "internal": (t, p, q, valid, af)}

def thrust_lock_local(t, af, valid, win=10.0):
    """Median over `win`-second windows of the thrust-lock measure computed against each window's own mean
    specific-force direction. Robust to a hand-carried start (EuRoC MH_01/02) and needs no labels."""
    fs = 1.0 / np.median(np.diff(t)); n = int(win * fs); al = lowpass_seg(af, fs, 1.0, valid); out = []
    for i in range(0, len(t) - n + 1, n // 2):
        v = valid[i:i + n]
        if v.sum() < n // 2: continue
        x = al[i:i + n][v]; zb = x.mean(0); zb = zb / np.linalg.norm(zb); perp = x - np.outer(x @ zb, zb); out.append(np.sqrt((perp ** 2).sum(1).mean()) / G)
    return float(np.median(out)) if out else 1.0

def gap_shrink_factors(t, valid, base=0.6):
    """Heading-uncertainty shrinkage. Heading across a long gap is unobservable, so horizontal motion estimated on the far
    side of a gap is, in expectation over the unknown yaw, the estimate scaled by E[cos(yaw error)] < 1. Each valid segment
    gets base**W, W = sum over the gaps separating it from the longest segment of min(1, (duration / 2 s)^2).
    Returns one factor per row (gap and trailing blind rows interpolate / hold). Apply to the two horizontal axes only."""
    N = len(t); dt = np.median(np.diff(t)); seg = np.cumsum(np.concatenate([[0], np.diff(valid.astype(np.int8)) != 0]))
    ids = np.unique(seg[valid]); lens = np.array([(seg == i).sum() for i in ids]); r = int(np.argmax(lens)); W = np.zeros(len(ids))
    gw = [min(1.0, (((seg > ids[k]) & (seg < ids[k + 1])).sum() * dt / 2.0) ** 2) for k in range(len(ids) - 1)]
    for k in range(r + 1, len(ids)): W[k] = W[k - 1] + gw[k - 1]
    for k in range(r - 1, -1, -1): W[k] = W[k + 1] + gw[k]
    fac = np.full(N, np.nan)
    for k, i in enumerate(ids): fac[seg == i] = base ** W[k]
    ok = ~np.isnan(fac); idx = np.arange(N); return np.interp(idx, idx[ok], fac[ok])
def prep3(tn, w, a, n_blind=0, drift=False, tau=3.0):
    """Front end once (ingest, attitude, blind-row extension) so that many position models can be compared on it."""
    t, wu, au, valid, rows = ingest(tn, w, a); N0 = len(t); q, bg, quiet, af = front3(t, wu, au, valid, tau=tau, drift=drift); lock = thrust_lock_local(t, af, valid); dtn = np.median(np.diff(t))
    if n_blind:
        t = np.concatenate([t, t[-1] + dtn * np.arange(1, n_blind + 1)]); valid = np.concatenate([valid, np.zeros(n_blind, bool)])
        q = np.vstack([q, np.repeat(q[-1:], n_blind, 0)]); af = np.vstack([af, np.repeat(af[-1:], n_blind, 0)])
    return t, q, af, valid, lock, N0

# ---- Phase 2 chosen architecture (e13): one front end, geometry-gated choice of the position model ----
K_DRAG = 0.1873                                  # mean of the 11 per-flight regression values in e05_drag.json (sd 0.014)
ROTOR = (20, 2.0, 0.3, 1.0, 20, 0.1)             # spring-damper used WITH the drag relation (e13, picked 10/11 LOSO)
def estimate_final(tn, w, a, n_blind=0, k_drag=None, sd=2.0, gate=0.04, force=None):
    """ingest -> rest-anchored gap-aware attitude -> thrust-lock gate -> Matern(1,2) per axis (hand-held) or
    spring(20,2,0.3,1) + rotor drag (multirotor). force in (None, 'matern', 'spring', 'drag') overrides the gate.
    Blind rows (n_blind) are the model's own prediction with the last orientation held. deploy/fbs_core.c mirrors this."""
    t, wu, au, valid, rows = ingest(tn, w, a); N0 = len(t); q, bg, quiet, af = front3(t, wu, au, valid); lock = thrust_lock_local(t, af, valid); dtn = np.median(np.diff(t))
    if n_blind:
        t = np.concatenate([t, t[-1] + dtn * np.arange(1, n_blind + 1)]); valid = np.concatenate([valid, np.zeros(n_blind, bool)])
        q = np.vstack([q, np.repeat(q[-1:], n_blind, 0)]); af = np.vstack([af, np.repeat(af[-1:], n_blind, 0)])
    rotor = (lock < gate) if force is None else (force == "drag")
    if rotor: p, xs, Ps = run_joint3(t, q, af, model_spring(*ROTOR), valid, k_drag=K_DRAG if k_drag is None else k_drag, sd=sd); v = xs[:, [1, 5, 9]]; sig = np.sqrt(Ps[:, 0, 0] + Ps[:, 4, 4] + Ps[:, 8, 8])
    else:
        p, xs, Ps = lin_smooth(t, q, af, valid, model_spring(*SPRING) if force == "spring" else model_matern52(*MATERN), bias_jump=0.5)[:3]; v = xs[:, :, 1]; sig = np.sqrt(3 * Ps[:, 0, 0])
    io = np.concatenate([rows, np.arange(N0, N0 + n_blind)]).astype(np.int64)
    return {"t": t[io], "p": p[io], "q": q[io], "v": v[io], "sig": sig[io], "valid": valid[io], "anchors": quiet, "bg": bg, "lock": lock, "rotor": bool(rotor), "n_rows": len(rows), "internal": (t, p, q, valid, af)}

# ---- zero-velocity pseudo-measurements at genuine rest (e14) ----
def rest_info(t, w, a, valid, win=0.5, frac=0.05, gate=0.035, min_windows=3):
    """Same acceptance test as rest_anchors; additionally returns a per-row mask of the accepted quiet windows."""
    mask = np.zeros(len(t), bool); fs = 1.0 / np.median(np.diff(t)); n = max(8, int(win * fs)); m = len(t) // n; idx = np.arange(m * n).reshape(m, n); okw = valid[idx].all(1)
    if okw.sum() < 10: return [], mask
    Wn = np.nan_to_num(w[:m * n]).reshape(m, n, 3); An = np.nan_to_num(a[:m * n]).reshape(m, n, 3)
    gs = Wn.std(1).max(1); as_ = An.std(1).max(1); gs[~okw] = np.inf; as_[~okw] = np.inf
    sc = gs / np.median(gs[okw]) + as_ / np.median(as_[okw]); k = max(3, int(frac * okw.sum())); cand = np.argsort(sc)[:k]; cand = np.sort(cand[gs[cand] < gate]); out = []
    if len(cand) == 0: return out, mask
    for g in np.split(cand, np.where(np.diff(cand) > 4)[0] + 1):
        if len(g) < min_windows: continue
        means = Wn[g].mean(1); spread = means.std(0).max(); allow = 1.0e-3 + 3.0 * gs[g].mean() / np.sqrt(n)
        if spread < allow: out.append((float(t[idx[g]].mean()), np.median(means, 0))); mask[idx[g].ravel()] = True
    return out, mask
@njit(cache=True)
def kf_rts3(A, Q, idx, H, y, Rk, mask, P0, binf, zmask, Rz):
    """kf_rts2 plus a zero-velocity pseudo-measurement (state index 1) on rows flagged in zmask."""
    N = y.shape[0]; n = A.shape[1]; xf = np.zeros((N, 3, n)); Pf = np.zeros((N, n, n)); xp = np.zeros((N, 3, n)); Pp = np.zeros((N, n, n)); x = np.zeros((3, n)); P = P0.copy()
    for k in range(N):
        if k > 0:
            Ak = A[idx[k]]; x = x @ Ak.T; P = Ak @ P @ Ak.T + Q[idx[k]]; P[n - 1, n - 1] += binf[k]
        xp[k] = x; Pp[k] = P
        if mask[k]:
            PH = P @ H; S = H @ PH + Rk[k]; K = PH / S
            for j in range(3): x[j] = x[j] + K * (y[k, j] - H @ x[j])
            P = P - np.outer(K, PH); P = 0.5 * (P + P.T)
        if zmask[k]:
            PH = P[:, 1].copy(); K = PH / (PH[1] + Rz)
            for j in range(3): x[j] = x[j] - K * x[j, 1]
            P = P - np.outer(K, PH); P = 0.5 * (P + P.T)
        xf[k] = x; Pf[k] = P
    xs = xf.copy(); Ps = Pf.copy()
    for k in range(N - 2, -1, -1):
        Ak = A[idx[k + 1]]; C = Pf[k] @ Ak.T @ np.linalg.inv(Pp[k + 1])
        for j in range(3): xs[k, j] = xf[k, j] + C @ (xs[k + 1, j] - xp[k + 1, j])
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return xs, Ps
def lin_smooth_z(t, q, af, valid, model, zmask, sz=0.02, bias_jump=0.5):
    aw = qrot(q, af); aw = aw - aw[valid].mean(0); F, Qc, P0, H = model; A, Q, idx = discretise(F, Qc, t)
    Rk = np.full(len(t), hf_std(aw[valid]) ** 2); binf = np.zeros(len(t)); binf[1:][valid[1:] & ~valid[:-1]] = bias_jump ** 2
    xs, Ps = kf_rts3(A, Q, idx, H, np.ascontiguousarray(aw), Rk, valid.astype(np.bool_), P0, binf, zmask.astype(np.bool_), sz ** 2); return xs[:, :, 0], xs, Ps
def run_joint4(t, q, af, model, valid, k_drag=0.0, sd=2.0, zmask=None, sz=0.02, bias_jump=0.5, tau_c=100.0, sc=0.15):
    """run_joint3 plus zero-velocity rows (zmask) - 14 states, up to 8 scalar measurements per step."""
    F4, Qc4, P04, H4 = model; n = 14; N = len(t); F = np.zeros((n, n)); Qc = np.zeros((n, n)); P0 = np.zeros((n, n))
    for j in range(3):
        sl = slice(4 * j, 4 * j + 4); F[sl, sl] = F4; Qc[sl, sl] = Qc4; P0[sl, sl] = P04
    for j in (12, 13): F[j, j] = -1.0 / tau_c; Qc[j, j] = 2 * sc ** 2 / tau_c; P0[j, j] = sc ** 2
    A, Q, idx = discretise(F, Qc, t); m = 8; H = np.zeros((N, m, n)); y = np.zeros((N, m)); R = np.ones((N, m)); mask = np.zeros((N, m), np.bool_)
    aw = qrot(q, af); aw = aw - aw[valid].mean(0); sn2 = hf_std(aw[valid]) ** 2
    for j in range(3): H[:, j, 4 * j:4 * j + 4] = H4; y[:, j] = aw[:, j]; R[:, j] = sn2; mask[:, j] = valid
    if k_drag > 0:
        zb, e1, e2 = thrust_frame(af[valid]); fl = flight_mask(t, af)
        for i, e in enumerate((e1, e2)):
            ew = qrot(q, np.broadcast_to(e, (N, 3)))
            for j in range(3): H[:, 3 + i, 4 * j + 1] = -k_drag * ew[:, j]
            H[:, 3 + i, 12 + i] = 1; y[:, 3 + i] = af @ e; R[:, 3 + i] = sd ** 2; mask[:, 3 + i] = valid & fl
    if zmask is not None:
        for j in range(3): H[:, 5 + j, 4 * j + 1] = 1; R[:, 5 + j] = sz ** 2; mask[:, 5 + j] = zmask
    binf = np.zeros(N); binf[1:][valid[1:] & ~valid[:-1]] = bias_jump ** 2
    xs, Ps, xf, Pf = kf_rts_seq2(A, Q, idx, H, y, R, mask, P0, binf); return xs[:, [0, 4, 8]], xs, Ps
def estimate_final2(tn, w, a, n_blind=0, k_drag=None, sd=2.0, gate=0.04, force=None, zupt=True, sz=0.02, sc=0.15):
    """estimate_final with optional zero-velocity rows at accepted rest windows and a tunable drag-bias prior (sc)."""
    t, wu, au, valid, rows = ingest(tn, w, a); N0 = len(t); q, bg, quiet, af = front3(t, wu, au, valid); lock = thrust_lock_local(t, af, valid); dtn = np.median(np.diff(t))
    zmask = rest_info(t, wu, au, valid)[1] if zupt else np.zeros(N0, bool)
    if n_blind:
        t = np.concatenate([t, t[-1] + dtn * np.arange(1, n_blind + 1)]); valid = np.concatenate([valid, np.zeros(n_blind, bool)]); zmask = np.concatenate([zmask, np.zeros(n_blind, bool)])
        q = np.vstack([q, np.repeat(q[-1:], n_blind, 0)]); af = np.vstack([af, np.repeat(af[-1:], n_blind, 0)])
    rotor = (lock < gate) if force is None else (force == "drag")
    if rotor: p, xs, Ps = run_joint4(t, q, af, model_spring(*ROTOR), valid, k_drag=K_DRAG if k_drag is None else k_drag, sd=sd, zmask=zmask, sz=sz, sc=sc); v = xs[:, [1, 5, 9]]; sig = np.sqrt(Ps[:, 0, 0] + Ps[:, 4, 4] + Ps[:, 8, 8])
    else: p, xs, Ps = lin_smooth_z(t, q, af, valid, model_spring(*SPRING) if force == "spring" else model_matern52(*MATERN), zmask, sz=sz); v = xs[:, :, 1]; sig = np.sqrt(3 * Ps[:, 0, 0])
    io = np.concatenate([rows, np.arange(N0, N0 + n_blind)]).astype(np.int64)
    return {"t": t[io], "p": p[io], "q": q[io], "v": v[io], "sig": sig[io], "valid": valid[io], "zupt_rows": int(zmask.sum()), "bg": bg, "lock": lock, "rotor": bool(rotor), "n_rows": len(rows)}

# ---- e14/e15: lever-arm corrected drag rows and a red + white drag-noise model ----
LEVER = (0.020, 0.016, 0.099)                    # mean of the 11 per-flight fits in e15 (sd 0.003, 0.009, 0.004 m), EuRoC sensor frame
def lever_acc(t, w, bg, r, fc=15.0):
    """Acceleration added at an accelerometer mounted r [m] from the centre of rotation: wdot x r + w x (w x r)."""
    fs = 1.0 / np.median(np.diff(t)); wl = lowpass(w - bg, fs, fc); wd = np.gradient(wl, axis=0) * fs; r = np.asarray(r, float)
    return np.cross(wd, r) + np.cross(wl, np.cross(wl, r))
def run_joint5(t, q, af, model, valid, k_drag=0.0, sd=2.0, zmask=None, sz=0.02, lev=None, fast=None, bias_jump=0.5, tau_c=100.0, sc=0.15):
    """run_joint4 plus (a) lev: (N,3) lever-arm acceleration removed from the drag rows, (b) fast=(tau_f, sf): a second,
    fast OU component per in-plane axis (red drag residual measured in e15) -> 14 or 16 states."""
    F4, Qc4, P04, H4 = model; n = 14 + (2 if fast else 0); N = len(t); F = np.zeros((n, n)); Qc = np.zeros((n, n)); P0 = np.zeros((n, n))
    for j in range(3):
        sl = slice(4 * j, 4 * j + 4); F[sl, sl] = F4; Qc[sl, sl] = Qc4; P0[sl, sl] = P04
    for j in (12, 13): F[j, j] = -1.0 / tau_c; Qc[j, j] = 2 * sc ** 2 / tau_c; P0[j, j] = sc ** 2
    if fast:
        for j in (14, 15): F[j, j] = -1.0 / fast[0]; Qc[j, j] = 2 * fast[1] ** 2 / fast[0]; P0[j, j] = fast[1] ** 2
    A, Q, idx = discretise(F, Qc, t); m = 8; H = np.zeros((N, m, n)); y = np.zeros((N, m)); R = np.ones((N, m)); mask = np.zeros((N, m), np.bool_)
    aw = qrot(q, af); aw = aw - aw[valid].mean(0); sn2 = hf_std(aw[valid]) ** 2
    for j in range(3): H[:, j, 4 * j:4 * j + 4] = H4; y[:, j] = aw[:, j]; R[:, j] = sn2; mask[:, j] = valid
    if k_drag > 0:
        zb, e1, e2 = thrust_frame(af[valid]); fl = flight_mask(t, af); ad = af if lev is None else af - lev
        for i, e in enumerate((e1, e2)):
            ew = qrot(q, np.broadcast_to(e, (N, 3)))
            for j in range(3): H[:, 3 + i, 4 * j + 1] = -k_drag * ew[:, j]
            H[:, 3 + i, 12 + i] = 1; y[:, 3 + i] = ad @ e; R[:, 3 + i] = sd ** 2; mask[:, 3 + i] = valid & fl
            if fast: H[:, 3 + i, 14 + i] = 1
    if zmask is not None:
        for j in range(3): H[:, 5 + j, 4 * j + 1] = 1; R[:, 5 + j] = sz ** 2; mask[:, 5 + j] = zmask
    binf = np.zeros(N); binf[1:][valid[1:] & ~valid[:-1]] = bias_jump ** 2
    xs, Ps, xf, Pf = kf_rts_seq2(A, Q, idx, H, y, R, mask, P0, binf); return xs[:, [0, 4, 8]], xs, Ps

# ---- e16: time-varying horizontal gyro bias (front4) and the lever arm estimated online ----
def attitude4(t, w, a, valid, tau=3.0, bg_track=None, win=30.0, knot=10.0, ridge=1e-1, iters=2):
    """attitude3 with a TIME-VARYING gyro-bias correction, also when rest anchors exist. The tilt correction drifts at
    the rate of the horizontal gyro-bias error; at knots every `knot` s a local ridge LS (triangular window +-win s)
    turns that drift into a bias increment, increments are interpolated and added to the anchor track. The component
    about gravity stays unobservable in motion (the ridge leaves it at the anchor value)."""
    fs = 1.0 / np.median(np.diff(t)); ez = np.array([0, 0, 1.0]); N = len(t); af = fill_gaps(t, a, valid); wf = fill_gaps(t, w, valid)
    seg = np.cumsum(np.concatenate([[0], np.diff(valid.astype(np.int8)) != 0])); bg = np.zeros((N, 3)) if bg_track is None else bg_track.copy()
    for it in range(iters + 1):
        qg = integrate_gyro(t, wf - bg); gI = lowpass_seg(qrot(qg, af), fs, 1.0 / (2 * np.pi * tau), valid); gI /= np.linalg.norm(gI, axis=1, keepdims=True)
        qc = shortest_arc(gI, np.broadcast_to(ez, gI.shape)); q = qmul(qc, qg); q /= np.linalg.norm(q, axis=1, keepdims=True)
        if it == iters: break
        step = max(1, int(fs * 1.0)); wcr = qlog(qmul(qc[step:], qinv(qc[:-step]))) / (t[step:] - t[:-step])[:, None]
        ii = np.where(valid[:-step] & valid[step:] & (seg[:-step] == seg[step:]))[0][::10]
        if len(ii) < 50: break
        A = Rot.from_quat(q[ii]).as_matrix()[:, :2, :]; y = -wcr[ii, :2]; tm = t[ii] + 0.5 * step / fs; knots = np.arange(t[0], t[-1] + knot, knot); db = np.zeros((len(knots), 3))
        for j, tk in enumerate(knots):
            wg = np.clip(1 - np.abs(tm - tk) / win, 0, None); sel = wg > 0
            if sel.sum() < 20: continue
            Aw = A[sel] * wg[sel][:, None, None]; db[j] = np.linalg.solve(np.einsum("nij,nik->jk", Aw, A[sel]) + 2 * wg[sel].sum() * ridge * np.eye(3), np.einsum("nij,ni->j", Aw, y[sel]))
        bg = bg + np.stack([np.interp(t, knots, db[:, i]) for i in range(3)], 1)
    return q, bg
def front4(t, w, a, valid, tau=3.0, win=30.0, knot=10.0):
    anc = rest_anchors(t, w, a, valid); q, bg = attitude4(t, w, a, valid, tau=tau, bg_track=bias_track(t, anc), win=win, knot=knot); return q, bg, bool(anc), fill_gaps(t, a, valid)
def run_joint6(t, q, af, model, valid, k_drag=0.0, sd=2.0, zmask=None, sz=0.02, wl=None, lever_prior=0.15, bias_jump=0.5, tau_c=100.0, sc=0.15):
    """Joint smoother with zero-velocity rows and, when wl (N,3 low-passed, bias-corrected gyro) is given, three constant
    lever-arm states r: the drag rows gain e_i . (wdot x r + w x (w x r)). 14 or 17 states. Works in any sensor frame."""
    F4, Qc4, P04, H4 = model; lev = wl is not None and k_drag > 0; n = 17 if lev else 14; N = len(t); F = np.zeros((n, n)); Qc = np.zeros((n, n)); P0 = np.zeros((n, n))
    for j in range(3):
        sl = slice(4 * j, 4 * j + 4); F[sl, sl] = F4; Qc[sl, sl] = Qc4; P0[sl, sl] = P04
    for j in (12, 13): F[j, j] = -1.0 / tau_c; Qc[j, j] = 2 * sc ** 2 / tau_c; P0[j, j] = sc ** 2
    if lev:
        for j in (14, 15, 16): P0[j, j] = lever_prior ** 2
    A, Q, idx = discretise(F, Qc, t); m = 8; H = np.zeros((N, m, n)); y = np.zeros((N, m)); R = np.ones((N, m)); mask = np.zeros((N, m), np.bool_)
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
    binf = np.zeros(N); binf[1:][valid[1:] & ~valid[:-1]] = bias_jump ** 2
    xs, Ps, xf, Pf = kf_rts_seq2(A, Q, idx, H, y, R, mask, P0, binf); return xs[:, [0, 4, 8]], xs, Ps

# ---- Phase 2 FINAL architecture (v4): e14-e16 decisions ----
DRIFT_THR = 2.0e-3      # rad/s; clean streams stay <= 1.46e-3, strong drift >= 2.78e-3 on TUM-VI (e16c)
SD_DRAG = 1.4           # m/s^2, drag-row noise with the lever states (e16)
def front5(t, w, a, valid, tau=3.0, win=60.0, knot=20.0, thr=DRIFT_THR):
    """Rest anchors (or constant tilt-drift LS) -> attitude; then a drift test: the windowed bias increments' RMS
    temporal variation D decides whether the time-varying correction is kept. Returns q, bg, anchors?, af, D, used."""
    anc = rest_anchors(t, w, a, valid); base = bias_track(t, anc)
    if base is None: q3, b3 = attitude3(t, w, a, valid); base = np.broadcast_to(b3, (len(t), 3)).copy()
    else: q3, b3 = attitude3(t, w, a, valid, bg_track=base)
    q4, b4 = attitude4(t, w, a, valid, tau=tau, bg_track=base, win=win, knot=knot); d = b4 - base; D = float(np.sqrt(((d - d.mean(0)) ** 2).sum(1).mean())); af = fill_gaps(t, a, valid)
    return (q4, b4, bool(anc), af, D, True) if D > thr else (q3, base, bool(anc), af, D, False)
def estimate_v4(tn, w, a, n_blind=0, k_drag=None, sd=SD_DRAG, gate=0.04, force=None, lever=True, zupt=True):
    """FINAL Phase 2 pipeline. Hand-held: Matern(1,2) per axis + zero-velocity rows. Rotor: spring(20,2,.3,1) + drag rows
    + 2 drag-bias + 3 lever-arm states + zero-velocity rows (17 states). Blind rows = model prediction, orientation held."""
    t, wu, au, valid, rows = ingest(tn, w, a); N0 = len(t); q, bg, quiet, af, D, drift = front5(t, wu, au, valid); lock = thrust_lock_local(t, af, valid); dtn = np.median(np.diff(t)); fs = 1.0 / dtn
    zmask = rest_info(t, wu, au, valid)[1] if zupt else np.zeros(N0, bool); wl = lowpass(fill_gaps(t, wu, valid) - bg, fs, 15.0)
    if n_blind:
        ext = lambda x: np.concatenate([x, np.repeat(x[-1:], n_blind, 0)]); t = np.concatenate([t, t[-1] + dtn * np.arange(1, n_blind + 1)]); valid = np.concatenate([valid, np.zeros(n_blind, bool)])
        zmask = np.concatenate([zmask, np.zeros(n_blind, bool)]); q, af, wl = ext(q), ext(af), ext(wl)
    rotor = (lock < gate) if force is None else (force == "drag")
    if rotor: p, xs, Ps = run_joint6(t, q, af, model_spring(*ROTOR), valid, k_drag=K_DRAG if k_drag is None else k_drag, sd=sd, zmask=zmask, wl=wl if lever else None); v = xs[:, [1, 5, 9]]; sig = np.sqrt(Ps[:, 0, 0] + Ps[:, 4, 4] + Ps[:, 8, 8])
    else: p, xs, Ps = lin_smooth_z(t, q, af, valid, model_matern52(*MATERN), zmask); v = xs[:, :, 1]; sig = np.sqrt(3 * Ps[:, 0, 0])
    io = np.concatenate([rows, np.arange(N0, N0 + n_blind)]).astype(np.int64)
    return {"t": t[io], "p": p[io], "q": q[io], "v": v[io], "sig": sig[io], "valid": valid[io], "zupt_rows": int(zmask.sum()), "bg": bg, "lock": lock, "rotor": bool(rotor), "drift_D": D, "drift_used": bool(drift), "n_rows": len(rows),
            "aw": (qrot(q, af) - qrot(q, af)[valid].mean(0))[io]}
