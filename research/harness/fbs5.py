"""fbs5.py - Phase 3 front-end candidates (RESEARCH INSTRUMENT). Each piece is switchable so that it can be ablated:
  repair      : conservative despiking + gravity-magnitude plausibility windows (ingest stage)
  gsmooth     : gravity direction from an integrated-random-walk Kalman smoother (same steady-state response as the zero-phase
                2nd-order Butterworth of v4) with rotation-uncertainty injection at gaps, instead of per-segment filtering
  coarse      : forward-backward complementary filter with integral feedback -> coarse gyro-bias track of any magnitude, used
                only when it is significant; v4's refinement then works on a small residual
  bias_jump_s : accelerometer-bias variance injection after a gap scaled with the gap duration
Nothing here uses ground truth."""
import numpy as np
from numba import njit
from fbs4 import *  # noqa: F401,F403
import fbs3, fbs4

# ------------------------------------------------------------------ 1. repair
def repair(t, w, a, valid, spike_k=10.0, floor_a=15.0, floor_w=1.5, g_lo=3.5, g_hi=17.7, win=1.0):
    """Returns w, a, valid, info. Isolated outliers (residual to a 5-sample median above max(floor, k x local robust sigma))
    are replaced by interpolation and stay valid; windows whose 1 s median |a| is outside [g_lo, g_hi] become invalid."""
    w = w.copy(); a = a.copy(); valid = valid.copy(); fs = 1.0 / np.median(np.diff(t)); N = len(t); X = np.hstack([w, a]); Xf = np.hstack([fill_gaps(t, w, valid), fill_gaps(t, a, valid)])
    pad = np.pad(Xf, ((2, 2), (0, 0)), mode="edge"); med = np.median(np.stack([pad[i:i + N] for i in range(5)]), 0); r = np.abs(Xf - med); n1 = int(fs)
    m = N // n1; sig = np.repeat(np.median(r[:m * n1].reshape(m, n1, 6), 1) * 1.4826, n1, 0) if m else np.zeros((0, 6)); sig = np.vstack([sig, np.repeat(sig[-1:], N - len(sig), 0)]) if len(sig) else np.zeros((N, 6))
    floor = np.array([floor_w] * 3 + [floor_a] * 3); spike = ((r > np.maximum(floor, spike_k * sig)).any(1)) & valid; ns = int(spike.sum())
    if ns:
        good = valid & ~spike
        for j in range(3): w[spike, j] = np.interp(t[spike], t[good], w[good, j]); a[spike, j] = np.interp(t[spike], t[good], a[good, j])
    an = np.linalg.norm(fill_gaps(t, a, valid), axis=1); h = max(1, int(win * fs) // 2); k = max(1, int(fs * 0.1)); idx = np.arange(0, N, k); medn = np.array([np.median(an[max(0, i - h):i + h + 1]) for i in idx]); bad = np.interp(np.arange(N), idx, ((medn < g_lo) | (medn > g_hi)).astype(float)) > 0.0
    if bad.any():
        d = int(0.5 * fs); bad = np.convolve(bad.astype(float), np.ones(2 * d + 1), "same") > 0; bad &= valid; valid = valid & ~bad; w[~valid] = np.nan; a[~valid] = np.nan
    return w, a, valid, dict(spikes=ns, implausible=int(bad.sum()))

# ------------------------------------------------------------------ 2. gravity smoother
@njit(cache=True)
def _irw_smooth(x, valid, dt, q, jump):
    """Per-column integrated-random-walk Kalman smoother. x (N,3); unit measurement variance; jump (N,) variance added to
    the level (and 0.01 x to the slope) before the update at that sample."""
    N = x.shape[0]; out = np.empty((N, 3)); xf = np.empty((N, 2)); Pf = np.empty((N, 2, 2)); xp = np.empty((N, 2)); Pp = np.empty((N, 2, 2))
    q11 = q * dt ** 3 / 3; q12 = q * dt ** 2 / 2; q22 = q * dt
    for c in range(3):
        m0 = 0.0; m1 = 0.0; p00 = 1e6; p01 = 0.0; p11 = 1e2; first = True
        for i in range(N):
            if not first:
                m0 = m0 + dt * m1; a00 = p00 + 2 * dt * p01 + dt * dt * p11 + q11; a01 = p01 + dt * p11 + q12; a11 = p11 + q22; p00 = a00; p01 = a01; p11 = a11
            p00 += jump[i]; p11 += 0.01 * jump[i]; xp[i, 0] = m0; xp[i, 1] = m1; Pp[i, 0, 0] = p00; Pp[i, 0, 1] = p01; Pp[i, 1, 0] = p01; Pp[i, 1, 1] = p11
            if valid[i]:
                if first: m0 = x[i, c]; first = False
                s = p00 + 1.0; k0 = p00 / s; k1 = p01 / s; r = x[i, c] - m0; m0 += k0 * r; m1 += k1 * r; n00 = p00 - k0 * p00; n01 = p01 - k0 * p01; n11 = p11 - k1 * p01; p00 = n00; p01 = n01; p11 = n11
            xf[i, 0] = m0; xf[i, 1] = m1; Pf[i, 0, 0] = p00; Pf[i, 0, 1] = p01; Pf[i, 1, 0] = p01; Pf[i, 1, 1] = p11
        s0 = xf[N - 1, 0]; s1 = xf[N - 1, 1]; out[N - 1, c] = s0
        for i in range(N - 2, -1, -1):
            # C = Pf A' Pp^-1
            f00 = Pf[i, 0, 0]; f01 = Pf[i, 0, 1]; f11 = Pf[i, 1, 1]; b00 = f00 + dt * f01; b01 = f01; b10 = f01 + dt * f11; b11 = f11
            d00 = Pp[i + 1, 0, 0]; d01 = Pp[i + 1, 0, 1]; d11 = Pp[i + 1, 1, 1]; det = d00 * d11 - d01 * d01
            c00 = (b00 * d11 - b01 * d01) / det; c01 = (-b00 * d01 + b01 * d00) / det; c10 = (b10 * d11 - b11 * d01) / det; c11 = (-b10 * d01 + b11 * d00) / det
            e0 = s0 - xp[i + 1, 0]; e1 = s1 - xp[i + 1, 1]; s0 = xf[i, 0] + c00 * e0 + c01 * e1; s1 = xf[i, 1] + c10 * e0 + c11 * e1; out[i, c] = s0
    return out
GS_JUMP = "quad"; GS_K2 = 1.0
def gravity_smooth(t, x, valid, tau=3.0, w_rms=None, sig_m=1.5, c_rot=0.5):
    """Zero-phase gravity estimate in the gyro-integrated frame. Equivalent (steady state) to the 2nd-order Butterworth
    filtfilt at 1/(2 pi tau); a gap of duration T adds level variance (9.81 * c_rot * w_rms * T / sig_m)^2."""
    dt = float(np.median(np.diff(t))); q = dt / tau ** 4; jump = np.zeros(len(t)); st = np.where(valid[1:] & ~valid[:-1])[0] + 1
    if len(st):
        en = np.where(~valid[1:] & valid[:-1])[0] + 1
        for s in st:
            prev = en[en < s]; T = (t[s] - t[prev[-1] - 1]) if len(prev) else 0.0
            # 'linear' (E25-E41): rotation uncertainty c_rot * w_rms * T -> a 5 ms drop already resets the filter, and the official data drop
            # up to 6 % of single samples. 'quad' (E44): the bridged gyro errs by ~ k2 T^2 / 2 (E26: k2 ~ 0.5 MAV, ~1 hand-held), capped at 1 rad.
            th = c_rot * (w_rms if w_rms is not None else 0.5) * T if GS_JUMP == "linear" else min(0.5 * GS_K2 * T * T, 1.0)
            jump[s] = (9.81 * th / sig_m) ** 2
    return _irw_smooth(np.ascontiguousarray(np.nan_to_num(x)), valid, dt, q, jump)

# ------------------------------------------------------------------ 3. coarse bias (forward-backward PI complementary filter)
@njit(cache=True)
def _mahony(w, a, valid, dt, kp, ki, b0):
    N = w.shape[0]; q = np.array([0.0, 0.0, 0.0, 1.0]); b = b0.copy(); B = np.empty((N, 3)); init = False; e = np.zeros(3)
    for i in range(N):
        if valid[i]:
            an = np.sqrt(a[i, 0] ** 2 + a[i, 1] ** 2 + a[i, 2] ** 2)
            if not init and an > 1e-6:
                # initial attitude: rotate body gravity direction onto world z (shortest arc)
                u0 = a[i, 0] / an; u1 = a[i, 1] / an; u2 = a[i, 2] / an; cx = u1; cy = -u0; cz = 0.0; d = u2; nq = np.sqrt(cx * cx + cy * cy + (1 + d) ** 2)
                if nq > 1e-9: q[0] = cx / nq; q[1] = cy / nq; q[2] = cz / nq; q[3] = (1 + d) / nq
                init = True
            if init and an > 4.0 and an < 16.0:
                # predicted gravity direction in body: R^T e_z
                x = q[0]; y = q[1]; z = q[2]; s = q[3]; v0 = 2 * (x * z - s * y); v1 = 2 * (y * z + s * x); v2 = 1 - 2 * (x * x + y * y)
                e[0] = (a[i, 1] * v2 - a[i, 2] * v1) / an; e[1] = (a[i, 2] * v0 - a[i, 0] * v2) / an; e[2] = (a[i, 0] * v1 - a[i, 1] * v0) / an
                b[0] -= ki * e[0] * dt; b[1] -= ki * e[1] * dt; b[2] -= ki * e[2] * dt
            else: e[0] = 0.0; e[1] = 0.0; e[2] = 0.0
            r0 = (w[i, 0] - b[0] + kp * e[0]) * dt; r1 = (w[i, 1] - b[1] + kp * e[1]) * dt; r2 = (w[i, 2] - b[2] + kp * e[2]) * dt
            th = np.sqrt(r0 * r0 + r1 * r1 + r2 * r2); k = np.sin(th / 2) / th if th > 1e-9 else 0.5; d0 = r0 * k; d1 = r1 * k; d2 = r2 * k; d3 = np.cos(th / 2)
            x = q[0]; y = q[1]; z = q[2]; s = q[3]
            q[0] = s * d0 + x * d3 + y * d2 - z * d1; q[1] = s * d1 - x * d2 + y * d3 + z * d0; q[2] = s * d2 + x * d1 - y * d0 + z * d3; q[3] = s * d3 - x * d0 - y * d1 - z * d2
            n = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2); q /= n
        B[i, 0] = b[0]; B[i, 1] = b[1]; B[i, 2] = b[2]
    return B
def coarse_bias(t, w, a, valid, kp=1.0, ki=0.3, smooth=5.0, b0=None):
    """Average of a causal and an anti-causal PI complementary filter's bias states (their lags cancel for a ramp),
    low-passed at `smooth` s. Works for any bias magnitude because the correction is applied in the loop."""
    dt = float(np.median(np.diff(t))); wf = np.ascontiguousarray(np.nan_to_num(w)); af = np.ascontiguousarray(np.nan_to_num(a)); z = np.zeros(3) if b0 is None else b0
    bf = _mahony(wf, af, valid, dt, kp, ki, z); bb = _mahony(-wf[::-1].copy(), af[::-1].copy(), valid[::-1].copy(), dt, kp, ki, -z)[::-1]; b = 0.5 * (bf - bb)
    return lowpass(b, 1.0 / dt, 1.0 / (2 * np.pi * smooth)), bf, -bb

# ------------------------------------------------------------------ attitude with the pieces switched in
def attitude5(t, w, a, valid, tau=3.0, bg_track=None, gsmooth=True, fit="const", ridge=1e-1, win=60.0, knot=20.0, iters=2):
    """attitude3 / attitude4 of v4 with the gravity smoother. fit: None (bias given), 'const' (constant tilt-drift LS),
    'knots' (windowed time-varying refinement). Returns q, bg."""
    fs = 1.0 / np.median(np.diff(t)); ez = np.array([0, 0, 1.0]); N = len(t); af = fill_gaps(t, a, valid); wf = fill_gaps(t, w, valid); w_rms = float(np.sqrt(np.mean(np.sum(np.nan_to_num(w[valid]) ** 2, 1))))
    seg = np.cumsum(np.concatenate([[0], np.diff(valid.astype(np.int8)) != 0])); bg = np.zeros((N, 3)) if bg_track is None else bg_track.copy(); n_it = iters if fit else 0
    for it in range(n_it + 1):
        qg = integrate_gyro(t, wf - bg); x = qrot(qg, af)
        gI = gravity_smooth(t, x, valid, tau, w_rms) if gsmooth else lowpass_seg(x, fs, 1.0 / (2 * np.pi * tau), valid)
        gI = gI / np.linalg.norm(gI, axis=1, keepdims=True); qc = shortest_arc(gI, np.broadcast_to(ez, gI.shape)); q = qmul(qc, qg); q /= np.linalg.norm(q, axis=1, keepdims=True)
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
    return q, bg
COARSE_THR = 0.03
def front6(t, w, a, valid, tau=3.0, win=60.0, knot=20.0, thr=DRIFT_THR, gsmooth=True, coarse=True, coarse_thr=COARSE_THR):
    """v4's front5 with switchable gravity smoother and coarse bias stage. Returns q, bg, anchors?, af, D, used, info."""
    anc = rest_anchors(t, w, a, valid); base = bias_track(t, anc); info = dict(coarse=False, coarse_max=0.0)
    if coarse:
        bc, bf, bb = coarse_bias(t, w, a, valid); ref = np.zeros(3) if base is None else base.mean(0); cm = float(np.abs(bc - ref).max()); info["coarse_max"] = cm
        if cm > coarse_thr:
            info["coarse"] = True; q4, b4 = attitude5(t, w, a, valid, tau, bg_track=bc, gsmooth=gsmooth, fit="knots", win=20.0, knot=5.0, iters=3); return q4, b4, bool(anc), fill_gaps(t, a, valid), float("nan"), True, info
    if base is None: q3, b3 = attitude5(t, w, a, valid, tau, gsmooth=gsmooth, fit="const"); base = np.broadcast_to(b3, (len(t), 3)).copy()
    else: q3, b3 = attitude5(t, w, a, valid, tau, bg_track=base, gsmooth=gsmooth, fit=None)
    q4, b4 = attitude5(t, w, a, valid, tau, bg_track=base, gsmooth=gsmooth, fit="knots", win=win, knot=knot); d = b4 - base; D = float(np.sqrt(((d - d.mean(0)) ** 2).sum(1).mean())); af = fill_gaps(t, a, valid)
    return (q4, b4, bool(anc), af, D, True, info) if D > thr else (q3, base, bool(anc), af, D, False, info)
def prep_v5(tn, w, a, do_repair=True, gsmooth=True, coarse=True, **kw):
    t, wu, au, valid, rows = ingest(tn, w, a); rinfo = dict(spikes=0, implausible=0)
    if do_repair: wu, au, valid, rinfo = repair(t, wu, au, valid)
    q, bg, quiet, af, D, drift, info = front6(t, wu, au, valid, gsmooth=gsmooth, coarse=coarse, **kw); lock = thrust_lock_local(t, af, valid)
    dtn = float(np.median(np.diff(t))); fs = 1.0 / dtn; zmask = rest_info(t, wu, au, valid)[1]; wc = fill_gaps(t, wu, valid) - bg; wl = lowpass(wc, fs, 15.0); aw = qrot(q, af); aw = aw - aw[valid].mean(0); info.update(rinfo)
    return dict(t=t, q=q, bg=bg, af=af, valid=valid, rows=rows, lock=lock, zmask=zmask, wc=wc, wl=wl, aw=aw, ww=qrot(q, wc), fs=fs, dtn=dtn, D=D, drift=drift, N0=len(t), info=info)
def estimate_fe(tn, w, a, n_blind=0, k_drag=None, do_repair=True, gsmooth=True, coarse=True, bias_jump_s=False, **kw):
    """v4 back end on the Phase 3 front end. bias_jump_s scales the post-gap accelerometer-bias variance with gap duration."""
    P = prep_v5(tn, w, a, do_repair, gsmooth, coarse)
    if bias_jump_s: kw["gap_ref"] = 2.0
    e = fbs4.estimate_v6(tn, w, a, n_blind=n_blind, k_drag=k_drag, P=P, **kw); e["info"] = P["info"]; return e

# ------------------------------------------------------------------ 4. OU bridge for the gyro across dropouts (E26)
def bridge_gyro(t, w, valid, tau=0.4, mwin=5.0):
    """Conditional mean of an Ornstein-Uhlenbeck process around the local mean (+-mwin s), pinned to the two end samples.
    Short gaps -> linear interpolation; long gaps -> the local mean in the interior."""
    out = fill_gaps(t, w, valid)
    if valid.all(): return out
    N = len(t); fs = 1.0 / np.median(np.diff(t)); k = int(mwin * fs)
    for i0, i1 in segments(~valid):
        if i0 == 0 or i1 >= N: continue
        T = t[i1] - t[i0 - 1]; tt = t[i0:i1] - t[i0 - 1]; L = w[max(0, i0 - k):i0][valid[max(0, i0 - k):i0]]; R_ = w[i1:i1 + k][valid[i1:i1 + k]]
        mu = 0.5 * (L.mean(0) + R_.mean(0)) if len(L) and len(R_) else (L.mean(0) if len(L) else R_.mean(0)); den = 1 - np.exp(-2 * T / tau)
        s0 = np.exp(-tt / tau) * (1 - np.exp(-2 * (T - tt) / tau)) / den; s1 = np.exp(-(T - tt) / tau) * (1 - np.exp(-2 * tt / tau)) / den
        out[i0:i1] = mu + s0[:, None] * (w[i0 - 1] - mu) + s1[:, None] * (w[i1] - mu)
    return out

# ------------------------------------------------------------------ 5. spline tilt-bias regression (replaces const LS / windowed LS + gate)
def _basis(tm, t0, t1, knot, linear):
    if linear or t1 - t0 < 1.5 * knot: s = (tm - 0.5 * (t0 + t1)) / max(t1 - t0, 1e-9); return np.stack([np.ones_like(tm), s], 1)
    K = int(np.ceil((t1 - t0) / knot)) + 1; u = np.clip((tm - t0) / knot, 0, K - 1 - 1e-9); i = np.floor(u).astype(int); fr = u - i; Phi = np.zeros((len(tm), K)); r = np.arange(len(tm)); Phi[r, i] = 1 - fr; Phi[r, i + 1] = fr; return Phi
def attitude6(t, w, a, valid, tau=3.0, bg0=None, knot=20.0, lam2=3.0, lam0=0.05, iters=6, gsmooth=True, bridge=True, linear=False, health=False, match=True, tol=3e-4):
    """Gyro-bias increments as a linear spline in time (or constant + linear), solved jointly by penalised least squares on
    the drift of the tilt correction; no local windows, hence no edge bias for a ramp. Returns q, bg[, min |g_I|/g]."""
    fs = 1.0 / np.median(np.diff(t)); ez = np.array([0, 0, 1.0]); N = len(t); af = fill_gaps(t, a, valid); wf = bridge_gyro(t, w, valid) if bridge else fill_gaps(t, w, valid); w_rms = float(np.sqrt(np.mean(np.sum(np.nan_to_num(w[valid]) ** 2, 1))))
    seg = np.cumsum(np.concatenate([[0], np.diff(valid.astype(np.int8)) != 0])); bg = np.zeros((N, 3)) if bg0 is None else bg0.copy(); hmin = 1.0; done = False
    for it in range(iters + 1):
        qg = integrate_gyro(t, wf - bg); x = qrot(qg, af); gI = gravity_smooth(t, x, valid, tau, w_rms) if gsmooth else lowpass_seg(x, fs, 1.0 / (2 * np.pi * tau), valid)
        gn = np.linalg.norm(gI, axis=1, keepdims=True); hmin = float(np.percentile(gn[valid], 1) / 9.81) if it == 0 else hmin; gI = gI / gn; qc = shortest_arc(gI, np.broadcast_to(ez, gI.shape)); q = qmul(qc, qg); q /= np.linalg.norm(q, axis=1, keepdims=True)
        if it == iters or done: break
        step = max(1, int(fs * 1.0)); wcr = qlog(qmul(qc[step:], qinv(qc[:-step]))) / (t[step:] - t[:-step])[:, None]; ii = np.where(valid[:-step] & valid[step:] & (seg[:-step] == seg[step:]))[0][::10]
        if len(ii) < 50: break
        y = -wcr[ii, :2]; tm = t[ii] + 0.5 * step / fs; Phi = _basis(tm, t[0], t[-1], knot, linear); K = Phi.shape[1]; n = len(ii)
        if match:
            # the tilt correction answers a drift only through the gravity filter and the 1 s difference: filter the regressor alike
            Rf = Rot.from_quat(q).as_matrix()[:, :2, :].reshape(N, 6); Rs = np.hstack([gravity_smooth(t, Rf[:, :3], valid, tau, w_rms, sig_m=1.5 / 9.81), gravity_smooth(t, Rf[:, 3:], valid, tau, w_rms, sig_m=1.5 / 9.81)]) if gsmooth else lowpass_seg(Rf, fs, 1.0 / (2 * np.pi * tau), valid)
            cs = np.vstack([np.zeros((1, 6)), np.cumsum(Rs, 0)]); A = ((cs[ii + step] - cs[ii]) / step).reshape(n, 2, 3)
        else: A = Rot.from_quat(q[ii]).as_matrix()[:, :2, :]
        M = np.einsum("nk,nij->nikj", Phi, A).reshape(2 * n, 3 * K); Nm = M.T @ M; rhs = M.T @ y.reshape(-1); wbar = n / K; Pn = lam0 * wbar * np.eye(3 * K)
        if K > 2 and not linear: D2 = np.diff(np.eye(K), 2, axis=0); Pn = Pn + lam2 * wbar * np.kron(D2.T @ D2, np.eye(3))
        beta = np.linalg.solve(Nm + Pn, rhs).reshape(K, 3); inc = _basis(t, t[0], t[-1], knot, linear) @ beta; bg = bg + inc; done = bool(np.abs(inc).max() < tol)
    return (q, bg, hmin) if health else (q, bg)
def front7(t, w, a, valid, tau=3.0, mode="spline", gsmooth=True, bridge=True, coarse=True, knot=20.0, lam2=3.0, health_thr=0.9, thr=DRIFT_THR):
    """mode 'v4': front5 logic (const / anchors, windowed refinement behind the D gate) on the new filters;
    mode 'spline' / 'linear': attitude6 always on. coarse: PI-filter bias track as the start when the first pass is unhealthy."""
    anc = rest_anchors(t, w, a, valid); base = bias_track(t, anc); info = dict(coarse=False, hmin=1.0); af = fill_gaps(t, a, valid)
    if mode == "v4":
        if base is None: q3, b3 = attitude5(t, w, a, valid, tau, gsmooth=gsmooth, fit="const"); base = np.broadcast_to(b3, (len(t), 3)).copy()
        else: q3, b3 = attitude5(t, w, a, valid, tau, bg_track=base, gsmooth=gsmooth, fit=None)
        q4, b4 = attitude5(t, w, a, valid, tau, bg_track=base, gsmooth=gsmooth, fit="knots"); d = b4 - base; D = float(np.sqrt(((d - d.mean(0)) ** 2).sum(1).mean()))
        return (q4, b4, bool(anc), af, D, True, info) if D > thr else (q3, base, bool(anc), af, D, False, info)
    q, bg, hmin = attitude6(t, w, a, valid, tau, bg0=base, knot=knot, lam2=lam2, gsmooth=gsmooth, bridge=bridge, linear=(mode == "linear"), health=True); info["hmin"] = hmin
    if coarse and hmin < health_thr:
        bc, bf, bb = coarse_bias(t, bridge_gyro(t, w, valid), af, valid); info["coarse"] = True
        q, bg, h2 = attitude6(t, w, a, valid, tau, bg0=bc, knot=knot, lam2=lam2, iters=10, gsmooth=gsmooth, bridge=bridge, health=True); info["hmin2"] = h2
    return q, bg, bool(anc), af, float("nan"), True, info
def prep_v7(tn, w, a, do_repair=True, **kw):
    t, wu, au, valid, rows = ingest(tn, w, a); rinfo = dict(spikes=0, implausible=0)
    if do_repair: wu, au, valid, rinfo = repair(t, wu, au, valid, g_lo=5.0, g_hi=15.5)
    q, bg, quiet, af, D, drift, info = front7(t, wu, au, valid, **kw); lock = thrust_lock_local(t, af, valid); dtn = float(np.median(np.diff(t))); fs = 1.0 / dtn; zmask = rest_info(t, wu, au, valid)[1]
    wc = (bridge_gyro(t, wu, valid) if kw.get("bridge", True) else fill_gaps(t, wu, valid)) - bg; wl = lowpass(wc, fs, 15.0); aw = qrot(q, af); aw = aw - aw[valid].mean(0); info.update(rinfo)
    return dict(t=t, q=q, bg=bg, af=af, valid=valid, rows=rows, lock=lock, zmask=zmask, wc=wc, wl=wl, aw=aw, ww=qrot(q, wc), fs=fs, dtn=dtn, D=D, drift=drift, N0=len(t), info=info)
def estimate_p3(tn, w, a, n_blind=0, k_drag=None, fe=None, bias_jump_s=False, **kw):
    """Phase 3 candidate: front end options in fe (dict for prep_v7), back end = v4 smoother (estimate_v6 keywords in kw)."""
    P = prep_v7(tn, w, a, **(fe or {}))
    if bias_jump_s: kw["gap_ref"] = 2.0
    e = fbs4.estimate_v6(tn, w, a, n_blind=n_blind, k_drag=k_drag, P=P, **kw); e["info"] = P["info"]; e["drift_used"] = bool(P["drift"]); return e
