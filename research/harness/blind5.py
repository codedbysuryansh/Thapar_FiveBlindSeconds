"""Phase 3 - candidate predictors for the five blind seconds, written against the frozen v5 back end (fbs6.estimate_final).
Every function only sees the estimator output (e, P) at and before the last valid IMU sample; nothing from the blind window or from
ground truth is used. Each position predictor returns the displacement dp (nt, 3) from the last valid pose; each orientation
predictor returns quaternions (nt, 4) in the estimator's world frame. `make(name)` turns a candidate into the callable that
estimate_final(blind=...) expects."""
import numpy as np
import fbs6 as f

def damp(tau, td):
    """Integral of a velocity that decays as exp(-t/td): td*(1-exp(-tau/td)); td None = constant velocity."""
    return tau if td is None else td * (1.0 - np.exp(-tau / td))

def ar_forecast(x10, p, steps, ridge=1e-2):
    """Per-axis AR(p) fit on a 10 Hz velocity history (ridge-regularised), iterated `steps` ahead."""
    out = np.zeros((steps, x10.shape[1]))
    for j in range(x10.shape[1]):
        x = x10[:, j]; n = len(x)
        if n < 4 * p: out[:, j] = x[-1] if n else 0.0; continue
        X = np.stack([x[p - k - 1:n - k - 1] for k in range(p)], 1); y = x[p:]
        c = np.linalg.solve(X.T @ X + ridge * len(y) * (np.var(x) + 1e-12) * np.eye(p), X.T @ y); h = list(x[-p:][::-1])
        for s in range(steps): nx = float(np.dot(c, h[:p])); out[s, j] = nx; h.insert(0, nx)
    return out

def state(e, P):
    """Full-rate position and velocity of the smoother (velocity interpolated from the smoother grid when dec > 1)."""
    xs, tk, N0, t = e["xs"], e["tk"], e["N0"], e["t_full"]
    pos = np.asarray(e["p"], float)
    vel = np.stack([np.interp(t, tk, xs[:, 1 + 4 * j]) for j in range(3)], 1)
    return pos, vel, N0, 1.0 / P["dtn"]

def position_candidates(e, P, nt):
    pos, vel, N0, fs = state(e, P); tau = np.arange(1, nt + 1) / fs; pT, vT = pos[N0 - 1], vel[N0 - 1]
    D = {"hold": np.zeros((nt, 3)), "model": pos[N0:N0 + nt] - pT}
    for td in (0.25, 0.5, 1.0, 2.0, None): D["cv_%s" % ("inf" if td is None else "%g" % td)] = damp(tau, td)[:, None] * vT
    for lam in (0.25, 0.5, 0.75): D["model_x%g" % lam] = lam * D["model"]
    step = max(1, int(round(fs / 10))); v10 = vel[:N0][::-1][::step][::-1][-600:]; t10 = np.arange(1, 51) * 0.1
    for p_ in (5, 10, 20):
        vf = ar_forecast(v10, p_, 50); dp = np.cumsum(vf, 0) * 0.1
        D["ar%d" % p_] = np.stack([np.interp(tau, np.concatenate([[0.0], t10]), np.concatenate([[0.0], dp[:, j]])) for j in range(3)], 1)
    cen = pos[:N0].mean(0)
    for tc in (5.0, 20.0): D["centroid_%g" % tc] = (1 - np.exp(-tau / tc))[:, None] * (cen - pT)
    D["cv1+cen5"] = D["cv_1"] + D["centroid_5"]
    for a_, b_ in (("model", "ar5"), ("model", "ar10"), ("model", "cv_1"), ("model", "cv_0.5"), ("ar10", "hold"), ("ar5", "hold"), ("ar10", "cv_1")):
        D["%s+%s" % (a_, b_)] = 0.5 * (D[a_] + D[b_])
    for td in (0.5, 1.0, 2.0):
        for k in ("ar5", "ar10"):
            inc = np.diff(np.vstack([np.zeros(3), D[k]]), axis=0); D["%s_damp%g" % (k, td)] = np.cumsum(inc * np.exp(-tau / td)[:, None], 0)
    return D

def orientation_candidates(e, P, nt):
    q, wc, vd = P["q"], P["wc"], P["valid"]; fs = 1.0 / P["dtn"]; tau = np.arange(1, nt + 1) / fs; qT = q[-1]; ww = f.qrot(q, wc)
    O = {"hold": np.repeat(qT[None], nt, 0)}
    for win in (1.0, 5.0):
        k = int(win * fs); sel = vd[-k:]; wb = wc[-k:][sel].mean(0) if sel.any() else np.zeros(3); yr = ww[-k:][sel][:, 2].mean() if sel.any() else 0.0
        for td in (0.5, 2.0, None):
            g = damp(tau, td); nm = "%g_%s" % (win, "inf" if td is None else "%g" % td)
            O["gyro_w" + nm] = f.qmul(np.repeat(qT[None], nt, 0), f.qexp(g[:, None] * wb))
            O["yaw_w" + nm] = f.qmul(np.stack([np.zeros(nt), np.zeros(nt), np.sin(yr * g / 2), np.cos(yr * g / 2)], 1), np.repeat(qT[None], nt, 0))
    return O

def make(pos_name="model", ori_name="hold"):
    """Callable for fbs6.estimate_final(blind=...)."""
    def blind(e, P, nt):
        dp = position_candidates(e, P, nt)[pos_name] if pos_name != "model" else None
        if dp is None:
            pos, _, N0, _ = state(e, P); dp = pos[N0:N0 + nt] - pos[N0 - 1]
        qb = orientation_candidates(e, P, nt)[ori_name]
        return dp, qb
    return blind

def final_blind_E53(e, P, nt):
    """E53 candidate (NOT frozen): rotor cv_0.5 + hold; hand-held mean(model, AR5) + yaw_w5_2. E57 fresh seeds: hand-held part
    significantly worse on full TUM sequences (b_abs +0.019 m, 1/6 wins) though better on crops (-0.045 m, 5/6) -> hand-held reverted."""
    pos, vel, N0, fs = state(e, P); tau = np.arange(1, nt + 1) / fs; pT, vT = pos[N0 - 1], vel[N0 - 1]; qT = P["q"][-1]
    if e["rotor"]:
        return damp(tau, 0.5)[:, None] * vT, np.repeat(qT[None], nt, 0)
    step = max(1, int(round(fs / 10))); v10 = vel[:N0][::-1][::step][::-1][-600:]; t10 = np.arange(1, 51) * 0.1
    dp = np.cumsum(ar_forecast(v10, 5, 50), 0) * 0.1
    ar5 = np.stack([np.interp(tau, np.concatenate([[0.0], t10]), np.concatenate([[0.0], dp[:, j]])) for j in range(3)], 1)
    d = 0.5 * ((pos[N0:N0 + nt] - pT) + ar5)
    wc, vd = P["wc"], P["valid"]; k = int(5.0 * fs); sel = vd[-k:]; yr = f.qrot(P["q"][-k:], wc[-k:])[sel][:, 2].mean() if sel.any() else 0.0
    g = damp(tau, 2.0); qb = f.qmul(np.stack([np.zeros(nt), np.zeros(nt), np.sin(yr * g / 2), np.cos(yr * g / 2)], 1), np.repeat(qT[None], nt, 0))
    return d, qb

def final_blind(e, P, nt):
    """FROZEN v5 blind rule (E53 selection on seeds 42, 1-4 + crops; E57 fresh-seed confirmation):
      rotor     : velocity of the smoother at the last valid sample decaying with a 0.5 s time constant (displacement saturates at
                  0.5 s x v): shrinkage of a velocity that IMU-only data barely observe (E57 full: anchored error -0.21 m, 10/11 sequences;
                  b_abs -0.015 m, n.s.); orientation held.
      hand-held : the smoother's own Matern prediction (unchanged v5 default), orientation held."""
    pos, vel, N0, fs = state(e, P); tau = np.arange(1, nt + 1) / fs; pT, vT = pos[N0 - 1], vel[N0 - 1]; qT = P["q"][-1]
    if e["rotor"]: return damp(tau, 0.5)[:, None] * vT, np.repeat(qT[None], nt, 0)
    return pos[N0:N0 + nt] - pT, np.repeat(qT[None], nt, 0)
