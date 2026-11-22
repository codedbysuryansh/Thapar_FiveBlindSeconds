#!/usr/bin/env python3
"""E63 - can the IRW gravity smoother (the most valuable v5 component, E58 +1.8 m when removed) run at 50 or 25 Hz in Phase 4?
gravity_smooth_dec: valid samples averaged over blocks of d (measurement variance 1/count), the same continuous-time intensity
(q = dt/tau^4 in the ORIGINAL dt), gap jumps (T^2 law) summed into the block that ends the gap, smoothed block series linearly
interpolated back to the full rate. Replaces gravity_smooth everywhere (attitude5b, attitude6 incl. the matched regressors).
Official full sequences, seeds 42 and 1, orders A/B (68 missions). Accuracy cost vs the full-rate smoother."""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p4, fbs5, fbs6 as f, blind5 as B
ORIG = fbs5.gravity_smooth
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
def one(args):
    n, o, sd = args; m = p4.official(n, o, seed=sd); out = {}
    for k, d in (("full", None), ("dec4", 4), ("dec8", 8), ("dec16", 16)):
        g = ORIG if d is None else (lambda *a, d=d, **kw: gravity_smooth_dec(*a, d=d, **kw)); fbs5.gravity_smooth = g; f.gravity_smooth = g
        F = f.front_final(m.tn, m.w, m.a); r = f.back_final(F, k_drag=m.kd, blind=B.final_blind); out[k] = p4.score(m, r["p"], r["q"]); out[k]["rotor"] = r["rotor"]
    fbs5.gravity_smooth = ORIG; f.gravity_smooth = ORIG; return dict(name=n, order=o, seed=sd, m=out)
if __name__ == "__main__":
    names = f.available()
    with Pool(10) as p: res = p.map(one, [(n, o, sd) for o in ("A", "B") for sd in (42, 1) for n in names], chunksize=1)
    json.dump(res, open(os.path.join(p4.ROOT, "research/results/e63_decimated_gravity.json"), "w")); rng = np.random.default_rng(0)
    for g, sel in (("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room"))):
        rr = [r for r in res if sel(r["name"])]; seqs = sorted({r["name"] for r in rr}); line = "[%s] " % g
        for k in ("full", "dec4", "dec8", "dec16"):
            d = np.array([np.mean([r["m"][k]["ate"] - r["m"]["full"]["ate"] for r in rr if r["name"] == s]) for s in seqs]); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
            line += " | %s ATE %.3f ori %.1f d %+.3f [%+.3f, %+.3f] gate %d" % (k, np.mean([r["m"][k]["ate"] for r in rr]), np.mean([r["m"][k]["ori"] for r in rr]), d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), sum(r["m"][k]["rotor"] for r in rr))
        print(line)
