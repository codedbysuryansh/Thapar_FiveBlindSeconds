#!/usr/bin/env python3
"""extract.py SET [--cand X104-combo5] - learned-model data: for every mission of SET, run the candidate's estimator (its own
attitude, no ground truth) and cut 1 s windows every 0.5 s of the gap-filled body-frame gyro and accelerometer (100 Hz),
with the estimator's gravity direction in the body frame and its attitude; labels = ground-truth body-frame velocity (for
training and scoring only). Also the candidate smoother's own velocity at the window centres (the physics baseline)."""
import os, sys, json, argparse, numpy as np
from multiprocessing import get_context
sys.path.insert(0, "~/SamsungPRISM_FiveBlindSeconds/opt/harness"); import fbsopt as H   # noqa: E402
OUT = "~/SamsungPRISM_FiveBlindSeconds/opt/learned/data"
def qrot(q, v):
    u = q[..., :3]; w = q[..., 3:4]; c = np.cross(u, v); return v + 2 * w * c + 2 * np.cross(u, c)
def qconj(q): return q * np.array([-1.0, -1.0, -1.0, 1.0])
def gt_velocity(n, tno):
    """(t [s from first row], body-frame GT velocity, ok) at the GT / mocap times."""
    if n.startswith("room"):
        ns, p, q = H.gt_cached(n); t = (ns - tno[0]) * 1e-9; lo = np.searchsorted(t, t - 0.1); hi = np.searchsorted(t, t + 0.1) - 1
        lo = np.clip(lo, 0, len(t) - 1); hi = np.clip(hi, 0, len(t) - 1); dt = t[hi] - t[lo]; ok = (dt > 0.12) & (dt < 0.3)
        v = np.where(ok[:, None], (p[hi] - p[lo]) / np.maximum(dt, 1e-6)[:, None], 0.0)
    else:
        g = np.loadtxt(os.path.join(H.mav0(n), "state_groundtruth_estimate0", "data.csv"), delimiter=",", comments="#")
        t = (g[:, 0].astype(np.int64) - tno[0]) * 1e-9; v = g[:, 8:11]; q = g[:, [5, 6, 7, 4]]; ok = np.ones(len(t), bool)
    qn = q / np.linalg.norm(q, axis=1, keepdims=True); vb = qrot(qconj(qn), v); return t, vb, ok
def job(args):
    cand, tag = args; M, cfg = H.load_candidate(cand); n = next(s for s in H.SEQS if tag.startswith(s + "_")); tno = np.load(os.path.join(H.CACHE, "missions", tag + ".npz"))["tno"]
    M.TRACE = {}; D, sol, info = M.estimate(M.Source(os.path.join(H.CACHE, "missions", tag + ".zip")), H.k_loso(n)); T = M.TRACE; F = T["F"]; att = T["att"]
    N0 = D.N0; half = int(round(0.5 * D.fs)); step = half; cs = np.arange(half, N0 - half, step)
    X = np.zeros((len(cs), 100, 6), np.float32); G = np.zeros((len(cs), 3), np.float32); Q = np.zeros((len(cs), 4)); vf = np.zeros(len(cs), np.float32)
    for i, c in enumerate(cs):
        a, b = c - half, c + half; w = np.nan_to_num(F.wf(a, b)); acc = np.nan_to_num(D.filled(a, b, "a")); x = np.concatenate([w, acc], 1)
        if len(x) % 2: x = x[:-1]
        x = x.reshape(-1, 2, 6).mean(1); m = min(100, len(x)); X[i, :m] = x[:m]; vf[i] = D.valid[a:b].mean()
        q = att.q(c // M.CH)[c - (c // M.CH) * M.CH]; Q[i] = q; G[i] = qrot(qconj(q[None]), np.array([[0, 0, 1.0]]))[0]
    t_c = D.t[cs]; tk = T["tk"]; vk = T["vk"]; vphys_w = np.stack([np.interp(t_c, tk, vk[:, j]) for j in range(3)], 1); vphys_b = qrot(qconj(Q), vphys_w)
    tg, vb, okg = gt_velocity(n, tno); lab = np.stack([np.interp(t_c, tg, vb[:, j]) for j in range(3)], 1); okl = (t_c > tg[0]) & (t_c < tg[-1]) & (np.interp(t_c, tg, okg.astype(float)) > 0.99)
    np.savez_compressed(os.path.join(OUT, "%s__%s.npz" % (cand, tag)), X=X, G=G, Q=Q, t=t_c, slot=cs, valid_frac=vf, label=lab.astype(np.float32), label_ok=okl, vphys_b=vphys_b.astype(np.float32),
                        rotor=bool(info["rotor"]), name=n)
    return tag, len(cs)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("set"); ap.add_argument("--cand", default="X104-combo5"); ap.add_argument("--procs", type=int, default=11); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); tags = [t for _n, _sp, t in H.missions(a.set) if not os.path.exists(os.path.join(OUT, "%s__%s.npz" % (a.cand, t)))]
    with get_context("spawn").Pool(a.procs) as pool:
        tot = 0
        for tag, k in pool.imap_unordered(job, [(a.cand, t) for t in tags]): tot += k
    print("extracted %d missions, %d windows" % (len(tags), tot))
