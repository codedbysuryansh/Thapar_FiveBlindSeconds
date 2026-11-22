#!/usr/bin/env python3
"""diag2.py TAG CAND [CAND ...] - overlay several candidates on one mission: position error, horizontal velocity error and
top view (each aligned on its own), to see where a change helps or hurts. Diagnostics only."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbsopt as H                                                           # noqa: E402
import numpy as np                                                           # noqa: E402
def main():
    tag = sys.argv[1]; cands = sys.argv[2:]; import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = next(s for s in H.SEQS if tag.startswith(s + "_")); zf = os.path.join(H.CACHE, "missions", tag + ".zip"); tno = np.load(os.path.join(H.CACHE, "missions", tag + ".npz"))["tno"]
    m = H.EV.mission(tno, H.gt_cached(n)); A = m.ok; od = os.path.join(H.OPT, "notes", "diag2"); os.makedirs(od, exist_ok=True)
    fig, ax = plt.subplots(3, 1, figsize=(13, 12)); k = max(1, int(m.fs * 1.0)); sm = lambda x: np.stack([np.convolve(x[:, i], np.ones(k) / k, "same") for i in range(3)], 1)
    vg = sm(np.gradient(m.gp, m.t, axis=0))
    for i, c in enumerate(cands):
        M, cfg = H.load_candidate(c); est = os.path.join(od, "%s__%s.txt" % (tag, c)); M.run(zf, est, k_drag=H.k_loso(n)); _ns, p, q = H.EV.read_estimate(est)
        Rm, tt = H.EV.umeyama(p[A], m.gp[A]); pa = p @ Rm.T + tt; e = np.linalg.norm(pa - m.gp, axis=1); s = H.EV.score(m, p, q)
        ve = sm(np.gradient(pa, m.t, axis=0)); dv = np.linalg.norm((ve - vg)[:, :2], axis=1)
        ax[0].plot(m.t[A], e[A], lw=0.9, label="%s ATE %.3f wp %.3f" % (c, s["ate"], s["wp"])); ax[1].plot(m.t[A], dv[A], lw=0.8, label="%s rms %.3f" % (c, np.sqrt(np.mean(dv[A] ** 2))))
        if i == 0: ax[2].plot(m.gp[A, 0], m.gp[A, 1], "k-", lw=1.2, label="ground truth")
        ax[2].plot(pa[A, 0], pa[A, 1], lw=0.8, label=c)
    ax[0].set_ylabel("position error [m]"); ax[1].set_ylabel("|horizontal velocity error| [m/s] (1 s mean)"); ax[2].axis("equal")
    for a in ax: a.legend(fontsize=8)
    fig.suptitle(tag); fig.tight_layout(); f = os.path.join(od, "%s__%s.png" % (tag, "_vs_".join(c[:12] for c in cands))); fig.savefig(f, dpi=85); print(f)
if __name__ == "__main__":
    main()
