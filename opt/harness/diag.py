#!/usr/bin/env python3
"""diag.py - run one candidate on a few cached missions, keep the outputs and plot them against ground truth (diagnostics only).

  diag.py CAND TAG [TAG ...] [--out notes/diag]
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbsopt as H                                                           # noqa: E402
import numpy as np                                                           # noqa: E402

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cand"); ap.add_argument("tags", nargs="+"); ap.add_argument("--out", default=os.path.join(H.OPT, "notes", "diag"))
    a = ap.parse_args(); import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    M, cfg = H.load_candidate(a.cand); od = os.path.join(a.out, a.cand); os.makedirs(od, exist_ok=True)
    for tag in a.tags:
        n = next(s for s in H.SEQS if tag.startswith(s + "_")); zf = os.path.join(H.CACHE, "missions", tag + ".zip"); tno = np.load(os.path.join(H.CACHE, "missions", tag + ".npz"))["tno"]
        est = os.path.join(od, tag + ".txt"); kd = H.k_loso(n); _p, info = M.run(zf, est, k_drag=kd)
        ens, p, q = H.EV.read_estimate(est); m = H.EV.mission(tno, H.gt_cached(n)); s = H.EV.score(m, p, q)
        A = m.ok; Rm, tt = H.EV.umeyama(p[A], m.gp[A]); pa = p @ Rm.T + tt; e = np.linalg.norm(pa - m.gp, axis=1)
        fig, ax = plt.subplots(2, 2, figsize=(13, 9))
        ax[0, 0].plot(m.gp[A, 0], m.gp[A, 1], "k-", lw=0.8, label="ground truth"); ax[0, 0].plot(pa[A, 0], pa[A, 1], "r-", lw=0.8, label="estimate (aligned)")
        ax[0, 0].plot(m.gp[m.blind & A, 0], m.gp[m.blind & A, 1], "k-", lw=2.5); ax[0, 0].plot(pa[m.blind & A, 0], pa[m.blind & A, 1], "r-", lw=2.5); ax[0, 0].axis("equal"); ax[0, 0].legend(); ax[0, 0].set_title("%s  ATE %.3f (motionless %.3f)" % (tag, s["ate"], np.nan))
        for j, c in enumerate("xyz"):
            ax[0, 1].plot(m.t[A], m.gp[A, j], "-", color="C%d" % j, lw=0.8, label="gt " + c); ax[0, 1].plot(m.t[A], pa[A, j], "--", color="C%d" % j, lw=0.8, label="est " + c)
        ax[0, 1].legend(ncol=2, fontsize=7); ax[0, 1].set_xlabel("t [s]")
        ax[1, 0].plot(m.t[A], e[A], "k-", lw=0.8); ax[1, 0].set_ylabel("position error [m]"); ax[1, 0].set_xlabel("t [s]")
        g = np.where(m.gap_after >= 5)[0]; ax[1, 0].plot(m.t[g], np.zeros(len(g)), "b|", ms=8)
        vg = np.gradient(m.gp, m.t, axis=0); ve = np.gradient(pa, m.t, axis=0); k = max(1, int(m.fs * 0.5)); sm = lambda x: np.stack([np.convolve(x[:, i], np.ones(k) / k, "same") for i in range(3)], 1)
        ax[1, 1].plot(m.t[A], np.linalg.norm(sm(vg)[A, :2], axis=1), "k-", lw=0.8, label="gt horizontal speed"); ax[1, 1].plot(m.t[A], np.linalg.norm(sm(ve)[A, :2], axis=1), "r-", lw=0.8, label="est horizontal speed")
        ax[1, 1].legend(); ax[1, 1].set_xlabel("t [s]")
        fig.suptitle("%s  %s  info %s" % (a.cand, tag, json.dumps({k: v for k, v in info.items() if k in ("rotor", "guard", "mode", "lock", "h", "D")}, default=float)), fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(od, tag + ".png"), dpi=90); plt.close(fig)
        print("%s ate %.3f wp %.3f b_abs %.3f -> %s" % (tag, s["ate"], s["wp"], s["b_abs"], os.path.join(od, tag + ".png")))

if __name__ == "__main__":
    main()
