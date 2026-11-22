"""an08: drag-row bias tracks (smoother states 12, 13) under the loose (champion) and tight drag-bias priors, against the offset
implied by ground truth: e_i . f_block - (-k * (R e_i) . v_gt) - lever terms, using the estimator's own attitude and rows.
Diagnostic only. Usage: an08_bias_tracks.py TAG"""
import os, sys, json, numpy as np, importlib.util
sys.path.insert(0, "~/SamsungPRISM_FiveBlindSeconds/opt/harness")
import fbsopt as H
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
tag = sys.argv[1]; n = next(s for s in H.SEQS if tag.startswith(s + "_")); zf = os.path.join(H.CACHE, "missions", tag + ".zip")
spec = importlib.util.spec_from_file_location("md", "~/SamsungPRISM_FiveBlindSeconds/opt/src/main_dev.py"); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
fig, ax = plt.subplots(3, 1, figsize=(13, 11)); tno = np.load(os.path.join(H.CACHE, "missions", tag + ".npz"))["tno"]; gns, gp, gq = H.gt_cached(n)
for lab, sets in (("loose (champion)", dict(SC0=None, SC=0.15, TAU_C=100.0)), ("tight 0.15/0.03/1000", dict(SC0=0.15, SC=0.03, TAU_C=1000.0))):
    for k_, v_ in sets.items(): setattr(M, k_, v_)
    M.TRACE = {}; D, sol, info = M.estimate(M.Source(zf), H.k_loso(n)); T = M.TRACE; X = T["rotor_states"]; tk = T["tk"]; Mz = T["Mz"]
    ax[0].plot(tk, X[:, 12], lw=0.9, label="c1 " + lab); ax[0].plot(tk, X[:, 13], lw=0.9, ls="--", label="c2 " + lab)
    # ground-truth velocity in the estimator's world frame at the block times: align estimate to GT (SE3), rotate GT velocity back
    t0 = tno[0]; gt_t = (gns - t0) * 1e-9; vg = np.stack([np.interp(tk, gt_t[1:-1], np.gradient(gp[:, i], gt_t)[1:-1]) for i in range(3)], 1)
    pk = T["pk"]; pg = np.stack([np.interp(tk, gt_t, gp[:, i]) for i in range(3)], 1); ok = (tk > gt_t[0]) & (tk < gt_t[-1])
    Rm, tt = H.EV.umeyama(pk[ok], pg[ok]); vgE = vg @ Rm                         # GT velocity in the estimate's frame (row vectors: v_E = Rm^T v_G)
    vk = T["vk"]; ax[1].plot(tk[ok], np.linalg.norm((vk - vgE)[ok, :2], axis=1), lw=0.8, label="|v_h error| " + lab)
    if lab.startswith("loose"):
        fi = Mz.fidx; sel = fi >= 0; j = fi[sel]; HV = Mz.HV[j]; YD = Mz.YD[j]; vv = vgE[sel]
        for i in range(2): off = YD[:, i] - np.einsum("nj,nj->n", HV[:, i], vv); ax[2].plot(tk[sel], off, ",", alpha=0.4, label="implied offset c%d (raw blocks)" % (i + 1))
        for i in range(2):
            off = YD[:, i] - np.einsum("nj,nj->n", HV[:, i], vv); kk = 50; sm = np.convolve(off, np.ones(kk) / kk, "same"); ax[0].plot(tk[sel], sm, "k-" if i == 0 else "k--", lw=1.2, label="GT-implied c%d (2 s mean)" % (i + 1))
for a in ax: a.legend(fontsize=7)
ax[0].set_ylabel("drag-row bias [m/s^2]"); ax[1].set_ylabel("|horizontal velocity error| [m/s]"); ax[2].set_ylim(-1.5, 1.5)
fig.suptitle(tag); fig.tight_layout(); out = "~/SamsungPRISM_FiveBlindSeconds/opt/notes/diag2/an08_%s.png" % tag; fig.savefig(out, dpi=85); print(out)
