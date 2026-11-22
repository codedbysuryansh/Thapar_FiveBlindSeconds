"""an01: where are the drones genuinely still (ground truth speed ~ 0), what do the IMU statistics look like there, and does the
champion's rest detector see it? Diagnostic only (ground truth used for analysis, not by any estimator)."""
import os, sys, json, numpy as np
sys.path.insert(0, "~/SamsungPRISM_FiveBlindSeconds/opt/harness")
import fbsopt as H
M, _ = H.load_candidate("CHAMP-1.0.0")
def win_stats(t, w, a, n):
    m = len(t) // n; W = w[:m * n].reshape(m, n, 3); A = a[:m * n].reshape(m, n, 3)
    return t[:m * n].reshape(m, n).mean(1), W.std(1).max(1), A.std(1).max(1), np.linalg.norm(W.mean(1), axis=1), np.linalg.norm(A.mean(1), axis=1)
out = {}
for n in H.EUROC + H.TUMVI[:2]:
    tag = n + "_A1001"; zf = os.path.join(H.CACHE, "missions", tag + ".zip"); src = M.Source(zf); D = M.Data(src); D.unit_check(); D.medians()
    N = D.N0; t = D.t[:N]; v = D.valid[:N]; W = np.nan_to_num(D.decode(0, N, "w")); A = np.nan_to_num(D.decode(0, N, "a"))
    anchors, zm = M.rest_info(D)
    gns, gp, gq = H.gt_cached(n); tn0 = np.load(os.path.join(H.CACHE, "missions", tag + ".npz"))["tno"][0]
    gt = (gns - tn0) * 1e-9; spd = np.linalg.norm(np.gradient(gp, gt, axis=0), axis=1)
    nwin = int(0.5 * D.fs); tc, gs, as_, gm, am = win_stats(t, W, A, nwin)
    sp = np.interp(tc, gt, spd, left=np.nan, right=np.nan)
    still = sp < 0.03; moving = sp > 0.2
    zfrac = zm[:N].mean()
    res = dict(still_frac=float(np.nanmean(still)), rest_mask_frac=float(zfrac), n_anchor=len(anchors),
               gyro_std_still=float(np.nanmedian(gs[still])) if still.any() else None, gyro_std_moving=float(np.nanmedian(gs[moving])) if moving.any() else None,
               acc_std_still=float(np.nanmedian(as_[still])) if still.any() else None, acc_std_moving=float(np.nanmedian(as_[moving])) if moving.any() else None)
    # how much of the genuinely still time does the rest mask cover?
    tm = t[zm[:N]]; covered = np.array([np.any(np.abs(tm - x) < 0.25) for x in tc[still]]) if still.any() and len(tm) else np.zeros(0, bool)
    res["still_covered_by_rest_mask"] = float(covered.mean()) if len(covered) else 0.0
    # still segments longer than 3 s
    segs = []; i = 0; s = still.astype(int)
    while i < len(s):
        if s[i]:
            j = i
            while j < len(s) and s[j]: j += 1
            if (j - i) * 0.5 >= 3.0: segs.append((round(float(tc[i]), 1), round(float(tc[j - 1]), 1)))
            i = j
        else: i += 1
    res["still_segments_ge3s"] = segs; out[n] = res
    print(n, json.dumps(res))
json.dump(out, open("~/SamsungPRISM_FiveBlindSeconds/opt/notes/an01_ground_segments.json", "w"), indent=1)
