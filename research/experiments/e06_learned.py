#!/usr/bin/env python3
"""E06 - small learned velocity prior (RoNIN/TLIO-style 1-D CNN on gravity-aligned 1 s windows) under honest splits.
Research instrument. Ground truth labels are used for TRAINING sequences only; the held-out sequence is scored only.
Protocols: LOSO (other 16 sequences), LOEO (exclude the test environment), LOPO (train on the other platform only).
Outputs: velocity RMSE, ATE of integrated velocity, ATE of the hybrid (learned velocity as a pseudo-measurement in the
stationary-GP smoother). Run: .venv/bin/python research/experiments/e06_learned.py"""
import sys, os, json, numpy as np, torch, torch.nn as nn
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import fbs
WIN, STRIDE, GP = 200, 20, (1, 2, 20, 0.1)
def feats(n):
    s = fbs.load(n); q, bg, quiet = fbs.estimate_attitude(s.t, s.w, s.a)
    ww = fbs.qrot(q, s.w - bg); aw = fbs.qrot(q, s.a); aw[:, 2] -= aw[:, 2].mean(); X = np.concatenate([ww, aw], 1).astype(np.float32)
    vt = fbs.qrot(q, fbs.qrot(fbs.qinv(s.gt_q), s.gt_v)).astype(np.float32)          # GT velocity expressed in the estimator's frame
    c = np.arange(WIN // 2, len(s.t) - WIN // 2, STRIDE); keep = np.array([s.ok[i - WIN // 2:i + WIN // 2].all() for i in c])
    Xw = np.stack([X[i - WIN // 2:i + WIN // 2].T for i in c]); Y = np.stack([vt[i - WIN // 2:i + WIN // 2].mean(0) for i in c])
    return {"n": n, "X": Xw, "Y": Y, "c": c, "keep": keep}
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.f = nn.Sequential(nn.Conv1d(6, 32, 7, 2, 3), nn.ReLU(), nn.Conv1d(32, 64, 5, 2, 2), nn.ReLU(), nn.Conv1d(64, 64, 5, 2, 2), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
    def forward(s, x): return s.f(x)
def yaw_aug(x, y):
    th = torch.rand(x.shape[0]) * 2 * np.pi; c, s_ = torch.cos(th)[:, None], torch.sin(th)[:, None]; x = x.clone(); y = y.clone()
    for a, b in ((0, 1), (3, 4)):
        xa, xb = x[:, a].clone(), x[:, b].clone(); x[:, a] = c * xa - s_ * xb; x[:, b] = s_ * xa + c * xb
    ya, yb = y[:, 0].clone(), y[:, 1].clone(); y[:, 0] = c[:, 0] * ya - s_[:, 0] * yb; y[:, 1] = s_[:, 0] * ya + c[:, 0] * yb
    return x, y
_D = {}
def fold(args):
    test, train_names, tag = args; torch.set_num_threads(1); torch.manual_seed(0)
    if not _D:
        for n in fbs.available(): _D[n] = feats(n)
    D = _D
    Xtr = np.concatenate([D[n]["X"][D[n]["keep"]] for n in train_names]); Ytr = np.concatenate([D[n]["Y"][D[n]["keep"]] for n in train_names])
    sd = Xtr.std((0, 2), keepdims=True) + 1e-6; Xt = torch.tensor(Xtr / sd); Yt = torch.tensor(Ytr); net = Net(); opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(12):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            xb, yb = yaw_aug(Xt[perm[i:i + 256]], Yt[perm[i:i + 256]]); loss = nn.functional.smooth_l1_loss(net(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
    d = D[test]
    with torch.no_grad(): vp = net(torch.tensor(d["X"] / sd)).numpy().astype(float)
    s = fbs.load(test); q, bg, quiet = fbs.estimate_attitude(s.t, s.w, s.a); ok = s.ok; c = d["c"]
    vrmse = float(np.sqrt(np.mean(np.sum((vp[d["keep"]] - d["Y"][d["keep"]]) ** 2, 1)))); vzero = float(np.sqrt(np.mean(np.sum(d["Y"][d["keep"]] ** 2, 1))))
    vfull = np.stack([np.interp(s.t, s.t[c], vp[:, j]) for j in range(3)], 1); p_int = np.cumsum(vfull * np.diff(s.t, prepend=s.t[0])[:, None], 0)
    r = {"test": test, "tag": tag, "n_train": int(len(Xt)), "vel_rmse": vrmse, "vel_rms_truth": vzero, "motionless": fbs.ate(np.zeros((ok.sum(), 3)), s.gt_p[ok]), "integrate": fbs.ate(p_int[ok], s.gt_p[ok])}
    sn = fbs.hf_std(fbs.world_acc_centered(q, s.a)); r["gp"] = fbs.ate(fbs.run_joint(s.t, q, s.a, GP, sn)["p"][ok], s.gt_p[ok])
    for sg in [0.5, 1.0, 2.0]: r["hybrid_%g" % sg] = fbs.ate(fbs.run_joint(s.t, q, s.a, GP, sn, vel_meas=(c, vp, sg))["p"][ok], s.gt_p[ok])
    return r
if __name__ == "__main__":
    names = fbs.available()
    jobs = []
    for n in names:
        jobs.append((n, [m for m in names if m != n], "LOSO")); jobs.append((n, [m for m in names if fbs.group(m) != fbs.group(n)], "LOEO")); jobs.append((n, [m for m in names if fbs.platform(m) != fbs.platform(n)], "LOPO"))
    with Pool(10) as p: res = p.map(fold, jobs, chunksize=1)
    json.dump(res, open(os.path.join(fbs.ROOT, "research/results/e06_learned.json"), "w"), indent=1)
    print("protocol group   n | vel RMSE (m/s) / RMS of true velocity | ATE m: motionless  GP   integrate-learned  hybrid s=0.5  s=1  s=2 | skill%: GP  integrate  hybrid(best fixed s)")
    for tag in ["LOSO", "LOEO", "LOPO"]:
        for g in ["MH", "V1", "V2", "TUM", "ALL"]:
            rr = [r for r in res if r["tag"] == tag and (g == "ALL" or fbs.group(r["test"]) == g)]; f = lambda k: np.mean([r[k] for r in rr]); sk = lambda k: 100 * np.mean([1 - r[k] / r["motionless"] for r in rr])
            print("%-5s %-4s %3d | %.3f / %.3f | %.3f  %.3f  %.3f   %.3f  %.3f  %.3f | %5.1f  %6.1f  %5.1f %5.1f %5.1f" % (tag, g, len(rr), f("vel_rmse"), f("vel_rms_truth"), f("motionless"), f("gp"), f("integrate"), f("hybrid_0.5"), f("hybrid_1"), f("hybrid_2"), sk("gp"), sk("integrate"), sk("hybrid_0.5"), sk("hybrid_1"), sk("hybrid_2")))
