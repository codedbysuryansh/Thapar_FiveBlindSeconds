#!/usr/bin/env python3
"""E18 - what can a compact LEARNED velocity model do here, and is a hybrid worth shipping?  (competitor threat model)
A 1-D CNN maps a 2 s gravity-aligned IMU window (v4 front end: bias-corrected gyro and mean-removed specific force in the
estimator's world frame, yaw-augmented) to the velocity at the window centre. Ground truth is used for TRAINING labels only.
Protocols: IN     = trained on all 17 public sequences (what a team does if the evaluation data are the public sequences),
           INbig  = same with 4x the channels and 2x the epochs (how far memorisation goes),
           LOEO   = the test sequence's whole environment excluded, LOSO = the test sequence excluded (honest generalisation).
Test streams are fresh realisations (clean, and L2 with a seed never used in training), cut at 60 % with 5 blind seconds.
The predicted velocities enter the SAME v4 smoother as extra rows (sigma 0.25 / 0.5 / 1.0 m/s). OUR simulator, OUR metrics.
Run: .venv/bin/python research/experiments/e18_learned_threat.py"""
import sys, os, json, numpy as np, torch, torch.nn as nn
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbs4 as f
from e10_mission import LEVELS, KD
CACHE = os.path.join(f.fbs.ROOT, "research/cache/e18"); WIN, HOP_TR, HOP_TE = 400, 20, 10; TRAIN_REAL = [("clean", 0), ("L2", 12)]; AR = torch.arange(-WIN // 2, WIN // 2)
feats_from = lambda P: np.concatenate([P["ww"], P["aw"]], 1).astype(np.float32)
def label(s, sl, P, meta): return f.qrot(P["q"], f.qrot(f.qinv(s.gt_q[sl]), s.gt_v[sl]) @ meta["Rr"].T).astype(np.float32)
def make_train(args):
    n, lvl, seed = args; fn = os.path.join(CACHE, "tr_%s_%s_%d.npz" % (n, lvl, seed))
    if os.path.exists(fn): return fn
    s = f.load(n); sl = slice(0, len(s.t)); w, a, vd, meta = f.degrade(s.t, s.w, s.a, LEVELS[lvl], seed=1000 * seed + sum(n.encode()) % 997); P = f.prep_v4(s.tn, w, a); X = feats_from(P); Y = label(s, sl, P, meta)
    c = np.arange(WIN // 2, len(X) - WIN // 2, HOP_TR); c = c[s.ok[c] & (np.array([P["valid"][i - WIN // 2:i + WIN // 2].mean() for i in c]) > 0.9)]
    np.savez(fn, X=X, Yc=np.stack([Y[i - 25:i + 25].mean(0) for i in c]), c=c); return fn
class Net(nn.Module):
    def __init__(s, ch=(32, 64, 96, 96)):
        super().__init__(); L = []; ci = 6
        for co, k in zip(ch, (7, 5, 5, 3)): L += [nn.Conv1d(ci, co, k, 2, k // 2), nn.ReLU()]; ci = co
        s.f = nn.Sequential(*L, nn.Flatten(), nn.Linear(ci * (WIN // 16), 64), nn.ReLU(), nn.Linear(64, 3))
    def forward(s, x): return s.f(x)
def yaw_aug(x, y):
    th = torch.rand(x.shape[0]) * 2 * np.pi; c, s_ = torch.cos(th)[:, None], torch.sin(th)[:, None]; x = x.clone(); y = y.clone()
    for a, b in ((0, 1), (3, 4)):
        xa, xb = x[:, a].clone(), x[:, b].clone(); x[:, a] = c * xa - s_ * xb; x[:, b] = s_ * xa + c * xb
    ya, yb = y[:, 0].clone(), y[:, 1].clone(); y[:, 0] = c[:, 0] * ya - s_[:, 0] * yb; y[:, 1] = s_[:, 0] * ya + c[:, 0] * yb; return x, y
CH = {False: (32, 64, 96, 96), True: (64, 128, 192, 192)}
def train(args):
    tag, names, big = args; fn = os.path.join(CACHE, "net_%s.pt" % tag)
    if os.path.exists(fn): return tag
    torch.set_num_threads(4 if big else 1); torch.manual_seed(0); Xs, Ys, Cs, off = [], [], [], 0
    for n in names:
        for lvl, seed in TRAIN_REAL:
            d = np.load(make_train((n, lvl, seed))); Xs.append(d["X"]); Ys.append(d["Yc"]); Cs.append(d["c"] + off); off += len(d["X"])
    X = torch.tensor(np.concatenate(Xs)); sd = X.std(0) + 1e-6; X = X / sd; Y = torch.tensor(np.concatenate(Ys)); C = torch.tensor(np.concatenate(Cs)); net = Net(CH[big]); ep = 40 if big else 20
    opt = torch.optim.Adam(net.parameters(), 1e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep * (len(C) // 256 + 1))
    for e in range(ep):
        perm = torch.randperm(len(C)); tot = 0.0
        for i in range(0, len(C), 256):
            b = perm[i:i + 256]; xb, yb = yaw_aug(X[C[b][:, None] + AR[None]].permute(0, 2, 1), Y[b]); loss = nn.functional.smooth_l1_loss(net(xb), yb); opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tot += float(loss) * len(b)
    torch.save({"sd": sd, "state": net.state_dict(), "big": big, "n_windows": int(len(C)), "final_loss": tot / len(C), "params": int(sum(p.numel() for p in net.parameters()))}, fn); return tag
def vel_fn_for(tag):
    ck = torch.load(os.path.join(CACHE, "net_%s.pt" % tag), weights_only=False); net = Net(CH[ck["big"]]); net.load_state_dict(ck["state"]); net.eval()
    def fn(P):
        X = torch.tensor(feats_from(P)) / ck["sd"]; c = np.arange(WIN // 2, len(X) - WIN // 2, HOP_TE); c = c[np.array([P["valid"][i - WIN // 2:i + WIN // 2].mean() for i in c]) > 0.9]
        with torch.no_grad(): v = net(X[torch.tensor(c)[:, None] + AR[None]].permute(0, 2, 1)).numpy().astype(float)
        return c, v
    return fn, ck
def evaluate(args):
    n, lvl = args; torch.set_num_threads(1); s = f.load(n); fs = 1 / np.median(np.diff(s.t)); nb = int(round(5 * fs)); oi = np.where(s.ok)[0]; t0, t1 = s.t[oi[0]], s.t[oi[-1]]; iT = int(np.searchsorted(s.t, t0 + 0.6 * (t1 - t0)))
    while not s.ok[iT:iT + nb + 1].all() and iT > 2000: iT -= 40
    sl = slice(0, iT + 1); w, a, vd, meta = f.degrade(s.t[sl], s.w[sl], s.a[sl], LEVELS[lvl], seed=5000 + 7 + sum(n.encode()) % 997); tn = s.tn[sl]
    M = iT + 1 + nb; gp = s.gt_p[:M]; ok = s.ok[:M]; t = s.t[:M]; okp = ok & (np.arange(M) <= iT); b = np.arange(iT + 1, M)
    cp = np.searchsorted(s.t, np.linspace(t0 + 10, s.t[iT], 6)).clip(oi[0], iT); cp = np.array([c if ok[c] else oi[np.argmin(np.abs(oi - c))] for c in cp])
    def score(p):
        Rm, tt = f.umeyama(p[okp], gp[okp]); er = np.linalg.norm(p @ Rm.T + tt - gp, axis=1); ga = [er[(t >= g0) & (t < g0 + d) & okp].mean() for g0, d in meta["gaps"] if ((t >= g0) & (t < g0 + d) & okp).sum() > 10]
        return [float(er[cp].mean()), float(er[b].mean()), float(np.linalg.norm((p[b] - p[iT]) @ Rm.T - (gp[b] - gp[iT]), axis=1).mean()), float(np.mean(ga)) if ga else float("nan"), float(np.sqrt(np.mean(er[okp] ** 2))), f.rpe_disp(t[okp], p[okp], gp[okp], Rm)]
    P = f.prep_v4(tn, w, a); kd = float(np.mean([v["fits"]["2.0"]["k"] for m, v in KD.items() if m != n])); vt = label(s, sl, P, meta); out = {"motionless": score(np.zeros((M, 3)))}
    out["v4"] = score(f.estimate_v5(tn, w, a, n_blind=nb, k_drag=kd, P=P)["p"]); vr = {}
    for proto, tag in (("IN", "IN"), ("INbig", "INbig"), ("LOEO", "LOEO_" + f.fbs.group(n)), ("LOSO", "LOSO_" + n)):
        fn, ck = vel_fn_for(tag); c, v = fn(P); m = s.ok[c]; vr[proto] = [float(np.sqrt(np.mean(np.sum((v[m] - vt[c][m]) ** 2, 1)))), float(np.sqrt(np.mean(np.sum(vt[c][m] ** 2, 1)))), ck["params"]]
        for sv in (0.25, 0.5, 1.0): out["%s|%g" % (proto, sv)] = score(f.estimate_v5(tn, w, a, n_blind=nb, k_drag=kd, vel_fn=lambda P_: (c, v), sv=sv, P=P)["p"])
        vi = np.stack([np.interp(np.arange(iT + 1), c, v[:, j]) for j in range(3)], 1); pi = np.cumsum(vi, 0) * P["dtn"]; out[proto + "|integrate"] = score(np.vstack([pi, np.repeat(pi[-1:], nb, 0)]))
    return dict(name=n, level=lvl, vr=vr, m=out)
if __name__ == "__main__":
    names = f.available(); os.makedirs(CACHE, exist_ok=True)
    with Pool(10) as p: p.map(make_train, [(n, l, sd) for n in names for l, sd in TRAIN_REAL], chunksize=1)
    jobs = [("INbig", names, True), ("IN", names, False)] + [("LOEO_" + g, [m for m in names if f.fbs.group(m) != g], False) for g in sorted({f.fbs.group(n) for n in names})] + [("LOSO_" + n, [m for m in names if m != n], False) for n in names]
    with Pool(9) as p: p.map(train, jobs, chunksize=1)
    with Pool(10) as p: res = p.map(evaluate, [(n, l) for l in ("clean", "L2") for n in names], chunksize=1)
    json.dump(res, open(os.path.join(f.fbs.ROOT, "research/results/e18_learned_threat.json"), "w")); K = ["wp", "b_abs", "b_anc", "gap", "ate", "rpe"]
    for grp, sel in (("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room")), ("ALL", lambda n: True)):
        for lvl in ("clean", "L2"):
            rr = [r for r in res if r["level"] == lvl and sel(r["name"])]; print("\n[%s | %s | %d missions]  velocity RMSE (true RMS %.3f m/s): %s" % (grp, lvl, len(rr), np.mean([r["vr"]["IN"][1] for r in rr]), "  ".join("%s %.3f" % (k, np.mean([r["vr"][k][0] for r in rr])) for k in ("IN", "INbig", "LOEO", "LOSO"))))
            for v in ["motionless", "v4"] + ["%s|%s" % (pr, x) for pr in ("IN", "INbig", "LOEO", "LOSO") for x in ("0.25", "0.5", "1", "integrate")]:
                print("  %-16s " % v + "  ".join("%s %.2f" % (k, np.nanmean([r["m"][v][i] for r in rr])) for i, k in enumerate(K)))
