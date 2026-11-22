#!/usr/bin/env python3
"""E22 - the learned velocity prior as a serious candidate: capacity, robustness to degradation, in-pool versus held-out.
Window regressor of the standard learned-inertial-odometry form (2 s of gravity-aligned gyro + specific force -> velocity at
the window centre), valid convolutions (so that a streaming C implementation computes the same features), trained on 8
corruption realisations per sequence with yaw / noise / bias augmentation. Ground truth is used for training labels only.
  IN-S / IN-M / IN-L : trained on all 17 public sequences (224k / 585k / 1.72M parameters)  -> in-pool performance
  LOEO-M             : the test sequence's whole environment excluded                        -> held-out performance
Test: missions with degradation seeds never used in training (clean, L1, L2, L3), failure at 60 %, 5 blind seconds; the
velocities enter the v4 smoother as rows with sigma 0.25 / 0.5 / 1.0 m/s.
Run: .venv/bin/python research/experiments/e22_learned_prior.py train   (MPS)   then   ... eval"""
import sys, os, json, time, numpy as np, torch, torch.nn as nn
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import fbs4 as f, p3
CACHE = os.path.join(p3.ROOT, "research/cache/e22"); WIN, HOP_TR, HOP_TE = 400, 20, 16; AR = torch.arange(-WIN // 2, WIN // 2)
REAL = [("clean", 0), ("L1", 11), ("L1", 12), ("L2", 13), ("L2", 14), ("L3", 15), ("noise", 16), ("gb05", 17)]
CH = {"S": (32, 64, 96, 96), "M": (64, 128, 192, 192), "L": (128, 256, 384, 384)}; EPOCHS = {"S": 40, "M": 40, "L": 30}
feats_from = lambda P: np.concatenate([P["ww"], P["aw"]], 1).astype(np.float32)
def label(s, sl, P, meta): return f.qrot(P["q"], f.qrot(f.qinv(s.gt_q[sl]), s.gt_v[sl]) @ meta["Rr"].T).astype(np.float32)
def make_train(args):
    n, lvl, seed = args; fn = os.path.join(CACHE, "tr_%s_%s_%d.npz" % (n, lvl, seed))
    if os.path.exists(fn): return fn
    s = f.load(n); sl = slice(0, len(s.t)); w, a, vd, meta = f.degrade(s.t, s.w, s.a, p3.LEVELS[lvl], seed=1000 * seed + sum(n.encode()) % 997); P = f.prep_v4(s.tn, w, a); X = feats_from(P); Y = label(s, sl, P, meta)
    c = np.arange(WIN // 2, len(X) - WIN // 2, HOP_TR); c = c[s.ok[c] & (np.array([P["valid"][i - WIN // 2:i + WIN // 2].mean() for i in c]) > 0.9)]
    np.savez(fn, X=X, Yc=np.stack([Y[i - 25:i + 25].mean(0) for i in c]), c=c); return fn
class Net(nn.Module):
    def __init__(s, ch):
        super().__init__(); L = []; ci = 6; n = WIN
        for co, k in zip(ch, (7, 5, 5, 3)): L += [nn.Conv1d(ci, co, k, 2, 0), nn.ReLU()]; ci = co; n = (n - k) // 2 + 1
        s.f = nn.Sequential(*L, nn.Flatten(), nn.Linear(ci * n, 64), nn.ReLU(), nn.Linear(64, 3))
    def forward(s, x): return s.f(x)
def train(tag, names, size, dev="mps"):
    fn = os.path.join(CACHE, "net_%s.pt" % tag)
    if os.path.exists(fn): return
    torch.manual_seed(0); Xs, Ys, Cs, off = [], [], [], 0
    for n in names:
        for lvl, seed in REAL:
            d = np.load(make_train((n, lvl, seed))); Xs.append(d["X"]); Ys.append(d["Yc"]); Cs.append(d["c"] + off); off += len(d["X"])
    X = torch.tensor(np.concatenate(Xs)); sd = X.std(0) + 1e-6; sd[0] = sd[1] = (sd[0] + sd[1]) / 2; sd[3] = sd[4] = (sd[3] + sd[4]) / 2; X = (X / sd).to(dev); Y = torch.tensor(np.concatenate(Ys)).to(dev); C = torch.tensor(np.concatenate(Cs)).to(dev); ar = AR.to(dev); sdd = sd.to(dev)
    net = Net(CH[size]).to(dev); ep = EPOCHS[size]; nb = len(C) // 256; opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=ep * nb); t0 = time.time()
    for e in range(ep):
        perm = torch.randperm(len(C), device=dev); tot = 0.0
        for i in range(nb):
            b = perm[i * 256:(i + 1) * 256]; x = X[C[b][:, None] + ar[None]].permute(0, 2, 1).clone(); y = Y[b].clone(); B = len(b)
            th = torch.rand(B, device=dev) * 6.2831853; c, s_ = torch.cos(th)[:, None], torch.sin(th)[:, None]
            for a_, b_ in ((0, 1), (3, 4)):
                xa, xb = x[:, a_].clone(), x[:, b_].clone(); x[:, a_] = c * xa - s_ * xb; x[:, b_] = s_ * xa + c * xb
            ya, yb = y[:, 0].clone(), y[:, 1].clone(); y[:, 0] = c[:, 0] * ya - s_[:, 0] * yb; y[:, 1] = s_[:, 0] * ya + c[:, 0] * yb
            x[:, :3] += torch.randn_like(x[:, :3]) * (torch.rand(B, 1, 1, device=dev) * 0.05) / sdd[:3, None]; x[:, 3:] += (torch.randn_like(x[:, 3:]) * (torch.rand(B, 1, 1, device=dev) * 0.5) + torch.randn(B, 3, 1, device=dev) * 0.1) / sdd[3:, None]
            loss = nn.functional.smooth_l1_loss(net(x), y); opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tot += float(loss.detach())
        if e % 5 == 4 or e == ep - 1: print("%s epoch %d/%d loss %.4f  %.0f s" % (tag, e + 1, ep, tot / nb, time.time() - t0), flush=True)
    torch.save({"sd": sd, "state": {k: v.cpu() for k, v in net.state_dict().items()}, "size": size, "n_windows": int(len(C)), "final_loss": tot / nb, "params": int(sum(p.numel() for p in net.parameters())), "train_s": time.time() - t0}, fn)
_NETS = {}
def predict(tag, P):
    if tag not in _NETS:
        ck = torch.load(os.path.join(CACHE, "net_%s.pt" % tag), weights_only=False); net = Net(CH[ck["size"]]); net.load_state_dict(ck["state"]); net.eval(); _NETS[tag] = (net, ck)
    net, ck = _NETS[tag]; X = torch.tensor(feats_from(P)) / ck["sd"]; c = np.arange(WIN // 2, len(X) - WIN // 2, HOP_TE); c = c[np.array([P["valid"][i - WIN // 2:i + WIN // 2].mean() for i in c]) > 0.9]
    with torch.no_grad(): v = torch.cat([net(X[torch.tensor(c[i:i + 512])[:, None] + AR[None]].permute(0, 2, 1)) for i in range(0, len(c), 512)]).numpy().astype(float) if len(c) else np.zeros((0, 3))
    return c, v, ck
def evaluate(args):
    n, lvl, seed = args; torch.set_num_threads(1); m = p3.mission(n, lvl, 1, seed); P = f.prep_v4(m.tn, m.w, m.a); vt = label(m.s, slice(0, m.iT + 1), P, m.meta); out = {"motionless": p3.score(m, np.zeros((m.iT + 1 + m.nb, 3)))}; vr = {}
    e = f.estimate_v5(m.tn, m.w, m.a, n_blind=m.nb, k_drag=m.kd, P=P); out["v4"] = p3.score(m, e["p"], e["q"])
    for proto, tag in (("IN-S", "IN_S"), ("IN-M", "IN_M"), ("IN-L", "IN_L"), ("LOEO-M", "LOEO_M_" + f.fbs.group(n))):
        if not os.path.exists(os.path.join(CACHE, "net_%s.pt" % tag)): continue
        c, v, ck = predict(tag, P); ok = m.s.ok[c]; vr[proto] = [float(np.sqrt(np.mean(np.sum((v[ok] - vt[c][ok]) ** 2, 1)))), float(np.sqrt(np.mean(np.sum(vt[c][ok] ** 2, 1))))]
        for sv in (0.25, 0.5, 1.0):
            e = f.estimate_v5(m.tn, m.w, m.a, n_blind=m.nb, k_drag=m.kd, vel_fn=lambda P_: (c, v), sv=sv, P=P); out["%s|%g" % (proto, sv)] = p3.score(m, e["p"], e["q"])
    return dict(name=n, lvl=lvl, seed=seed, vr=vr, m=out)
if __name__ == "__main__":
    names = f.available(); os.makedirs(CACHE, exist_ok=True); mode = sys.argv[1]
    if mode == "train":
        with Pool(6) as p: p.map(make_train, [(n, l, sd) for n in names for l, sd in REAL], chunksize=1)
        for size in sys.argv[2:] or ["S", "M", "L"]: train("IN_" + size, names, size)
        if len(sys.argv) == 2:
            for g in sorted({f.fbs.group(n) for n in names}): train("LOEO_M_" + g, [x for x in names if f.fbs.group(x) != g], "M")
    else:
        levels = ("clean", "L1", "L2", "L3")
        with Pool(8) as p: res = p.map(evaluate, [(n, l, sd) for l in levels for n in names for sd in range(2)], chunksize=1)
        json.dump(res, open(os.path.join(p3.ROOT, "research/results/e22_learned_prior.json"), "w")); var = ["motionless", "v4"] + [k for k in res[0]["m"] if "|" in k]
        for lvl in levels:
            rr = [r for r in res if r["lvl"] == lvl]; print("\n[%s] velocity RMSE (truth RMS %.3f): %s" % (lvl, np.mean([list(r["vr"].values())[0][1] for r in rr]), "  ".join("%s %.3f (EuRoC %.3f, TUM %.3f)" % (k, np.mean([r["vr"][k][0] for r in rr]), np.mean([r["vr"][k][0] for r in rr if not r["name"].startswith("room")]), np.mean([r["vr"][k][0] for r in rr if r["name"].startswith("room")])) for k in rr[0]["vr"])))
        p3.table(res, var, levels)
