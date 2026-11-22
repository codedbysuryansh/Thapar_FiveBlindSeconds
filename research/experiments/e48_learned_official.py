#!/usr/bin/env python3
"""E48 - competitor threat on OFFICIAL data: a learned velocity regressor (the standard learned-inertial-odometry window CNN) trained on
public sequences degraded by the organisers' script (8 realisations per sequence: column orders A/B, seeds 100-102, max-level 0.5/1/1.5;
never the evaluation seeds 42, 1, 2), features from the FROZEN v5 front end (fbs6.front_final), labels = ground-truth velocity in the estimator frame (TRAINING ONLY).
  IN   : trained on all 17 sequences (what a team can do if the hidden data are degraded public sequences: in-pool / memorisation)
  LOEO : the test sequence's environment group excluded (honest generalisation)
Evaluation: official missions (orders A/B; full seeds 42, 1, 2 and crop60 seed 42), v5c alone vs v5c + network velocity rows (sigma 0.5 / 1).
Run: .venv/bin/python research/experiments/e48_learned_official.py train   then   ... eval"""
import sys, os, json, time, numpy as np, torch, torch.nn as nn
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p4, fbs6 as f
CACHE = os.path.join(p4.ROOT, "research/cache/e48"); WIN, HOP_TR, HOP_TE = 400, 20, 16; AR = torch.arange(-WIN // 2, WIN // 2)
REAL = [("A", 100, 1.0), ("B", 100, 1.0), ("A", 101, 1.0), ("B", 101, 1.0), ("A", 102, 0.5), ("B", 102, 0.5), ("A", 102, 1.5), ("B", 102, 1.5)]
CH = (64, 128, 192, 192); EPOCHS = 30
def front(m):
    F = f.front_final(m.tn, m.w, m.a); return F, F["P"], F["rows"], F["L"], F["nt"]
def feats(P): return np.concatenate([P["ww"], P["aw"]], 1).astype(np.float32)
def slot_index(m, P): return np.clip(np.searchsorted(m.s.t - m.s.t[0], P["t"] - 1e-4), 0, len(m.s.t) - 1)
def labels(m, P):
    k = slot_index(m, P); s = m.s; return f.qrot(P["q"], f.qrot(f.qinv(s.gt_q[k]), s.gt_v[k])).astype(np.float32), s.ok[k]
def make_train(args):
    n, o, sd, ml = args; fn = os.path.join(CACHE, "tr_%s_%s_%d_%g.npz" % (n, o, sd, ml))
    if os.path.exists(fn): return fn
    m = p4.official(n, o, seed=sd, max_level=ml); F, P, rows, L, nt = front(m); X = feats(P); Y, ok = labels(m, P)
    c = np.arange(WIN // 2, len(X) - WIN // 2, HOP_TR); vf = np.convolve(P["valid"].astype(float), np.ones(WIN) / WIN, "same"); c = c[ok[c] & (vf[c] > 0.8)]
    np.savez(fn, X=X, Yc=np.stack([Y[i - 25:i + 25].mean(0) for i in c]), c=c); return fn
class Net(nn.Module):
    def __init__(s, ch=CH):
        super().__init__(); L = []; ci = 6; n = WIN
        for co, k in zip(ch, (7, 5, 5, 3)): L += [nn.Conv1d(ci, co, k, 2, 0), nn.ReLU()]; ci = co; n = (n - k) // 2 + 1
        s.f = nn.Sequential(*L, nn.Flatten(), nn.Linear(ci * n, 64), nn.ReLU(), nn.Linear(64, 3))
    def forward(s, x): return s.f(x)
def train(tag, names, dev="mps"):
    fn = os.path.join(CACHE, "net_%s.pt" % tag)
    if os.path.exists(fn): return
    torch.manual_seed(0); Xs, Ys, Cs, off = [], [], [], 0
    for n in names:
        for o, sd, ml in REAL:
            d = np.load(make_train((n, o, sd, ml))); Xs.append(d["X"]); Ys.append(d["Yc"]); Cs.append(d["c"] + off); off += len(d["X"])
    X = torch.tensor(np.concatenate(Xs)); sd_ = X.std(0) + 1e-6; sd_[0] = sd_[1] = (sd_[0] + sd_[1]) / 2; sd_[3] = sd_[4] = (sd_[3] + sd_[4]) / 2; X = (X / sd_).to(dev); Y = torch.tensor(np.concatenate(Ys)).to(dev); C = torch.tensor(np.concatenate(Cs)).to(dev); ar = AR.to(dev)
    net = Net().to(dev); nb = len(C) // 256; opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * nb); t0 = time.time()
    for e in range(EPOCHS):
        perm = torch.randperm(len(C), device=dev); tot = 0.0
        for i in range(nb):
            b = perm[i * 256:(i + 1) * 256]; x = X[C[b][:, None] + ar[None]].permute(0, 2, 1).clone(); y = Y[b].clone(); B = len(b); th = torch.rand(B, device=dev) * 6.2831853; c, s_ = torch.cos(th)[:, None], torch.sin(th)[:, None]
            for a_, b_ in ((0, 1), (3, 4)): xa, xb = x[:, a_].clone(), x[:, b_].clone(); x[:, a_] = c * xa - s_ * xb; x[:, b_] = s_ * xa + c * xb
            ya, yb = y[:, 0].clone(), y[:, 1].clone(); y[:, 0] = c[:, 0] * ya - s_[:, 0] * yb; y[:, 1] = s_[:, 0] * ya + c[:, 0] * yb
            loss = nn.functional.smooth_l1_loss(net(x), y); opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tot += float(loss.detach())
        if e % 5 == 4 or e == EPOCHS - 1: print("%s epoch %d/%d loss %.4f  %.0f s" % (tag, e + 1, EPOCHS, tot / nb, time.time() - t0), flush=True)
    torch.save({"sd": sd_.cpu(), "state": {k: v.cpu() for k, v in net.state_dict().items()}, "params": int(sum(p.numel() for p in net.parameters())), "n_windows": int(len(C))}, fn)
_N = {}
def predict(tag, P):
    if tag not in _N:
        ck = torch.load(os.path.join(CACHE, "net_%s.pt" % tag), weights_only=False); net = Net(); net.load_state_dict(ck["state"]); net.eval(); _N[tag] = (net, ck)
    net, ck = _N[tag]; X = torch.tensor(feats(P)) / ck["sd"]; c = np.arange(WIN // 2, len(X) - WIN // 2, HOP_TE); vf = np.convolve(P["valid"].astype(float), np.ones(WIN) / WIN, "same"); c = c[vf[c] > 0.8]
    with torch.no_grad(): v = torch.cat([net(X[torch.tensor(c[i:i + 512])[:, None] + AR[None]].permute(0, 2, 1)) for i in range(0, len(c), 512)]).numpy().astype(float)
    return c, v
def evaluate(args):
    n, o, cname, ckw = args; torch.set_num_threads(1); m = p4.official(n, o, **ckw); F, P, rows, L, nt = front(m); out = {"motionless": p4.score(m, np.zeros((len(m.tn), 3)))}; vr = {}
    e = f.back_final(F, k_drag=m.kd); out["v5"] = p4.score(m, e["p"], e["q"]); Yt, ok = labels(m, P)
    for proto, tag in (("IN", "IN"), ("LOEO", "LOEO_" + f.fbs.group(n))):
        if not os.path.exists(os.path.join(CACHE, "net_%s.pt" % tag)): continue
        c, v = predict(tag, P); vr[proto] = [float(np.sqrt(np.mean(np.sum((v[ok[c]] - Yt[c][ok[c]]) ** 2, 1)))), float(np.sqrt(np.mean(np.sum(Yt[c][ok[c]] ** 2, 1))))]
        for sv in (0.5, 1.0):
            e = f.back_final(F, k_drag=m.kd, vel=(c, v, sv)); out["%s|%g" % (proto, sv)] = p4.score(m, e["p"], e["q"])
    return dict(name=n, order=o, cfg=cname, m=out, vr=vr, rotor=F["rotor"])
if __name__ == "__main__":
    names = f.available(); os.makedirs(CACHE, exist_ok=True)
    if sys.argv[1] == "train":
        with Pool(8) as p: p.map(make_train, [(n, o, sd, ml) for n in names for o, sd, ml in REAL], chunksize=1)
        train("IN", names)
        for g in sorted({f.fbs.group(n) for n in names}): train("LOEO_" + g, [x for x in names if f.fbs.group(x) != g])
    else:
        CF = [("full", dict(seed=s)) for s in (42, 1, 2)] + [("crop60", dict(seed=42, crop=0.6))]
        with Pool(8) as p: res = p.map(evaluate, [(n, o, c, kw) for o in ("A", "B") for c, kw in CF for n in names], chunksize=1)
        json.dump(res, open(os.path.join(p4.ROOT, "research/results/e48_learned_official.json"), "w"))
        for g, sel in (("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room"))):
            rr = [r for r in res if sel(r["name"])]; print("\n[%s | %d missions] velocity RMSE: %s" % (g, len(rr), "  ".join("%s %.3f (truth RMS %.3f)" % (k, np.mean([r["vr"][k][0] for r in rr]), np.mean([r["vr"][k][1] for r in rr])) for k in rr[0]["vr"])))
            for v in rr[0]["m"]: print("  %-12s " % v + "  ".join("%s %.3f" % (k, np.nanmean([r["m"][v][k] for r in rr])) for k in ("ate", "wp", "b_abs", "drop", "rpe_t")))
