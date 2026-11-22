#!/usr/bin/env python3
"""train.py --fold {inpool,MH,V1,V2,room1..room6} [--cand X104-combo5] - learned body-frame velocity regressor (1-D CNN,
heteroscedastic Gaussian output), the threat a team with learned inertial odometry would bring (TLIO/RoNIN-style).
Training: TRAIN-set windows (seeds 5001-5004) of the sequences outside the held-out group; test: DEV-set windows (seeds
1001-1004) of the held-out sequences (inpool: all 17 sequences train AND test - memorisation threat). Writes per-mission
predictions (body velocity, log-variance) for fusion, and a velocity-accuracy summary against zero velocity and the
estimator's own smoother velocity."""
import os, sys, glob, json, argparse, time, numpy as np
import torch, torch.nn as nn
sys.path.insert(0, "~/SamsungPRISM_FiveBlindSeconds/opt/harness"); import fbsopt as H   # noqa: E402
DATA = "~/SamsungPRISM_FiveBlindSeconds/opt/learned/data"; OUT = "~/SamsungPRISM_FiveBlindSeconds/opt/learned/runs"
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.c = nn.Sequential(nn.Conv1d(6, 32, 7, padding=3), nn.GELU(), nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(), nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.GELU(),
                            nn.Conv1d(64, 96, 5, stride=2, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(1))
        s.h = nn.Sequential(nn.Linear(96 + 3, 96), nn.GELU(), nn.Linear(96, 6))
    def forward(s, x, g):
        z = s.c(x).squeeze(-1); o = s.h(torch.cat([z, g], 1)); return o[:, :3], o[:, 3:].clamp(-6, 3)
def load(cand, setname, names):
    Xs, Gs, Ys, info = [], [], [], []
    for n, sp, tag in H.missions(setname):
        if n not in names: continue
        z = np.load(os.path.join(DATA, "%s__%s.npz" % (cand, tag))); ok = z["label_ok"] & (z["valid_frac"] > 0.6)
        Xs.append(z["X"][ok]); Gs.append(z["G"][ok]); Ys.append(z["label"][ok]); info.append((tag, n, np.flatnonzero(ok), z))
    return np.concatenate(Xs), np.concatenate(Gs), np.concatenate(Ys), info
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", required=True); ap.add_argument("--cand", default="X104-combo5"); ap.add_argument("--epochs", type=int, default=14); a = ap.parse_args()
    torch.manual_seed(20261122); np.random.seed(20261122); dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if a.fold == "inpool": test = set(H.SEQS); train = set(H.SEQS)
    elif a.fold.startswith("room"): test = {a.fold}; train = set(H.SEQS) - test
    else: test = {s for s in H.SEQS if H.env_group(s) == a.fold}; train = set(H.SEQS) - test
    X, G, Y, _ = load(a.cand, "TRAIN", train); mu = X.reshape(-1, 6).mean(0); sd = X.reshape(-1, 6).std(0) + 1e-6
    Xt = torch.tensor(((X - mu) / sd).transpose(0, 2, 1), dtype=torch.float32); Gt = torch.tensor(G); Yt = torch.tensor(Y)
    net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4); nb = int(np.ceil(len(Xt) / 256)); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=a.epochs * nb)
    t0 = time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(len(Xt)); tot = 0.0
        for b in range(nb):
            i = perm[b * 256:(b + 1) * 256]; m, lv = net(Xt[i].to(dev), Gt[i].to(dev)); y = Yt[i].to(dev)
            loss = (0.5 * ((m - y) ** 2 * torch.exp(-lv) + lv)).mean(); opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tot += float(loss)
        print("fold %s epoch %d loss %.4f (%.0f s)" % (a.fold, ep, tot / nb, time.time() - t0), flush=True)
    od = os.path.join(OUT, "%s__%s" % (a.cand, a.fold)); os.makedirs(od, exist_ok=True); net.eval(); summ = {}
    torch.save(dict(state=net.state_dict(), mu=mu, sd=sd), os.path.join(od, "model.pt"))
    _X, _G, _Y, info = load(a.cand, "DEV", test)
    for tag, n, okidx, z in info:
        x = torch.tensor(((z["X"] - mu) / sd).transpose(0, 2, 1), dtype=torch.float32); g = torch.tensor(z["G"])
        with torch.no_grad(): m, lv = net(x.to(dev), g.to(dev))
        m = m.cpu().numpy(); lv = lv.cpu().numpy(); np.savez(os.path.join(od, tag + ".npz"), slot=z["slot"], t=z["t"], v_body=m, logvar=lv)
        lab = z["label"][okidx]; pr = m[okidx]; ph = z["vphys_b"][okidx]
        summ[tag] = dict(name=n, rmse_net=float(np.sqrt(np.mean(np.sum((pr - lab) ** 2, 1)))), rmse_zero=float(np.sqrt(np.mean(np.sum(lab ** 2, 1)))),
                         rmse_phys=float(np.sqrt(np.mean(np.sum((ph - lab) ** 2, 1)))), rmse_fused_ideal=float(np.sqrt(np.mean(np.sum((0.5 * (pr + ph) - lab) ** 2, 1)))),
                         sigma_mean=float(np.sqrt(np.exp(lv[okidx]).sum(1)).mean()))
    json.dump(summ, open(os.path.join(od, "summary.json"), "w"), indent=1)
    for d in ("EuRoC", "TUM-VI"):
        rr = [v for v in summ.values() if H.dataset(v["name"]) == d]
        if rr: print("fold %s %s: velocity RMSE net %.3f | estimator %.3f | zero %.3f | mean(net, estimator) %.3f | predicted sigma %.3f m/s (%d missions)" % (
            a.fold, d, np.mean([r["rmse_net"] for r in rr]), np.mean([r["rmse_phys"] for r in rr]), np.mean([r["rmse_zero"] for r in rr]), np.mean([r["rmse_fused_ideal"] for r in rr]), np.mean([r["sigma_mean"] for r in rr]), len(rr)))
if __name__ == "__main__":
    main()
