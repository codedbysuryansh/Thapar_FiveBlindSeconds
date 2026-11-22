#!/usr/bin/env python3
"""E18b - free diagnostic from the cached e18 nets: does disagreement between networks separate 'this sequence was in my
training set' from 'it was not'?  For test sequence n: the 16 LOSO nets that excluded ANOTHER sequence all saw n (in-pool
ensemble); LOSO_n and LOEO_group(n) did not (out-of-pool pair). Compares error and spread of the two sets."""
import sys, os, json, numpy as np, torch
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbs4 as f, p3
import e18_learned_threat as e18
def one(args):
    n, lvl = args; torch.set_num_threads(1); m = p3.mission(n, lvl, 1, 7); P = f.prep_v4(m.tn, m.w, m.a); names = f.available(); s = m.s; sl = slice(0, m.iT + 1); vt = e18.label(s, sl, P, m.meta)
    def pred(tag): fn, ck = e18.vel_fn_for(tag); return fn(P)
    c, _ = pred("IN"); ok = s.ok[c]; ins = np.stack([pred("LOSO_" + x)[1] for x in names if x != n]); outs = np.stack([pred("LOSO_" + n)[1], pred("LOEO_" + f.fbs.group(n))[1]]); vt = vt[c]
    rms = lambda x: float(np.sqrt(np.mean(np.sum(x[ok] ** 2, -1))))
    sp_in2 = np.sqrt(((ins[:2] - ins[:2].mean(0)) ** 2).sum(-1).sum(0)); sp_out2 = np.sqrt(((outs - outs.mean(0)) ** 2).sum(-1).sum(0)); er_in = np.linalg.norm(ins[:2].mean(0) - vt, axis=1); er_out = np.linalg.norm(outs.mean(0) - vt, axis=1)
    return dict(name=n, lvl=lvl, e_in=rms(ins.mean(0) - vt), e_in2=rms(ins[:2].mean(0) - vt), s_in16=float(np.sqrt(((ins - ins.mean(0)) ** 2).sum(-1).mean(0)[ok].mean())), s_in2=float(np.sqrt((sp_in2[ok] ** 2).mean())),
                e_out=rms(outs.mean(0) - vt), s_out2=float(np.sqrt((sp_out2[ok] ** 2).mean())), med_in=float(np.median(sp_in2[ok])), med_out=float(np.median(sp_out2[ok])),
                corr_in=float(np.corrcoef(sp_in2[ok], er_in[ok])[0, 1]), corr_out=float(np.corrcoef(sp_out2[ok], er_out[ok])[0, 1]), q_in=np.percentile(sp_in2[ok], [10, 50, 90]).tolist(), q_out=np.percentile(sp_out2[ok], [10, 50, 90]).tolist())
if __name__ == "__main__":
    names = f.available()
    with Pool(9) as p: res = p.map(one, [(n, l) for l in ("clean", "L2") for n in names], chunksize=1)
    json.dump(res, open(os.path.join(p3.ROOT, "research/results/e18b_spread_diag.json"), "w"))
    print("seq            lvl   | in-pool: err(16) err(2) spread(2) | out-of-pool: err(2) spread(2) | median spread in/out | corr(spread,err) in/out")
    for r in res: print("%-15s %-5s | %.3f %.3f %.3f | %.3f %.3f | %.3f / %.3f | %.2f / %.2f" % (r["name"], r["lvl"], r["e_in"], r["e_in2"], r["s_in2"], r["e_out"], r["s_out2"], r["med_in"], r["med_out"], r["corr_in"], r["corr_out"]))
    for l in ("clean", "L2"):
        rr = [r for r in res if r["lvl"] == l]; print(l, "mean: in err %.3f spread %.3f | out err %.3f spread %.3f | sequences where median out-spread > median in-spread: %d/%d ; where 10th pct out > 90th pct in: %d" % (np.mean([r["e_in2"] for r in rr]), np.mean([r["s_in2"] for r in rr]), np.mean([r["e_out"] for r in rr]), np.mean([r["s_out2"] for r in rr]), sum(r["med_out"] > r["med_in"] for r in rr), len(rr), sum(r["q_out"][0] > r["q_in"][2] for r in rr)))
