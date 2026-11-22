#!/usr/bin/env python3
"""E53 analysis: blind-window candidates on the frozen v5 pipeline. Tables per (group, failure point), regret across cells,
paired bootstrap over sequences, orientation candidates, and forecast combination (stacking) / selection under grouped CV
(rotor: leave-one-environment-out over MH / V1 / V2; hand-held: leave-one-sequence-out within TUM)."""
import sys, os, json, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
res = json.load(open(os.path.join(ROOT, "research/results/e53_blind_final.json")))
GR = {"EuRoC": lambda r: not r["name"].startswith("room"), "TUM": lambda r: r["name"].startswith("room")}; CF = ("full", "crop80", "crop60", "crop40")
P = [k for k in res[0]["m"] if not k.startswith("ori:") and not k.startswith("ORACLE")]; ORC = [k for k in res[0]["m"] if k.startswith("ORACLE")]
O = [k for k in res[0]["m"] if k.startswith("ori:")]
def mean(rr, k, x): return float(np.nanmean([r["m"][k][x] for r in rr]))
cell = {}
for g, sel in GR.items():
    for c in CF:
        rr = [r for r in res if sel(r) and r["cfg"] == c]; cell[(g, c)] = rr
        print("\n=== %s %s: %d missions; true blind displacement %.2f m; speed at failure %.2f m/s; velocity error at failure %.2f m/s; rotor %d/%d" % (
            g, c, len(rr), np.nanmean([r["disp"] for r in rr]), np.mean([r["speed"] for r in rr]), np.mean([r["vel_err"] for r in rr]), sum(r["rotor"] for r in rr), len(rr)))
        order = sorted(P, key=lambda k: mean(rr, k, "b_abs"))
        for k in order[:12] + [x for x in ("model", "hold") if x not in order[:12]] + ORC:
            print("  %-16s b_abs %.3f  b_anc %.3f  b_end %.3f  ate %.4f  wp %.4f" % (k, mean(rr, k, "b_abs"), mean(rr, k, "b_anc"), mean(rr, k, "b_end"), mean(rr, k, "ate"), mean(rr, k, "wp")))
print("\n=== regret across the 8 cells (metric / best-in-cell - 1), mean and worst; plus pooled means")
for x in ("b_abs", "b_anc"):
    rows = []
    for k in P:
        reg = [mean(cell[c], k, x) / min(mean(cell[c], j, x) for j in P) - 1 for c in cell]; full = [mean(cell[(g, "full")], k, x) / min(mean(cell[(g, "full")], j, x) for j in P) - 1 for g in GR]
        rows.append((np.mean(reg), np.max(reg), np.mean(full), k))
    rows.sort()
    print("  [%s]  candidate          mean-regret  worst-regret  full-only-regret" % x)
    for a, b, c_, k in rows[:14] + [r for r in rows if r[3] in ("model", "hold") and r not in rows[:14]]: print("   %-20s %6.1f %%   %6.1f %%    %6.1f %%" % (k, 100 * a, 100 * b, 100 * c_))
# paired bootstrap over sequences (both orders, all seeds and failure points pooled per sequence)
rng = np.random.default_rng(0)
def boot(sel, a, b, x="b_abs", cfgs=CF, B=4000):
    rr = [r for r in res if sel(r) and r["cfg"] in cfgs]; names = sorted({r["name"] for r in rr})
    d = np.array([np.mean([r["m"][a][x] - r["m"][b][x] for r in rr if r["name"] == n]) for n in names]); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), int((d < 0).sum()), len(d)
print("\n=== paired bootstrap over sequences: mean difference (candidate - reference) in metres, 95 % CI, wins")
for a in ("ar10+hold", "ar5+hold", "model_x0.5", "model_x0.75", "cv_0.5", "model+ar10", "ar10", "hold", "ar10_damp2", "model+cv_0.5"):
    for ref in ("model", "hold"):
        if a == ref: continue
        for g in GR:
            for cf_name, cfs in (("full", ("full",)), ("all", CF)):
                for x in ("b_abs", "b_anc"):
                    mu, lo, hi, w, n = boot(GR[g], a, ref, x, cfs); print("  %-13s vs %-6s %-5s %-4s %-5s %+.3f [%+.3f, %+.3f] wins %d/%d" % (a, ref, g, cf_name, x, mu, lo, hi, w, n))
print("\n=== orientation candidates (position = model): ori_b [deg] per cell; ori (whole trajectory)")
for k in O: print("  %-22s " % k[4:] + "  ".join("%s/%s %.1f" % (g[:1], c, mean(cell[(g, c)], k, "ori_b")) for g in GR for c in CF) + "   pooled ori %.2f" % np.mean([mean(cell[c], k, "ori") for c in cell]))
# stacking / selection under grouped CV on the anchored error at 0.1 s marks (pre-failure alignment frame)
S = [r for r in res if r["stack"] is not None]; K = list(S[0]["stack"]["X"].keys())
def arr(r): st = r["stack"]; return np.stack([np.asarray(st["X"][k]) for k in K], 0), np.asarray(st["tgt"]), np.asarray(st["tau"])
def anc_err(Xk, tgt): return float(np.linalg.norm(Xk - tgt, axis=1).mean())
def fit_w(rr, ridge=1e-3, tbin=None):
    A, y = [], []
    for r in rr:
        X, tg, tau = arr(r)
        for j in range(3): A.append(X[:, :, j].T); y.append(tg[:, j])
    A = np.concatenate(A); y = np.concatenate(y); w = np.linalg.solve(A.T @ A + ridge * len(y) * np.eye(len(K)), A.T @ y); return w
def evaluate(train, test, w=None, pick=None):
    out = []
    for r in test:
        X, tg, tau = arr(r); pred = np.tensordot(w, X, 1) if w is not None else X[K.index(pick)]; out.append(anc_err(pred, tg))
    return out
print("\n=== forecast combination / selection under grouped CV (anchored error, metres; candidates %s)" % K)
for plat, groups, key in (("rotor (EuRoC)", ("MH", "V1", "V2"), lambda r: r["name"][:2]), ("hand-held (TUM)", sorted({r["name"] for r in S if r["name"].startswith("room")}), lambda r: r["name"])):
    rr = [r for r in S if (key(r) in groups)]; tot = {"stack": [], "select": []}; base = {k: [] for k in K}; ws = []
    for gv in groups:
        tr = [r for r in rr if key(r) != gv]; te = [r for r in rr if key(r) == gv]
        w = fit_w(tr); ws.append(w); tot["stack"] += evaluate(tr, te, w=w)
        best = min(K, key=lambda k: np.mean(evaluate(None, tr, pick=k))); tot["select"] += evaluate(None, te, pick=best)
        for k in K: base[k] += evaluate(None, te, pick=k)
        print("  %-16s fold %-6s select=%-10s stack weights %s" % (plat, gv, best, " ".join("%s:%+.2f" % (k, v) for k, v in zip(K, w))))
    print("  %-16s held-out anchored error: stack %.3f  select %.3f  |  " % (plat, np.mean(tot["stack"]), np.mean(tot["select"])) + "  ".join("%s %.3f" % (k, np.mean(v)) for k, v in base.items()))
