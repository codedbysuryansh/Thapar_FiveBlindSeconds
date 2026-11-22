#!/usr/bin/env python3
"""E12 - dropout and gyro-bias study with the v3 pipeline (whole sequences, 17 real streams, OUR simulator).
(a) four dropout encodings must give the same answer; (b) how much of the dropout loss is heading: oracle rotation inside
gaps (DIAGNOSTIC ONLY) vs linear-rate vs zero-rotation bridging; (c) shrinking post-gap horizontal motion; (d) gyro bias
ramp / random walk / abrupt step with constant, two-sided-anchor and linear-drift bias models; (e) gap position;
(f) attitude time constant. Ground truth only scores. Run: .venv/bin/python research/experiments/e12_gaps_bias.py"""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import fbs3 as f
def run(t, w, a, valid, **kw):
    bridge = kw.pop("bridge", "lin"); oracle = kw.pop("oracle", None); q, bg, quiet, af = f.front3(t, w, a, valid, bridge=bridge, true_gap_rot=oracle, **kw)
    return f.lin_smooth(t, q, af, valid, f.model_spring(*f.SPRING), bias_jump=0.5)[0], q
def encode(tn, w, a, valid, kind):
    w, a = w.copy(), a.copy(); bad = ~valid
    if kind == "zero": w[bad] = 0.0; a[bad] = 0.0
    if kind == "held":
        last = np.maximum.accumulate(np.where(valid, np.arange(len(valid)), 0)); w = w[last]; a = a[last]
    if kind == "removed": return tn[valid], w[valid], a[valid], np.where(valid)[0]
    return tn, w, a, np.arange(len(tn))
def work(n):
    s = f.load(n); ok = s.ok; ml = f.ate(np.zeros((ok.sum(), 3)), s.gt_p[ok]); r = {"name": n, "ml": ml}; A = lambda p, m=ok: f.ate(p[m], s.gt_p[m])
    for gname, gaps in [("3x2s", [2.0] * 3), ("3x0.5s", [0.5] * 3), ("6x2s", [2.0] * 6)]:
        for seed in (0, 1):
            w, a, valid, meta = f.degrade(s.t, s.w, s.a, dict(gaps=gaps), seed=1000 * seed + sum(n.encode()) % 997); k = "%s_s%d" % (gname, seed)
            if gname == "3x2s" and seed == 0:
                base = None
                for kind in ("nan", "zero", "held", "removed"):
                    tn2, w2, a2, keep = encode(s.tn, w, a, valid, kind); e = f.estimate3(tn2, w2, a2); pfull = np.zeros((len(s.t), 3)); pfull[keep] = e["p"]; m = ok & np.isin(np.arange(len(ok)), keep) & valid
                    v = A(pfull, m); r["enc_" + kind] = v; r["enc_rows_" + kind] = [len(tn2), len(e["p"])]
            t, wu, au, vu, rows = f.ingest(s.tn, w, a)
            for tag, kw in [("lin", {}), ("zero", dict(bridge="zero")), ("oracle", dict(oracle=s.w)), ("tau1.5", dict(tau=1.5))]:
                p, q = run(t, wu, au, vu, **kw); r["%s_%s" % (k, tag)] = A(p)
                if tag == "lin":
                    seg = np.cumsum(np.concatenate([[0], np.diff(vu.astype(np.int8)) != 0])); ids, cnt = np.unique(seg[vu], return_counts=True); ref = ids[np.argmax(cnt)]
                    for sh in (0.5, 0.0):
                        p2 = p.copy(); p2[seg != ref, :2] *= sh; r["%s_shrink%g" % (k, sh)] = A(p2)
    rng = np.random.default_rng(sum(n.encode())); d = rng.normal(size=3); d /= np.linalg.norm(d)
    for bname, mk in [("ramp0.02", lambda: f.degrade(s.t, s.w, s.a, dict(gb=0.02), seed=3)[0]), ("ramp0.05", lambda: f.degrade(s.t, s.w, s.a, dict(gb=0.05), seed=3)[0]), ("rw0.02", lambda: f.degrade(s.t, s.w, s.a, dict(gb=0.02, bias_shape="rw"), seed=3)[0]),
                      ("step0.02", lambda: s.w + 0.02 * d * (s.t > 0.5 * s.t[-1])[:, None]), ("abias_ramp0.2", None)]:
        if mk is None: w = s.w; a = f.degrade(s.t, s.w, s.a, dict(ab=0.2), seed=3)[1]
        else: w = mk(); a = s.a
        v = np.ones(len(s.t), bool)
        for tag, kw in [("default", {}), ("one_anchor", dict(two_sided=False)), ("drift", dict(drift=True)), ("tau1.5", dict(tau=1.5))]:
            p, q = run(s.t, w, a, v, **kw); r["%s_%s" % (bname, tag)] = A(p)
    for pos in (0.25, 0.5, 0.75, 0.97):
        v = np.ones(len(s.t), bool); g1 = pos * s.t[-1]; v[(s.t >= g1 - 2.0) & (s.t < g1)] = False; w = s.w.copy(); a = s.a.copy(); w[~v] = np.nan; a[~v] = np.nan
        p, q = run(s.t, w, a, v); r["gap_at_%g" % pos] = A(p); last = ok & (s.t > s.t[ok][-1] - 5.0); Rm, tt = f.umeyama(p[ok], s.gt_p[ok]); r["gap_at_%g_last5" % pos] = float(np.linalg.norm(p[last] @ Rm.T + tt - s.gt_p[last], axis=1).mean())
    p, q = run(s.t, s.w, s.a, np.ones(len(s.t), bool)); r["clean"] = A(p); last = ok & (s.t > s.t[ok][-1] - 5.0); Rm, tt = f.umeyama(p[ok], s.gt_p[ok]); r["clean_last5"] = float(np.linalg.norm(p[last] @ Rm.T + tt - s.gt_p[last], axis=1).mean())
    return r
if __name__ == "__main__":
    names = f.available()
    with Pool(10) as p: res = p.map(work, names, chunksize=1)
    json.dump(res, open(os.path.join(f.fbs.ROOT, "research/results/e12_gaps_bias.json"), "w"), indent=1)
    ml = np.array([r["ml"] for r in res]); hh = np.array([r["name"].startswith("room") for r in res])
    def sk(keys):
        v = np.mean([[r[k] for r in res] for k in keys], 0); return "%5.1f /%5.1f /%5.1f (agg %5.1f)" % (100 * np.mean(1 - v / ml), 100 * np.mean((1 - v / ml)[~hh]), 100 * np.mean((1 - v / ml)[hh]), 100 * (1 - v.mean() / ml.mean()))
    print("skill %% = mean per-sequence (ALL / EuRoC / TUM) and aggregate ATE reduction")
    print("clean                     ", sk(["clean"]))
    print("encodings (3x2s, seed 0): " + " | ".join("%s %s" % (k, sk(["enc_" + k])) for k in ("nan", "zero", "held", "removed")))
    print("  max |ATE difference| across encodings: %.4f m; rows in/out for 'removed': %s" % (max(max(abs(r["enc_" + k] - r["enc_nan"]) for k in ("zero", "held", "removed")) for r in res), res[0]["enc_rows_removed"]))
    for g in ("3x0.5s", "3x2s", "6x2s"):
        print("gaps %-7s " % g + " | ".join("%s %s" % (tag, sk(["%s_s%d_%s" % (g, sd, tag) for sd in (0, 1)])) for tag in ("lin", "zero", "oracle", "tau1.5", "shrink0.5", "shrink0")))
    for bname in ("ramp0.02", "ramp0.05", "rw0.02", "step0.02", "abias_ramp0.2"):
        print("gyro/acc bias %-13s " % bname + " | ".join("%s %s" % (tag, sk(["%s_%s" % (bname, tag)])) for tag in ("default", "one_anchor", "drift", "tau1.5")))
    print("single 2 s gap ending at fraction of stream: " + " | ".join("%g: %s last-5s err %.2f" % (pos, sk(["gap_at_%g" % pos]), np.mean([r["gap_at_%g_last5" % pos] for r in res])) for pos in (0.25, 0.5, 0.75, 0.97)) + " | no gap last-5s err %.2f" % np.mean([r["clean_last5"] for r in res]))
