"""AN10: paired comparison of two official-protocol (T3) runs: seeds 42 + 5-13, orders A/B, 17 sequences = 340 missions,
k_drag leave-one-sequence-out. Per metric and dataset: means, relative change, a 95 % bootstrap CI that resamples whole
SEQUENCES (all seeds and orders of a sequence move together; 10 000 replicates, fixed seed), and the number of sequences whose
mean improved. Usage: an10_t3_paired.py REF.json CAND.json OUT.json [label_ref label_cand]"""
import sys, json
import numpy as np

def load(p):
    d = json.load(open(p)); d = d["missions"] if isinstance(d, dict) else d
    return {(m["name"], m["order"], int(m["seed"])): m["score"] for m in d}

def main():
    ref, cand, out = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3]
    lr, lc = (sys.argv[4], sys.argv[5]) if len(sys.argv) > 5 else ("ref", "cand")
    keys = sorted(set(ref) & set(cand)); assert len(keys) == len(ref) == len(cand), (len(keys), len(ref), len(cand))
    metrics = ["ate", "wp", "b_abs", "b_end", "b_anc", "drop", "rpe_t", "rpe_r", "ori", "ori_b"]
    rng = np.random.default_rng(20261122); res = {"ref": lr, "cand": lc, "missions": len(keys), "groups": {}}
    for grp in ("EuRoC", "TUM-VI"):
        ks = [k for k in keys if (k[0].startswith("room") == (grp == "TUM-VI"))]
        seqs = sorted(set(k[0] for k in ks)); G = {}
        for mt in metrics:
            if not all(mt in ref[k] and mt in cand[k] for k in ks): continue
            r = np.array([[np.mean([ref[k][mt] for k in ks if k[0] == s])] for s in seqs])[:, 0]
            c = np.array([[np.mean([cand[k][mt] for k in ks if k[0] == s])] for s in seqs])[:, 0]
            rm, cm = float(np.mean([ref[k][mt] for k in ks])), float(np.mean([cand[k][mt] for k in ks]))
            idx = rng.integers(0, len(seqs), (10000, len(seqs))); rel = (c[idx].mean(1) / r[idx].mean(1) - 1) * 100
            G[mt] = dict(ref=rm, cand=cm, rel=(cm / rm - 1) * 100, ci=[float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))],
                         seq_better=int(np.sum(c < r)), seqs=len(seqs))
        res["groups"][grp] = dict(n=len(ks), metrics=G)
    json.dump(res, open(out, "w"), indent=1)
    print("%s -> %s on %d missions (relative change of the mean; 95 %% CI over sequences; sequences improved)" % (lr, lc, len(keys)))
    for grp, g in res["groups"].items():
        for mt, v in g["metrics"].items():
            print("  %-6s n=%-3d %-6s %9.4f -> %9.4f  %+7.2f%%  [%+7.2f, %+7.2f]  %2d/%d" % (grp, g["n"], mt, v["ref"], v["cand"], v["rel"], v["ci"][0], v["ci"][1], v["seq_better"], v["seqs"]))

if __name__ == "__main__":
    main()
