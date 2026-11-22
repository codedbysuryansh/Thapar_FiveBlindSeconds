"""an04: does the rotor smoother's own innovation likelihood pick the right drag-bias prior per flight? For every DEV EuRoC
mission: log-likelihood of the forward filter under six drag-bias priors (the six equal existing candidates), then the
per-mission errors of the candidate the likelihood picks, against the champion and against each fixed setting."""
import os, sys, json, numpy as np
from multiprocessing import get_context
sys.path.insert(0, "~/SamsungPRISM_FiveBlindSeconds/opt/harness")
import fbsopt as H
SET = {"A": ("CHAMP-1.0.0", (0.15, 100.0, 0.15)), "B": ("P15-sc0.04-tauc300", (0.04, 300.0, 0.04)), "C": ("P16-sc0_0.15-sc0.05-tauc300", (0.05, 300.0, 0.15)),
       "D": ("P16-sc0_0.15-sc0.03-tauc1000", (0.03, 1000.0, 0.15)), "E": ("P15-sc0.1-tauc300", (0.1, 300.0, 0.1)), "F": ("P15-sc0.06-tauc1000", (0.06, 1000.0, 0.06))}
DEVSRC = "~/SamsungPRISM_FiveBlindSeconds/opt/src/main_dev.py"
def job(tag):
    import importlib.util
    spec = importlib.util.spec_from_file_location("md", DEVSRC); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    n = next(s for s in H.SEQS if tag.startswith(s + "_")); M.TRACE = {}
    D, sol, info = M.estimate(M.Source(os.path.join(H.CACHE, "missions", tag + ".zip")), H.k_loso(n))
    Mz = M.TRACE["Mz"]
    if not info["rotor"]: return tag, None
    return tag, {k: M.smooth_rotor(Mz, drag=v[1], ll_only=True) for k, v in SET.items()}
if __name__ == "__main__":
    tags = [t for n, sp, t in H.missions("DEV") if not n.startswith("room")]
    with get_context("spawn").Pool(11) as pool: LL = dict(pool.map(job, tags))
    res = {k: H.load_results(v[0], "DEV") for k, v in SET.items()}
    rows = []
    for t in tags:
        ll = LL[t]; best = max(ll, key=ll.get); rows.append((t, best, {k: ll[k] - ll["A"] for k in ll}))
    pick = {t: b for t, b, _ in rows}
    def agg(choose):
        """mean ate/wp/b_abs/drop over missions with the setting chosen per mission."""
        return {m: np.mean([res[choose(t)][t]["score"][m] for t in tags]) for m in ("ate", "wp", "b_abs", "drop")}
    base = agg(lambda t: "A")
    def rel(d): return "  ".join("%s %+.2f%%" % (m, 100 * (d[m] - base[m]) / base[m]) for m in d) + "  composite %+.2f%%" % ((30 * (d["wp"] / base["wp"] - 1) + 20 * (d["b_abs"] / base["b_abs"] - 1) + 10 * (d["drop"] / base["drop"] - 1)) / 60 * 100)
    for k in SET: print("fixed %s %-30s %s" % (k, SET[k][0], rel(agg(lambda t, k=k: k))))
    print("ML pick per mission                  ", rel(agg(lambda t: pick[t])))
    seqs = sorted({t.rsplit("_", 1)[0] for t in tags})
    for s in H.EUROC:
        ts = [t for t in tags if t.startswith(s + "_")]
        print("  %-16s picks %s   mean dLL vs A: %s" % (s, "".join(pick[t] for t in ts), "  ".join("%s %+.0f" % (k, np.mean([dict(r[2])[k] for r in rows if r[0] in ts])) for k in SET)))
    json.dump({t: LL[t] for t in tags}, open("~/SamsungPRISM_FiveBlindSeconds/opt/notes/an04_drag_ml.json", "w"), indent=1)
