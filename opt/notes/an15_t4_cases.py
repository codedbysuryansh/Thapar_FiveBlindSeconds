"""AN15: inspect a T4 robustness run (p5_t4_robustness.py JSON): every case whose decisions differ from the champion
(input-side decisions, rotor/guard), with the champion and candidate scores, and the worst ATE regressions.
Usage: an15_t4_cases.py T4.json OUT.txt"""
import sys, json
import numpy as np
d = json.load(open(sys.argv[1])); res = d["cases"]; L = []
SAME = ("rotor_gate", "mode", "n_tail", "L", "implausible")
def sc(r, lab): s = r.get(lab + "_score", {}); return "ate %.3f wp %.3f b_abs %.3f drop %.3f" % tuple(s.get(k, np.nan) for k in ("ate", "wp", "b_abs", "drop"))
for title, sel in (("input-side decision differs", lambda r: any(r["cand"][k] != r["champ"][k] for k in SAME)),
                   ("rotor/guard differs", lambda r: (r["cand"]["rotor"], r["cand"]["guard"]) != (r["champ"]["rotor"], r["champ"]["guard"]))):
    rr = [r for r in res if "cand" in r and "champ" in r and sel(r)]; L.append("## %s: %d cases" % (title, len(rr)))
    for r in rr:
        diff = {k: (r["champ"][k], r["cand"][k]) for k in SAME + ("rotor", "guard") if r["champ"][k] != r["cand"][k]}
        a0, a1 = r.get("champ_score", {}).get("ate", np.nan), r.get("cand_score", {}).get("ate", np.nan)
        L.append("  %-58s %-40s champ %s | cand %s | ATE %+6.1f%%" % (str(r["case"])[:58], str(diff)[:40], sc(r, "champ"), sc(r, "cand"), 100 * (a1 / a0 - 1)))
    if rr:
        a0 = np.array([r["champ_score"]["ate"] for r in rr]); a1 = np.array([r["cand_score"]["ate"] for r in rr])
        L.append("  mean ATE %.3f -> %.3f (%+.1f %%), wins %d losses %d" % (a0.mean(), a1.mean(), 100 * (a1.mean() / a0.mean() - 1), int(np.sum(a1 < a0)), int(np.sum(a1 > a0))))
both = [r for r in res if "cand_score" in r and "champ_score" in r]
rel = sorted(both, key=lambda r: -(r["cand_score"]["ate"] / r["champ_score"]["ate"]))
L.append("## 15 worst ATE changes (of %d scored cases)" % len(both))
for r in rel[:15]: L.append("  %-58s champ %s | cand %s | ATE %+6.1f%%" % (str(r["case"])[:58], sc(r, "champ"), sc(r, "cand"), 100 * (r["cand_score"]["ate"] / r["champ_score"]["ate"] - 1)))
for fam in ("E46", "E55", "E60"):
    rr = [r for r in both if r["case"][0] == fam]
    for k in ("ate", "wp", "b_abs", "drop"):
        a0 = np.array([r["champ_score"][k] for r in rr]); a1 = np.array([r["cand_score"][k] for r in rr])
        L.append("%s %-5s mean %.4f -> %.4f (%+.2f %%)  worse > 25 %%: %d / %d" % (fam, k, np.nanmean(a0), np.nanmean(a1), 100 * (np.nanmean(a1) / np.nanmean(a0) - 1), int(np.sum(a1 > 1.25 * a0)), len(rr)))
open(sys.argv[2], "w").write("\n".join(L) + "\n"); print("\n".join(L))
