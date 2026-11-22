#!/usr/bin/env python3
"""Tables from research/results/e14_final.json. Run: .venv/bin/python research/experiments/e14_report.py"""
import json, os, sys, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); res = json.load(open(os.path.join(ROOT, "research/results/e14_final.json")))
IX = dict(wp=0, b_abs=1, b_anc=2, gap=3, ate=4, rpe=5, b_end=6, anc_end=7, ate_yaw=8, ate_none=9, wp_max=10, hit=11, cal=12, cal_b=13)
def get(r, v, key):
    m = r["m"]; v2 = v if v in m else ("final_zupt" + ("|hold" if v.endswith("|hold") else "") if v.startswith(("zupt", "sens")) else v); return m[v2][IX[key]]
def mean(rr, v, key): return float(np.nanmean([get(r, v, key) for r in rr])) if rr else float("nan")
def comp(r, v, w, mode):
    ref = r["m"]["motionless"]; parts = []
    for key, wg in zip(("wp", "b_abs", "gap"), w):
        e = get(r, v, key); e0 = ref[IX[key]]
        if np.isnan(e): continue
        parts.append(((np.clip(1 - e / e0, 0, 1) if mode == "lin" else np.exp(-e / 1.0)), wg))
    return sum(a * b for a, b in parts) / sum(b for _, b in parts) * sum(w)
G = {"ALL": lambda r: True, "EuRoC": lambda r: not r["name"].startswith("room"), "TUM": lambda r: r["name"].startswith("room")}
V = ["motionless", "matern", "spring", "final", "final_zupt", "zupt_lever", "zupt_sd1", "zupt_lever_sd1", "zupt_red", "zupt_lever_red", "zupt_lever_red8"]
if __name__ == "__main__":
    levels = ["clean", "gaps", "L1", "L2", "ab03", "ab10", "gb05"]
    for g in ("EuRoC", "TUM", "ALL"):
        for lvl in levels:
            rr = [r for r in res if r["level"] == lvl and G[g](r)]; print("\n[%s | %s | %d missions]  wp  wpmax hit<1m | blind abs/end  anchored/end | in-gap | ATE se3 / yaw / none | RPE1s | calib pre/blind | acc.score lin A(60) B(70), exp A" % (g, lvl, len(rr)))
            for v in (V if g != "TUM" else V[:5]):
                print("  %-16s %5.2f %5.2f %4.2f | %5.2f %5.2f  %5.2f %5.2f | %5.2f | %5.2f %5.2f %6.2f | %5.3f | %4.2f %4.2f | %5.1f %5.1f %5.1f" % (v, mean(rr, v, "wp"), mean(rr, v, "wp_max"), mean(rr, v, "hit"), mean(rr, v, "b_abs"), mean(rr, v, "b_end"), mean(rr, v, "b_anc"), mean(rr, v, "anc_end"),
                      mean(rr, v, "gap"), mean(rr, v, "ate"), mean(rr, v, "ate_yaw"), mean(rr, v, "ate_none"), mean(rr, v, "rpe"), mean(rr, v, "cal"), mean(rr, v, "cal_b"),
                      np.mean([comp(r, v, (30, 20, 10), "lin") for r in rr]), np.mean([comp(r, v, (30, 20, 20), "lin") for r in rr]), np.mean([comp(r, v, (30, 20, 10), "exp") for r in rr])))
    print("\n== blind window: model prediction vs hold (anchored mean / anchored end / abs mean), by failure position; clean level")
    for g in ("EuRoC", "TUM"):
        for kf in range(3):
            rr = [r for r in res if r["level"] == "clean" and G[g](r) and r["kf"] == kf]; rest = np.mean([r["ends_at_rest"] for r in rr])
            print("  %-5s kf=%d (ends at rest %.0f%%): " % (g, kf, 100 * rest) + " | ".join("%s %.2f/%.2f/%.2f" % (v, mean(rr, v, "b_anc"), mean(rr, v, "anc_end"), mean(rr, v, "b_abs")) for v in ("final|hold", "final", "final_zupt", "zupt_lever", "zupt_lever_sd1", "spring", "matern")))
    print("\n== sensitivities on EuRoC (wp / ATE / blind anchored), clean and L2")
    for lvl in ("clean", "L2", "ab10"):
        rr = [r for r in res if r["level"] == lvl and G["EuRoC"](r)]; print("  %-5s " % lvl + " | ".join("%s %.2f/%.2f/%.2f" % (v, mean(rr, v, "wp"), mean(rr, v, "ate"), mean(rr, v, "b_anc")) for v in ("zupt_lever", "sens_k_lo", "sens_k_hi", "sens_sc05")))
    print("\n== orientation (independent of the position model): tilt [deg], rotational RPE 1 s [deg], blind-window orientation error mean/end [deg]")
    for g in ("EuRoC", "TUM"):
        for lvl in ("clean", "gaps", "L1", "L2", "gb05"):
            rr = [r for r in res if r["level"] == lvl and G[g](r)]
            print("  %-5s %-5s tilt %.2f  rpe_rot %.2f | " % (g, lvl, np.mean([r["tilt"] for r in rr]), np.nanmean([r["rpe_rot"] for r in rr])) + " ".join("%s %.1f/%.1f" % (k, np.mean([r["ori"][k][0] for r in rr]), np.mean([r["ori"][k][1] for r in rr])) for k in rr[0]["ori"]))
    print("\n== gate: rotor decision by dataset and level (fraction rotor), lock range")
    for lvl in levels:
        print("  %-5s EuRoC rotor %.2f (lock max %.3f) | TUM rotor %.2f (lock min %.3f)" % (lvl, np.mean([r["rotor"] for r in res if r["level"] == lvl and G["EuRoC"](r)]), max(r["lock"] for r in res if r["level"] == lvl and G["EuRoC"](r)), np.mean([r["rotor"] for r in res if r["level"] == lvl and G["TUM"](r)]), min(r["lock"] for r in res if r["level"] == lvl and G["TUM"](r))))
