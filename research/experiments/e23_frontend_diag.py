#!/usr/bin/env python3
"""E23 - diagnostics for the Phase 3 front-end pieces (no position estimation):
 (a) repair on CLEAN data: how many samples does the despiker touch, how many does the gravity-magnitude test invalidate
 (b) coarse bias stage: its significance statistic (max |bias - reference|) on clean / L1 / L2 / L3 versus injected ramps,
     and the error of the coarse track against the injected ramp (full vector and the part perpendicular to gravity)."""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3, fbs5 as f
RATES = (0.0005, 0.002, 0.01, 0.05)
def one(n):
    out = dict(name=n); m = p3.mission(n, "clean", 1, 0); t, wu, au, valid, rows = f.ingest(m.tn, m.w, m.a); w2, a2, v2, info = f.repair(t, wu, au, valid); out["repair"] = info; out["N"] = len(t)
    an = np.linalg.norm(au, axis=1); fs = 1 / np.median(np.diff(t)); h = int(fs) // 2; med = np.array([np.median(an[max(0, i - h):i + h + 1]) for i in range(0, len(t), 20)]); out["med1s"] = [float(med.min()), float(med.max())]
    for lvl in ("clean", "L1", "L2", "L3"):
        mm = p3.mission(n, lvl, 1, 0); t, wu, au, valid, rows = f.ingest(mm.tn, mm.w, mm.a); bc, bf, bb = f.coarse_bias(t, wu, au, valid); out["cm_" + lvl] = float(np.abs(bc).max()); out["crms_" + lvl] = float(np.sqrt((bc ** 2).sum(1).mean()))
    for axis in (0, None):
        for r in RATES:
            rng = np.random.default_rng(5); d = np.eye(3)[axis] if axis is not None else (lambda x: x / np.linalg.norm(x))(rng.normal(size=3)); fault = lambda t_, w_, a_, rg: (w_ + r * t_[:, None] * d, a_)
            mm = p3.mission(n, "clean", 1, 0, fault=fault); t, wu, au, valid, rows = f.ingest(mm.tn, mm.w, mm.a); bc, bf, bb = f.coarse_bias(t, wu, au, valid); truth = r * t[:, None] * d; err = bc - truth
            u = f.lowpass(au, fs, 0.5); u /= np.linalg.norm(u, axis=1, keepdims=True); perp = err - (err * u).sum(1, keepdims=True) * u; tp = truth - (truth * u).sum(1, keepdims=True) * u
            out["ramp_%s_%g" % ("x" if axis == 0 else "r", r)] = [float(np.abs(bc).max()), float(np.sqrt((err ** 2).sum(1).mean())), float(np.sqrt((perp ** 2).sum(1).mean())), float(np.sqrt((truth ** 2).sum(1).mean())), float(np.sqrt((tp ** 2).sum(1).mean()))]
    return out
if __name__ == "__main__":
    names = f.available()
    with Pool(9) as p: res = p.map(one, names, chunksize=1)
    json.dump(res, open(os.path.join(p3.ROOT, "research/results/e23_frontend_diag.json"), "w"))
    print("seq             N     spikes implaus 1s-median|a| min/max | coarse max|b| [mrad/s]: clean L1 L2 L3 | ramp_r max|b|: " + " ".join("%g" % r for r in RATES) + " | ramp_x max|b|")
    for r in res: print("%-15s %6d %5d %6d  %5.2f/%5.2f | %5.1f %5.1f %5.1f %5.1f | %s | %s" % (r["name"], r["N"], r["repair"]["spikes"], r["repair"]["implausible"], r["med1s"][0], r["med1s"][1], 1e3 * r["cm_clean"], 1e3 * r["cm_L1"], 1e3 * r["cm_L2"], 1e3 * r["cm_L3"], " ".join("%6.0f" % (1e3 * r["ramp_r_%g" % x][0]) for x in RATES), " ".join("%6.0f" % (1e3 * r["ramp_x_%g" % x][0]) for x in RATES)))
    print("\ncoarse-track error vs injected ramp [mrad/s], mean over sequences: rate | truth rms | error rms (full) | truth rms perp-to-gravity | error rms perp-to-gravity")
    for ax in ("r", "x"):
        for x in RATES:
            v = np.array([r["ramp_%s_%g" % (ax, x)] for r in res]) * 1e3; eu = np.array([r["ramp_%s_%g" % (ax, x)] for r in res if not r["name"].startswith("room")]) * 1e3; tu = np.array([r["ramp_%s_%g" % (ax, x)] for r in res if r["name"].startswith("room")]) * 1e3
            print("  ramp_%s %-7g | %7.1f | %7.1f (EuRoC %6.1f, TUM %6.1f) | %7.1f | %7.1f (EuRoC %6.1f, TUM %6.1f)" % (ax, x, v[:, 3].mean(), v[:, 1].mean(), eu[:, 1].mean(), tu[:, 1].mean(), v[:, 4].mean(), v[:, 2].mean(), eu[:, 2].mean(), tu[:, 2].mean()))
