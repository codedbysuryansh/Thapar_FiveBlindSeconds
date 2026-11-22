#!/usr/bin/env python3
"""E40 - audit of the organisers' degrade_imu.py on all 17 public sequences, both input column orders, the script's own seed.
(1) the diagnostic copy reproduces the original output exactly; (2) what the script does in OUR convention (gyro first):
    detected units, added noise / bias at the end, counts of spikes, stuck rows, drops, duplicates, failure rows;
(3) clean-data facts that decide the ingest rules (exact repeats, repeated timestamps);
(4) spike detectors scored against the true spike rows (rows from the copy's masks), on degraded and on clean data."""
import sys, os, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import p4, fbs4 as f
RULES = [("legacy", None, None)] + [("z%g_k%d" % (z, k), z, k) for z in (3, 4, 5, 6, 8) for k in (1, 2, 3, 4, 5, 6)]
def detect(tsec, X, rules=RULES):
    N = len(X); pad = np.pad(X, ((2, 2), (0, 0)), mode="edge"); med = np.median(np.stack([pad[i:i + N] for i in range(5)]), 0); r = np.abs(X - med)
    fs = 1.0 / np.median(np.diff(tsec)); n1 = max(5, int(fs)); m = max(1, N // n1); sig = np.median(r[:m * n1].reshape(m, n1, 6), 1) * 1.4826; sig = np.repeat(sig, n1, 0); sig = np.vstack([sig, np.repeat(sig[-1:], N - len(sig), 0)]); sig = np.maximum(sig, 1e-6)
    z = r / sig; out = {}
    for name, zt, k in rules:
        if name == "legacy": out[name] = (r > np.maximum(np.array([1.5] * 3 + [15.0] * 3), 10.0 * sig)).any(1)
        else: out[name] = (z > zt).sum(1) >= k
    return out
def one(args):
    n, order = args; m = p4.official(n, order, 42, masks=True); mk = m.masks; s = m.s
    out2, _ = p4.degrade_masks(m.df, m.cfg, 42, orig=p4.DEG); same = bool(out2.equals(m.out))
    X = np.hstack([s.w, s.a]); rep = int(((X[1:] == X[:-1]).all(1)).sum()); dts = int((np.diff(s.tn) == 0).sum())
    first = np.concatenate([[True], np.diff(m.tn) > 0]); tn, w, a, idx = m.tn[first], m.w[first], m.a[first], m.idx[first]; tsec = (tn - tn[0]) * 1e-9
    fail = mk["failed"][idx]; sp = mk["spike"][idx]; stk = mk["stuck"][idx]; Tp = m.T - 5.0
    late = (tsec > 0.9 * Tp) & (tsec < Tp) & ~sp & ~stk & ~fail; dw = w[late] - s.w[idx[late]]; da = a[late] - s.a[idx[late]]
    Xo = np.hstack([w, a]); act = ~fail; det = detect(tsec[act], Xo[act]); detc = detect((s.tn - s.tn[0]) * 1e-9, X); spa = sp[act]
    rules = {k: dict(tp=int((v & spa).sum()), fp=int((v & ~spa).sum()), fp_clean=int(detc[k].sum())) for k, v in det.items()}
    st = m.stats
    return dict(name=n, order=order, identical=same, rows_in=st["rows_in"], rows_out=st["rows_out"], T=float(m.T), acc_unit=mk["acc_unit"], acc_scale=float(mk["acc_scale"]), ts_unit=mk["ts_unit"],
                spikes=st["spikes"], stuck=st["stuck_samples"], drop_single=st["dropped_single"], bursts=st["dropped_burst_gaps"], dropped=st["dropped_total"], dups=st["duplicated_rows"], failure_rows=st["failure_frames"],
                n_spike_rows_out=int(sp.sum()), clean_exact_repeats=rep, clean_dup_ts=dts, median_gyro_norm=float(np.median(np.linalg.norm(s.w, axis=1))),
                late_gyro_noise=float(np.std(dw - dw.mean(0), 0).mean()), late_gyro_offset=float(np.linalg.norm(dw.mean(0))), late_acc_noise=float(np.std(da - da.mean(0), 0).mean()), late_acc_offset=float(np.linalg.norm(da.mean(0))),
                end_gyro_bias_true=float(np.linalg.norm((mk["acc_bias"] if order == "B" else mk["gyr_bias"])[-1])), end_acc_bias_true=float(np.linalg.norm((mk["gyr_bias"] if order == "B" else mk["acc_bias"])[-1])),
                max_gap=int(m.gap_after.max()), n_gaps5=int((m.gap_after >= 5).sum()), rules=rules)
if __name__ == "__main__":
    names = f.available()
    with Pool(10) as p: res = p.map(one, [(n, o) for o in ("A", "B") for n in names], chunksize=1)
    json.dump(res, open(os.path.join(p4.ROOT, "research/results/e40_official_audit.json"), "w"))
    print("copy reproduces the original output on all %d runs: %s" % (len(res), all(r["identical"] for r in res)))
    print("clean data: exact 6-channel repeats per sequence %s ; repeated timestamps %s" % ([r["clean_exact_repeats"] for r in res if r["order"] == "A"], [r["clean_dup_ts"] for r in res if r["order"] == "A"]))
    for o in ("A", "B"):
        print("\n=== order %s (%s)" % (o, "timestamp, accel, gyro  (docstring)" if o == "A" else "EuRoC file as is: timestamp, gyro, accel"))
        print("%-15s %6s %6s | unit-detect acc_scale | spikes stuck drops(single/bursts) dups fail | late gyro noise/offset [rad/s] | late acc noise/offset [m/s2] | end bias gyro/acc | max gap" % ("seq", "T", "rows"))
        for r in [x for x in res if x["order"] == o]:
            print("%-15s %6.1f %6d | %-6s %.3f | %4d %4d %5d/%3d %3d %4d | %.4f / %.4f | %.4f / %.4f | %.4f / %.4f | %d" % (r["name"], r["T"], r["rows_out"], r["acc_unit"], r["acc_scale"], r["spikes"], r["stuck"], r["drop_single"], r["bursts"], r["dups"], r["failure_rows"],
                  r["late_gyro_noise"], r["late_gyro_offset"], r["late_acc_noise"], r["late_acc_offset"], r["end_gyro_bias_true"], r["end_acc_bias_true"], r["max_gap"]))
        print("\n  spike detectors (rows): rule | TPR EuRoC / TUM | false positives per 10k degraded rows EuRoC / TUM | false positives on CLEAN data (total over sequences) EuRoC / TUM")
        for rule, _, _ in RULES:
            line = "  %-9s |" % rule
            for sel in (lambda n: not n.startswith("room"), lambda n: n.startswith("room")):
                rr = [x for x in res if x["order"] == o and sel(x["name"])]; tp = sum(x["rules"][rule]["tp"] for x in rr); ns = sum(x["n_spike_rows_out"] for x in rr); fp = sum(x["rules"][rule]["fp"] for x in rr); nr = sum(x["rows_out"] for x in rr)
                line += " %.3f" % (tp / max(ns, 1)) + " %7.2f" % (1e4 * fp / nr) + " %5d |" % sum(x["rules"][rule]["fp_clean"] for x in rr)
            print(line)
