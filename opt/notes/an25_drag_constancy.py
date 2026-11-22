"""AN25: when does the steady drag prior help? For rotor missions: run main_dev K (a) with the loose prior only (switch off,
rest + no drag rows at rest) recording the smoother states, and (b) with the steady prior forced; benefit = ATE_b / ATE_a - 1.
Candidate gate statistics from the loose run, flight blocks (with drag rows) only: the 10 s moving average of the drag-row
offsets c1, c2 (states 12, 13): S_std = RMS deviation from its mean, S_rng = largest range of the two; plus rr0, rh.
Cases: harness missions (DEV/DEV2/CONF tags) and T4-style cases (E46 condition, sequence, seed). Usage:
  an25_drag_constancy.py OUT.json --tags TAG,... | --set DEV --seqs MH_01_easy,... | --e46 SEED --seqs ... [--conds c1,c2]"""
import os, sys, json, argparse, tempfile, importlib.util, numpy as np, pandas as pd
from multiprocessing import get_context
ROOT = "~/SamsungPRISM_FiveBlindSeconds"
for p in ("research/phase4", "research/harness", "research/experiments", "opt/harness"): sys.path.insert(0, os.path.join(ROOT, p))
SNAP = os.path.join(ROOT, "opt/src/snapshots/main_dev_K.py")
B = dict(rest_frac=0.5, rest_gate=0.06, drag_off_rest=1, drag_tight=[0.03, 1000.0, 0.15], tau_g_final_hand=24.0, final_mu_max=0.002, final_sw_max=0.04, guard_ref="loose")
def mod(cfg):
    s = importlib.util.spec_from_file_location("mdk", SNAP); M = importlib.util.module_from_spec(s); s.loader.exec_module(M); M.CFG.update(cfg); return M
def stats(T):
    X = T["rotor_states"]; cd = np.asarray(T["Mz"].cd); fl = cd > 0
    if fl.sum() < 300: return dict(S_std=float("nan"), S_rng=float("nan"), flight_s=float(fl.sum()) / 25.0)
    c = X[fl][:, 12:14]; w = 250; k = np.ones(w) / w
    ma = np.stack([np.convolve(c[:, j], k, mode="valid") for j in range(2)], 1)
    return dict(S_std=float(np.sqrt(np.mean(np.sum((ma - ma.mean(0)) ** 2, 1)))), S_rng=float((ma.max(0) - ma.min(0)).max()), flight_s=float(fl.sum()) / 25.0,
                c_mean=[float(x) for x in c.mean(0)])
def one(job):
    import fbsopt as H
    kind, spec = job; out = dict(kind=kind, spec=spec)
    if kind == "tag":
        tag = spec; n = next(s for s in H.SEQS if tag.startswith(s + "_")); path = os.path.join(H.CACHE, "missions", tag + ".zip"); tno = np.load(os.path.join(H.CACHE, "missions", tag + ".npz"))["tno"]; tmp = None
        order = np.arange(len(tno)); tns = tno
    else:
        import p4_t4_robustness as T4, p4
        case = tuple(spec); n = case[2] if case[0] == "E46" else case[1]; tn, w, a = T4.case_input(case); w = np.round(w, 6); a = np.round(a, 6)
        fd, tmp = tempfile.mkstemp(suffix=".csv", prefix="fbs_an25_"); os.close(fd); path = tmp
        pd.DataFrame(dict(zip(p4.EUROC_COLS, [tn, w[:, 0], w[:, 1], w[:, 2], a[:, 0], a[:, 1], a[:, 2]]))).to_csv(path, index=False)
        order = np.argsort(tn, kind="stable"); tns = tn[order]
    out["name"] = n
    try:
        m = H.EV.mission(tns, H.gt_cached(n))
        for lab, cfg in (("loose", dict(B, drag_switch_r=None)), ("steady", dict(B, drag_switch_r=0.0))):
            M = mod(cfg); M.TRACE = {} if lab == "loose" else None; fd2, est = tempfile.mkstemp(suffix=".txt"); os.close(fd2)
            _p, info = M.run(path, est, k_drag=H.k_loso(n)); ens, p, q = H.EV.read_estimate(est); os.remove(est); sc = H.EV.score(m, p[order], q[order])
            out[lab] = dict(ate=sc["ate"], wp=sc["wp"], b_abs=sc["b_abs"], drop=sc["drop"], rotor=info.get("rotor"), guard=info.get("guard"), rh=info.get("rh"), rr0=info.get("rr0"), rr=info.get("rr"))
            if lab == "loose" and info.get("rotor_gate"): out.update(stats(M.TRACE))
        out["benefit"] = out["steady"]["ate"] / out["loose"]["ate"] - 1
    except Exception as ex: out["error"] = repr(ex)[:200]                                          # noqa: BLE001
    finally:
        if tmp: os.remove(tmp)
    return out
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--tags", default=""); ap.add_argument("--set", default=""); ap.add_argument("--seqs", default="")
    ap.add_argument("--e46", type=int, default=None); ap.add_argument("--conds", default=""); a = ap.parse_args(); jobs = []
    import fbsopt as H
    seqs = a.seqs.split(",") if a.seqs else [s for s in H.SEQS if not s.startswith("room")]
    if a.tags: jobs += [("tag", t) for t in a.tags.split(",")]
    if a.set: jobs += [("tag", tag) for n, sp, tag in H.missions(a.set, seqs)]
    if a.e46 is not None:
        from e46_stress_v5c import CONDS
        conds = a.conds.split(",") if a.conds else CONDS; jobs += [("case", ["E46", c, n, a.e46]) for c in conds for n in seqs]
    with get_context("spawn").Pool(11) as pool: res = pool.map(one, jobs, chunksize=1)
    json.dump(res, open(a.out, "w"), indent=1, default=float); ok = [r for r in res if "benefit" in r and r.get("S_std") == r.get("S_std") and "S_std" in r]
    print("%d jobs, %d with statistics, %d errors" % (len(res), len(ok), sum("error" in r for r in res)))
    for r in sorted(ok, key=lambda r: r["S_std"]):
        print("  %-52s S_std %.3f S_rng %.3f flight %5.0f s rr0 %.2f rh %.2f | ATE loose %.3f steady %.3f benefit %+6.1f%%" % (str(r["spec"])[:52], r["S_std"], r["S_rng"], r["flight_s"], r["loose"]["rr0"] or 0, r["loose"]["rh"] or 0, r["loose"]["ate"], r["steady"]["ate"], 100 * r["benefit"]))
if __name__ == "__main__":
    main()
