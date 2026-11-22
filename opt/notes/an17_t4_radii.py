"""AN17: radii and decisions (rh, rr0, rr, steady switch, guard) of selected T4 robustness cases, for named main_dev
configurations (snapshot J), to explain the T4 regressions of RC v1.1.0b (guard flips on MH_02, acc_ramp_1 blow-ups).
Usage: an17_t4_radii.py OUT.json"""
import os, sys, json, tempfile, importlib.util, numpy as np, pandas as pd
from multiprocessing import Pool
ROOT = "~/SamsungPRISM_FiveBlindSeconds"
for p in ("research/phase4", "research/harness", "research/experiments", "opt/harness"): sys.path.insert(0, os.path.join(ROOT, p))
import p4_t4_robustness as T4, p4                                                                # noqa: E402
import fbsopt as H                                                                               # noqa: E402
SNAP = os.path.join(ROOT, "opt/src/snapshots/main_dev_J.py")
B = dict(rest_frac=0.5, rest_gate=0.06, drag_off_rest=1, drag_switch_r=1.7, drag_tight=[0.03, 1000.0, 0.15], tau_g_final_hand=24.0, final_mu_max=0.002, final_sw_max=0.04)
CONFS = {"champ": {}, "rc110b": B}
CASES = [("E46", "clean", "MH_02_easy", 0), ("E46", "gap_3x4", "MH_02_easy", 0), ("E46", "gain_x2", "MH_04_difficult", 0), ("E46", "noise_heavy", "MH_01_easy", 0),
         ("E46", "acc_ramp_1", "V2_02_medium", 0), ("E46", "acc_ramp_1", "MH_03_medium", 0), ("E46", "acc_ramp_1", "MH_02_easy", 0), ("E46", "acc_ramp_1", "room1", 0),
         ("E46", "ramp_r_0.0005", "room1", 0), ("E46", "rampwin_0.05_10s_persist", "MH_05_difficult", 0), ("E46", "ramp_r_0.01", "V1_01_easy", 0),
         ("E46", "L2+ramp_x_0.002", "V2_02_medium", 0), ("E46", "half_rate", "room1", 0)]
def one(case):
    tn, w, a = T4.case_input(case); w = np.round(w, 6); a = np.round(a, 6); name = case[2]
    fd, path = tempfile.mkstemp(suffix=".csv", prefix="fbs_an17_"); os.close(fd); out = dict(case=list(case))
    try:
        pd.DataFrame(dict(zip(p4.EUROC_COLS, [tn, w[:, 0], w[:, 1], w[:, 2], a[:, 0], a[:, 1], a[:, 2]]))).to_csv(path, index=False)
        order = np.argsort(tn, kind="stable"); m = H.EV.mission(tn[order], H.gt_cached(name))
        for lab, cfg in CONFS.items():
            s = importlib.util.spec_from_file_location("mdj_" + lab, SNAP); M = importlib.util.module_from_spec(s); s.loader.exec_module(M); M.CFG.update(cfg)
            fd2, est = tempfile.mkstemp(suffix=".txt"); os.close(fd2); _p, info = M.run(path, est, k_drag=H.k_loso(name))
            ens, p, q = H.EV.read_estimate(est); os.remove(est); sc = H.EV.score(m, p[order], q[order])
            out[lab] = dict(ate=sc["ate"], wp=sc["wp"], b_abs=sc["b_abs"], **{k: info.get(k) for k in ("rotor_gate", "rotor", "guard", "mode", "rh", "rr0", "rr", "tight", "mu", "sw", "final_long")})
    finally: os.remove(path)
    return out
if __name__ == "__main__":
    with Pool(11) as pool: res = pool.map(one, CASES)
    json.dump(res, open(sys.argv[1], "w"), indent=1, default=float)
    for r in res:
        L = []
        for lab in CONFS:
            x = r[lab]; L.append("%s ate %.3f %s%s rh %s rr0 %s rr %s%s" % (lab, x["ate"], "rotor" if x["rotor"] else "hand", "(guard)" if x["guard"] else "", *(("%.2f" % x[k]) if x.get(k) is not None else "-" for k in ("rh", "rr0", "rr")), " steady" if x.get("tight") else ""))
        print("%-52s | %s" % (str(r["case"])[:52], " | ".join(L)))
