"""AN18: which input statistics separate the hand-held T4 cases where the long (24 s) final gravity of RC v1.1.0b loses from
those where it wins? For the hand-held T4 cases with the largest ATE losses and gains (from the T4 JSON of rc v1.1.0b vs the
champion), re-run main_dev J with the X108b configuration and record the drift statistic D, the residual constant bias |mu|,
the gyro sample noise sw, the recording length and whether the long gravity was used. Usage: an18_t4_hand_stats.py T4.json OUT.json"""
import os, sys, json, tempfile, importlib.util, numpy as np, pandas as pd
from multiprocessing import Pool
ROOT = "~/SamsungPRISM_FiveBlindSeconds"
for p in ("research/phase4", "research/harness", "research/experiments", "opt/harness"): sys.path.insert(0, os.path.join(ROOT, p))
import p4_t4_robustness as T4, p4                                                                # noqa: E402
import fbsopt as H                                                                               # noqa: E402
SNAP = os.path.join(ROOT, "opt/src/snapshots/main_dev_J.py")
B = dict(rest_frac=0.5, rest_gate=0.06, drag_off_rest=1, drag_switch_r=1.7, drag_tight=[0.03, 1000.0, 0.15], tau_g_final_hand=24.0, final_mu_max=0.002, final_sw_max=0.04)
def one(case):
    tn, w, a = T4.case_input(case); w = np.round(w, 6); a = np.round(a, 6); name = case[2] if case[0] == "E46" else case[1]
    fd, path = tempfile.mkstemp(suffix=".csv", prefix="fbs_an18_"); os.close(fd); out = dict(case=case, seconds=float((tn.max() - tn.min()) * 1e-9))
    try:
        pd.DataFrame(dict(zip(p4.EUROC_COLS, [tn, w[:, 0], w[:, 1], w[:, 2], a[:, 0], a[:, 1], a[:, 2]]))).to_csv(path, index=False)
        s = importlib.util.spec_from_file_location("mdj", SNAP); M = importlib.util.module_from_spec(s); s.loader.exec_module(M); M.CFG.update(B)
        fd2, est = tempfile.mkstemp(suffix=".txt"); os.close(fd2); _p, info = M.run(path, est, k_drag=H.k_loso(name)); os.remove(est)
        out.update({k: info.get(k) for k in ("rotor", "mode", "D", "mu", "sw", "final_long", "h")})
    finally: os.remove(path)
    return out
if __name__ == "__main__":
    d = json.load(open(sys.argv[1]))["cases"]
    hand = [r for r in d if "cand_score" in r and "champ_score" in r and not r["cand"]["rotor"] and not r["champ"]["rotor"]]
    for r in hand: r["rel"] = r["cand_score"]["ate"] / r["champ_score"]["ate"] - 1
    hand.sort(key=lambda r: r["rel"]); pick = hand[:20] + hand[-20:]
    cases = [tuple(r["case"]) if r["case"][0] != "E60" else None for r in pick]
    sel = [(c, r) for c, r in zip(cases, pick) if c is not None]
    def fix(c): return c if c[0] != "E55" else (c[0], c[1], c[2], c[3], c[4], c[5])
    with Pool(11) as pool: res = pool.map(one, [fix(c) for c, _ in sel])
    rows = []
    for (c, r), x in zip(sel, res):
        x["rel"] = r["rel"]; x["ate_champ"] = r["champ_score"]["ate"]; x["ate_cand"] = r["cand_score"]["ate"]; rows.append(x)
        print("%-52s %+7.1f%%  %6.1f s  mode %-6s long %-5s D %s mu %s sw %s" % (str(c)[:52], 100 * x["rel"], x["seconds"], x.get("mode"), x.get("final_long"),
              *(("%.5f" % x[k]) if isinstance(x.get(k), (int, float)) and x.get(k) == x.get(k) else str(x.get(k)) for k in ("D", "mu", "sw"))))
    json.dump(rows, open(sys.argv[2], "w"), indent=1, default=float)
