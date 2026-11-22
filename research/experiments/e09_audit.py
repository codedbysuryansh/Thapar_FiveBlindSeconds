#!/usr/bin/env python3
"""E09 - Phase 2 audit: (1) verify every data file against its provenance SHA-256; (2) recompute the Phase 1B headline
numbers from the saved JSON under BOTH skill definitions; (3) v3 front end on all 17 clean streams (rest anchors,
false-rest rejection, thrust-lock geometry, ingest false positives); (4) clean-stream table with the single v3 front end.
Ground truth only scores. Run: .venv/bin/python research/experiments/e09_audit.py"""
import sys, os, json, glob, hashlib, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import fbs3 as f
R = f.fbs.ROOT
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def checksums():
    ok = bad = 0; key = {}
    for pj in sorted(glob.glob(R + "/data/euroc/PROVENANCE_*.json")) + sorted(glob.glob(R + "/data/tumvi/PROVENANCE_*.json")):
        d = json.load(open(pj)); eu = "euroc" in pj; seq = os.path.basename(pj)[11:-5]
        for m in d["members"]:
            p = os.path.join(R, "data/euroc", seq, m["member"]) if eu else os.path.join(R, "data/tumvi", m["member"])
            good = os.path.exists(p) and sha(p) == m["sha256"]; ok += good; bad += (not good)
            if m["member"].endswith(("imu0/data.csv", "state_groundtruth_estimate0/data.csv", "mocap0/data.csv")): key[(seq if eu else m["member"].split("/")[0]) + "/" + "/".join(m["member"].split("/")[-2:])] = m["sha256"][:16]
    json.dump(key, open(R + "/research/results/e09_checksums.json", "w"), indent=1); return ok, bad, len(key)
def work(n):
    s = f.load(n); ok = s.ok; t, wu, au, valid, rows = f.ingest(s.tn, s.w, s.a); anc = f.rest_anchors(t, wu, au, valid)
    old_b, old_quiet = f.gyro_bias(s.t, s.w, s.a); r = {"name": n, "ingest_invalid": int((~valid).sum()), "n_anchors": len(anc), "anchor_t": [round(x[0], 1) for x in anc], "old_quiet": bool(old_quiet)}
    q, bg, quiet, af = f.front3(t, wu, au, valid); r["lock"] = f.thrust_lock(t, af, valid); r["tilt"] = float(np.mean(f.tilt_deg(q[ok], s.gt_q[ok])))
    if s.gt_bg is not None and anc: r["bias_err_mrad"] = float(1e3 * np.linalg.norm(np.mean([x[1] for x in anc], 0) - s.gt_bg.mean(0)))
    from scipy.spatial.transform import Rotation as Rot
    rel = f.qmul(s.gt_q[ok], f.qinv(q[ok])); yaw = np.unwrap(Rot.from_quat(rel).as_euler("zyx")[:, 0]); r["max_yaw_err_deg"] = float(np.degrees(np.abs(yaw - yaw[0]).max()))
    r["motionless"] = f.ate(np.zeros((ok.sum(), 3)), s.gt_p[ok])
    for name, model in [("matern", f.model_matern52(*f.MATERN)), ("spring", f.model_spring(*f.SPRING))]:
        e = f.estimate3(s.tn, s.w, s.a, model=model); r[name] = f.ate(e["p"][ok], s.gt_p[ok]); Rm, _ = f.umeyama(e["p"][ok], s.gt_p[ok]); r[name + "_rpe1"] = f.rpe_disp(s.t[ok], e["p"][ok], s.gt_p[ok], Rm)
    r["motionless_rpe1"] = f.rpe_disp(s.t[ok], np.zeros((ok.sum(), 3)), s.gt_p[ok], np.eye(3))
    aw = f.qrot(q, af); aw -= aw.mean(0); r["heave05"] = f.ate(f.highpass_pos(t, aw, 0.05)[ok], s.gt_p[ok])
    return r
def two_skills(est, ref):
    est, ref = np.array(est), np.array(ref); return 100 * (1 - est.mean() / ref.mean()), 100 * np.mean(1 - est / ref)
if __name__ == "__main__":
    print("checksums ok / bad / key files:", checksums())
    d = json.load(open(R + "/research/results/e02_baselines.json")); ns = [k for k in d if not k.startswith("_")]
    a, b = two_skills([d[n]["sel"]["loso"] for n in ns], [d[n]["motionless"]["ate_se3"] for n in ns]); print("Phase 1B e02 GP LOSO, 17 seq: aggregate ATE reduction %.1f%% | mean per-sequence skill %.1f%%" % (a, b))
    e = json.load(open(R + "/research/results/e05b_drag_smoother.json")); eu = [n for n in e if not n.startswith("_") and not n.startswith("room")]
    for k in ["gp_loso", "drag_loso", "drag_loeo"]:
        a, b = two_skills([e[n][k] for n in eu], [e[n]["motionless"] for n in eu]); print("Phase 1B e05b %-9s EuRoC 11 seq: mean ATE %.3f | aggregate reduction %.1f%% | mean per-sequence skill %.1f%%" % (k, np.mean([e[n][k] for n in eu]), a, b))
    names = f.available()
    with Pool(10) as p: res = p.map(work, names)
    json.dump(res, open(R + "/research/results/e09_audit.json", "w"), indent=1)
    print("seq              anchors(t)            old_quiet lock_g  tilt  bias_err  max_yaw | motionless  heave05  matern  spring | RPE1s: motionless matern spring")
    for r in res:
        print("%-16s %d %-20s %-5s   %.3f  %.2f  %-7s  %6.1f  | %7.3f  %7.3f %7.3f %7.3f | %.3f %.3f %.3f" % (r["name"], r["n_anchors"], str(r["anchor_t"])[:20], r["old_quiet"], r["lock"], r["tilt"], ("%.1f" % r["bias_err_mrad"]) if "bias_err_mrad" in r else "-", r["max_yaw_err_deg"],
              r["motionless"], r["heave05"], r["matern"], r["spring"], r["motionless_rpe1"], r["matern_rpe1"], r["spring_rpe1"]))
    print("ingest false positives on clean streams (invalid samples):", sum(r["ingest_invalid"] for r in res))
    for grp, sel in [("ALL", lambda n: True), ("EuRoC", lambda n: not n.startswith("room")), ("TUM", lambda n: n.startswith("room"))]:
        rr = [r for r in res if sel(r["name"])]; ref = [r["motionless"] for r in rr]
        for k in ["heave05", "matern", "spring"]:
            a, b = two_skills([r[k] for r in rr], ref); print("%-5s %-7s mean ATE %.3f (motionless %.3f) | aggregate reduction %5.1f%% | mean per-sequence skill %5.1f%% | wins %d/%d" % (grp, k, np.mean([r[k] for r in rr]), np.mean(ref), a, b, sum(r[k] < r["motionless"] for r in rr), len(rr)))
