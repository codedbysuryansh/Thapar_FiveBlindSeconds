#!/usr/bin/env python3
"""E01 - inspect the real IMU telemetry (EuRoC MAV + TUM-VI). Research instrument.
Run: .venv/bin/python research/experiments/e01_inspect.py  -> research/results/e01_inspect.json"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import fbs
from scipy.spatial.transform import Rotation as Rot
out = {}
hdr = "seq        N     dur  fs_med  dt_min  dt_max  dt_sd_us gaps  preGT postGT |a|mean  g_axis(sensor)        rest%  tilt_mean tilt_p95  bg_est(mrad/s)        bg_gt(mrad/s)         ba_gt(m/s2)"
print(hdr)
for n in fbs.available():
    s = fbs.load(n); dt = np.diff(s.t); med = np.median(dt); r = {}
    r["n"] = len(s.t); r["duration_s"] = float(s.t[-1]); r["fs_median_hz"] = float(1 / med)
    r["dt_min_ms"] = float(dt.min() * 1e3); r["dt_max_ms"] = float(dt.max() * 1e3); r["dt_std_us"] = float(dt.std() * 1e6)
    r["n_gaps_gt_1p5dt"] = int((dt > 1.5 * med).sum()); r["n_nonpositive_dt"] = int((dt <= 0).sum())
    r["dt_unique_ns_top"] = [[int(a), int(b)] for a, b in zip(*np.unique(np.diff(s.tn), return_counts=True))][:0]
    u, c = np.unique(np.diff(s.tn), return_counts=True); o = np.argsort(-c)[:4]; r["dt_ns_modes"] = [[int(u[i]), int(c[i])] for i in o]
    ok = s.ok; r["imu_before_gt_s"] = float(s.t[ok][0]); r["imu_after_gt_s"] = float(s.t[-1] - s.t[ok][-1]); r["gt_coverage_frac"] = float(ok.mean())
    r["gt_rate_hz"] = float(1 / np.median(np.diff(s.gt_raw_t))); r["gt_gaps_gt_50ms"] = int((np.diff(s.gt_raw_t) > 0.05).sum())
    an = np.linalg.norm(s.a, axis=1); r["acc_norm_mean"] = float(an.mean()); r["acc_norm_std"] = float(an.std())
    gm = s.a.mean(0); r["mean_specific_force_sensor"] = gm.tolist(); r["first2s_specific_force_sensor"] = s.a[s.t < 2].mean(0).tolist()
    r["gyro_mean"] = s.w.mean(0).tolist(); r["gyro_std"] = s.w.std(0).tolist(); r["acc_std"] = s.a.std(0).tolist()
    r["gyro_absmax"] = float(np.abs(s.w).max()); r["acc_absmax"] = float(np.abs(s.a).max())
    rm = fbs.rest_mask(s.t, s.w, s.a); r["rest_frac"] = float(rm.mean())
    if rm.sum() > 100:
        r["rest_gyro_mean"] = s.w[rm].mean(0).tolist(); r["rest_gyro_std"] = s.w[rm].std(0).tolist(); r["rest_acc_std"] = s.a[rm].std(0).tolist()
        r["rest_first_s"] = float(s.t[rm][0]); r["rest_last_s"] = float(s.t[rm][-1])
        r["rest_gyro_noise_density"] = (np.array(r["rest_gyro_std"]) * np.sqrt(med)).tolist(); r["rest_acc_noise_density"] = (np.array(r["rest_acc_std"]) * np.sqrt(med)).tolist()
    # high-frequency content (vibration): std of first difference / sqrt(2)
    r["acc_hf_std"] = (np.diff(s.a, axis=0).std(0) / np.sqrt(2)).tolist(); r["gyro_hf_std"] = (np.diff(s.w, axis=0).std(0) / np.sqrt(2)).tolist()
    q, bg = fbs.attitude(s.t, s.w, s.a); r["bg_est"] = bg.tolist()
    gw = fbs.qrot(s.gt_q[ok], s.a[ok]).mean(0); r["gt_frame_gravity_dir"] = (gw / np.linalg.norm(gw)).tolist(); r["gt_frame_mean_specific_force_norm"] = float(np.linalg.norm(gw))
    ge = fbs.qrot(fbs.qinv(q[ok]), np.array([0, 0, 1.0])); gg = fbs.qrot(fbs.qinv(s.gt_q[ok]), gw / np.linalg.norm(gw))
    tilt = np.degrees(np.arccos(np.clip(np.sum(ge * gg, 1), -1, 1))); r["tilt_err_mean_deg"] = float(tilt.mean()); r["tilt_err_p95_deg"] = float(np.percentile(tilt, 95))
    # heading drift relative to GT over the covered span (gyro-only heading after bias refinement)
    rel = fbs.qmul(s.gt_q[ok], fbs.qinv(q[ok])); yaw = np.unwrap(Rot.from_quat(rel).as_euler("zyx")[:, 0]); r["yaw_drift_deg"] = float(np.degrees(yaw[-1] - yaw[0])); r["yaw_drift_span_s"] = float(s.t[ok][-1] - s.t[ok][0])
    if s.gt_bg is not None:
        r["bg_gt_mean"] = s.gt_bg.mean(0).tolist(); r["bg_gt_range"] = np.ptp(s.gt_bg, 0).tolist(); r["ba_gt_mean"] = s.gt_ba.mean(0).tolist(); r["ba_gt_range"] = np.ptp(s.gt_ba, 0).tolist()
    r["gt_speed_mean"] = float(np.linalg.norm(s.gt_v[ok], axis=1).mean()); r["gt_speed_max"] = float(np.linalg.norm(s.gt_v[ok], axis=1).max())
    out[n] = r
    f3 = lambda v: "[" + " ".join("%6.1f" % (1e3 * x) for x in v) + "]"
    print("%-8s %6d %6.1f %6.2f %6.3f %7.3f %7.1f %4d %6.2f %6.2f %6.3f  [%5.2f %5.2f %5.2f] %5.1f  %6.2f %8.2f   %s %s %s" % (
        n[:8], r["n"], r["duration_s"], r["fs_median_hz"], r["dt_min_ms"], r["dt_max_ms"], r["dt_std_us"], r["n_gaps_gt_1p5dt"], r["imu_before_gt_s"], r["imu_after_gt_s"],
        r["acc_norm_mean"], *(gm / np.linalg.norm(gm)), 100 * r["rest_frac"], r["tilt_err_mean_deg"], r["tilt_err_p95_deg"], f3(bg),
        f3(r["bg_gt_mean"]) if s.gt_bg is not None else "-", "[" + " ".join("%6.3f" % x for x in r["ba_gt_mean"]) + "]" if s.gt_bg is not None else "-"))
os.makedirs(os.path.join(fbs.ROOT, "research/results"), exist_ok=True)
json.dump(out, open(os.path.join(fbs.ROOT, "research/results/e01_inspect.json"), "w"), indent=1)
