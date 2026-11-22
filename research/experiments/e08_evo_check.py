#!/usr/bin/env python3
"""E08 - cross-check the harness ATE against the evo package itself and exercise the output contract
(TUM text: 'timestamp tx ty tz qx qy qz qw', seconds, one pose per IMU row). Research instrument."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
import fbs
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
out = os.path.join(fbs.ROOT, "research/results/e08"); os.makedirs(out, exist_ok=True)
for n in ["room1", "V1_02_medium", "MH_04_difficult"]:
    s = fbs.load(n); q, bg, quiet, wf, af = fbs.front_end2(s.t, s.w, s.a); p = fbs.gp_smooth(s.t, q, af, np.ones(len(s.t), bool))[0]
    ts = s.tn * 1e-9; est = np.column_stack([ts, p, q]); f_est = os.path.join(out, n + "_estimated_trajectory.txt"); np.savetxt(f_est, est, fmt="%.9f %.6f %.6f %.6f %.7f %.7f %.7f %.7f")
    gts = s.gt_raw_t + s.tn[0] * 1e-9; f_gt = os.path.join(out, n + "_gt.txt"); np.savetxt(f_gt, np.column_stack([gts, s.gt_raw_p, s.gt_raw_q]), fmt="%.9f %.6f %.6f %.6f %.7f %.7f %.7f %.7f")
    tg = file_interface.read_tum_trajectory_file(f_gt); te = file_interface.read_tum_trajectory_file(f_est); rows_in = len(s.tn); rows_out = sum(1 for _ in open(f_est))
    tg2, te2 = sync.associate_trajectories(tg, te, max_diff=0.01); te3 = PoseTrajectory3D(te2.positions_xyz.copy(), te2.orientations_quat_wxyz.copy(), te2.timestamps.copy()); te3.align(tg2, correct_scale=False)
    ape = metrics.APE(metrics.PoseRelation.translation_part); ape.process_data((tg2, te3)); a_evo = ape.get_statistic(metrics.StatisticsType.rmse)
    ape0 = metrics.APE(metrics.PoseRelation.translation_part); ape0.process_data((tg2, te2)); a_raw = ape0.get_statistic(metrics.StatisticsType.rmse)
    rpe = metrics.RPE(metrics.PoseRelation.translation_part, delta=int(round(1.0 / np.median(np.diff(tg2.timestamps)))), delta_unit=metrics.Unit.frames, all_pairs=False); rpe.process_data((tg2, te2)); r_evo = rpe.get_statistic(metrics.StatisticsType.rmse)
    zero = PoseTrajectory3D(np.zeros_like(te2.positions_xyz) + 1e-6 * np.random.default_rng(0).normal(size=te2.positions_xyz.shape), te2.orientations_quat_wxyz.copy(), te2.timestamps.copy())
    rz = metrics.RPE(metrics.PoseRelation.translation_part, delta=int(round(1.0 / np.median(np.diff(tg2.timestamps)))), delta_unit=metrics.Unit.frames, all_pairs=False); rz.process_data((tg2, zero)); zero.align(tg2); az = metrics.APE(metrics.PoseRelation.translation_part); az.process_data((tg2, zero))
    print("%-16s rows in/out %d/%d  matched %d | ATE harness %.4f  evo(aligned) %.4f  evo(no align) %.4f | evo RPE(1 s, trans) est %.4f  motionless %.4f | evo ATE motionless(aligned) %.4f | file %.1f MB" % (
        n, rows_in, rows_out, tg2.num_poses, fbs.ate(p[s.ok], s.gt_p[s.ok]), a_evo, a_raw, r_evo, rz.get_statistic(metrics.StatisticsType.rmse), az.get_statistic(metrics.StatisticsType.rmse), os.path.getsize(f_est) / 1e6))
