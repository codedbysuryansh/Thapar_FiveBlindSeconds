#!/bin/zsh
# Round 2 analyses (RC v1.1.0b verification and the robustness findings that followed). Registered so negative results stay.
cd ~/SamsungPRISM_FiveBlindSeconds/opt
P="../.venv/bin/python harness/fbsopt.py analysis"
eval $P AN14-t3-protocol-v110b --title "'Official protocol (T3), champion v1.0.0 vs RC v1.1.0b, paired'" --status done \
  --question "'Are the RC gains reproduced on the official protocol (seeds 42 + 5-13, both orders, 340 missions, LOSO k), and is any metric worse?'" \
  --method "'rc/tools/evaluate.py --protocol on rc/main.py; per-mission scores paired with the Phase 4 T3 file of v1.0.0; 95 % bootstrap CI resampling whole sequences (10 000 replicates)'" \
  --result "'EuRoC (11 seq): ATE 2.670 -> 2.294 (-14.1 % [-21.2, -4.8], 10/11 better), waypoints -15.8 % [-23.8, -4.8], after-dropout -13.3 % [-21.8, -2.2], blind b_abs -3.2 % [-13.2, +4.8], b_anc +2.2 % [-1.8, +5.7], RPE_t -15.3 %. TUM-VI (6 rooms): ATE 0.451 -> 0.418 (-7.3 % [-13.8, +1.5], 3/6), waypoints -11.4 % [-18.1, +0.1], drop -5.2 % [-10.4, +2.4], b_abs -1.0 %, b_anc +1.1 % [-7.5, +19.5]. No metric significantly worse. Per sequence: MH_03-MH_05 -23..-30 % ATE; V rooms within +-2 %; room4 -15 %, room6 -20 %, room1 -7 %, room3 +5.6 %, room5 +1.5 %; anchored blind error worse on MH_05 (+12 %), room3 (+30 %), room6 (+33 %). 0 contract failures or fallbacks, guard fired 0.'" \
  --decision "'Official-protocol gains confirmed on fresh seeds; the TUM-VI gain is not significant at 6 rooms and comes from 3 of 6 rooms'" \
  --files "'regression/p5_t3_protocol_rcb.log/.json, notes/an10_t3_paired.py, notes/an10_t3_paired_v100_vs_rcb.txt/.json, notes/an10b_t3_per_sequence.py, notes/an10b_t3_per_sequence_v100_vs_rcb.txt'"
eval $P AN15-t4-rc110b --title "'Robustness (T4) of rc/main.py v1.1.0b: 935 cases'" --status done \
  --question "'Does the release candidate itself survive the Phase 4 robustness inputs, and where does it lose?'" \
  --method "'regression/p5_t4_robustness.py on rc/main.py (E46 stress suite 493, E55 official-degradation windows 306, E60 rotated sensor frames 136), each against the champion; notes/an15_t4_cases.py lists every decision change and the worst cases'" \
  --result "'0 errors, 0 fallbacks. Mean ATE: E46 -2.4 %, E55 -6.7 %, E60 -13.4 %. Decision changes: 1 bias mode (E46 half_rate room1 const -> spline, +5 % ATE, a consequence of the wider rest detector) and 24 radius-guard flips (23 E46 + 1 E55, all Machine Hall flights: the guard now fires and the hand-held solution is used, ATE +8.6..+18.9 %, mean +10.3 %, 22 losses / 2 wins). Worst cases: E46 acc_ramp_1 V2_02 +182 %, ramp_r_0.0005 room1 +52 %, E55 room5 B s60 +42 %, acc_ramp_1 MH_03 +32 %, rampwin MH_05 +30 %. Cases > 25 % worse: E46 6/493, E55 2/306, E60 0/136. The strict criterion (input decisions identical) reports FAIL because of the one mode change.'" \
  --decision "'RC v1.1.0b NOT promoted: the guard flips are a defect of the combination (switch x guard); round 2 (batch K) fixes and re-tests'" \
  --files "'regression/p5_t4_rcb.log/.json, notes/an15_t4_cases.py, notes/an15_t4_cases_rcb.txt'"
eval $P AN16-guard-margin --title "'Margin of the radius guard before and after the drag switch'" --status done \
  --question "'Why does the guard fire on Machine Hall flights in T4 but never on the official sets?'" \
  --method "'rr/rh of every rotor-gate mission from the harness records (rh = hand-held smoother radius on the rotor measurements, rr0 = loose rotor radius, rr = final radius) for the champion-equivalent build and X108b on DEV, DEV2, CONF, STRESS'" \
  --result "'Champion-equivalent (DEV0-J, DEV): max rr/rh 6.95, none >= 8. X108b: DEV max 10.83, CONF max 10.84 (21/220 >= 8), STRESS max 20.6 (1/220 > 12: guard fires, MH_02 crop 0.6). The steady prior enlarges the rotor radius 0.84-2.15x (median 1.54-1.63, toward the true size 4-7 m), so the unchanged 12x threshold now sits inside the normal range of MH_02/MH_04 flights.'" \
  --decision "'Guard to be checked on the loose solution (X110)'" --files "'notes/an16_guard_margin.py'"
eval $P AN17-t4-radii --title "'Radii in the T4 failure cases of RC v1.1.0b'" --status done \
  --question "'What drives the guard flips and the large E46 losses?'" \
  --method "'main_dev J with the champion and X108b configurations on 13 selected T4 cases; rh, rr0, rr, switch and guard recorded'" \
  --result "'Guard flips: the RC loose radius rr0 is already larger than the champion radius (clean MH_02 1.04 -> 2.01 m, rest detector + no drag rows at rest), the switch raises it again (3.67 m), rr/rh 16.7 > 12. acc_ramp_1 (accelerometer ramp to 1 m/s^2, 10x the official bias walk): rr0 1.96-2.06 triggers the switch on V2_02 / MH_03, the steady drag offset cannot follow the ramp and the radius inflates 2.3-2.5x (ATE +182 % / +32 %). V-room flights under gyro ramps sit just above 1.7 m (1.74-1.84) and switch (+20..+25 %).'" \
  --decision "'Guard fix (X110); the acc_ramp misfire is outside the official envelope and no radius statistic separates it (AN22)'" --files "'notes/an17_t4_radii.py, notes/an17_t4_radii.json'"
eval $P AN18-t4-hand-stats --title "'Can input statistics predict where the long hand-held gravity loses?'" --status done \
  --question "'Do the drift statistic D, the residual bias |mu| or the gyro sample noise separate the T4 hand-held cases where the 24 s final gravity loses from those where it wins?'" \
  --method "'The 20 best and 20 worst hand-held T4 cases of rc v1.1.0b re-run with main_dev J (X108b configuration) recording D, |mu|, sw, length and whether the long gravity was used'" \
  --result "'No separation: losses D 0.0008-0.0016, |mu| 0.0005-0.0018, sw 0.009-0.027 rad/s; gains D 0.0011-0.0015, |mu| 0.0009-0.0012, sw 0.009-0.027. The effect is sequence-dependent (room6/room1 gain 20-26 %, room3/room5 lose 13-42 %); short MH ground prefixes (E55 p15, hand-held) lose 8-14 %.'" \
  --decision "'No new gate; shorter final time constants (12 s, 8 s) tested for robustness instead (X110b/c)'" --files "'notes/an18_t4_hand_stats.py, notes/an18_t4_hand_stats.json'"
eval $P AN19-compile-transient --title "'Heap peak of compiling main.py (measurement-window risk)'" --status done \
  --question "'How much traced heap does compiling main.py take if a grader starts tracemalloc before main.py is compiled (first run, no .pyc)?'" \
  --method "'tracemalloc around compile(source) for both builds on Python 3.10.20, 3.12.7, 3.13.7 (macOS) and 3.13.18 (Linux x86-64); per-top-level-statement compile peaks; source without comments and docstrings'" \
  --result "'Whole file 10.3-11.2 MB peak (v1.0.0 and v1.1.0b alike; 0.4-0.9 MB retained), on every version; proportional to the code size (comments and docstrings do not matter: 10.81 MB without them); largest single top-level statement 1.26 MB (class Data). With a cached .pyc the +main window is 3.1 MB.'" \
  --decision "'Not changed: only a code-in-string loader (exec of separately compiled sections) or several modules would bound it, both against the single readable main.py; documented in the README as a measurement-window risk and left to the user'" --files "'notes/an19 (commands in the session log), /tmp/fbs_compile_peak.py'"
eval $P AN20-t5-xeon-rc110b --title "'Efficiency of RC v1.1.0b on the 2.1 GHz Xeon cloud vCPU (T5)'" --status done \
  --question "'Does RC v1.1.0b stay within the runtime and heap targets on the slow reference machine?'" \
  --method "'tools/measure.py (5 repeats, single thread) for the champion and rc on MH_01 A/B, room2 A, MH_04 A, 10-minute rotor and hand-held inputs; Python 3.13.18, NumPy 2.5.3'" \
  --result "'ms per IMU-second (min) champion -> rc: MH_01 A 7.52 -> 7.10, MH_01 B 9.62 -> 8.91, room2 7.61 -> 7.84, MH_04 9.37 -> 9.05, long rotor 9.31 -> 11.15 (+20 %, second rotor smoother pass), long hand 7.89 -> 8.17; worst median 11.45 (target <= 12, limit 20). Own heap unchanged: 2.61-2.64 MB (184 s), 6.86 / 6.69 MB (600 s); +main 3.14-3.15 / 7.37 / 7.20 MB.'" \
  --decision "'Efficiency criterion met'" --files "'notes/measure_xeon_rcb/*.json'"
eval $P AN21-x86-identity-rc110b --title "'Cross-machine identity and output contract of RC v1.1.0b'" --status done \
  --question "'Are the outputs byte-identical on Linux x86-64 and macOS arm64, and do all 34 official-format files pass T1?'" \
  --method "'rc/main.py on the 34 seed-42 official-format inputs on the Mac (Python 3.12.7, NumPy 2.5.4) and on the Xeon (Python 3.13.18, NumPy 2.5.3); tools/check_output.py; SHA-256 of every output'" \
  --result "'T1 34/34 on both machines; all 34 SHA-256 identical across machines'" \
  --decision "'T1 and the x86-64 part of T6 pass'" --files "'regression/p5_t1_rc.sh, regression/p5_t1_rcb.log, regression/t1_rcb/SHA256_mac.txt, regression/p5_t6_x86_rcb.log'"
eval $P AN22-switch-ratio-gate --title "'Gate the drag switch on rr0/rh? (X112, dropped before running)'" --status done \
  --question "'Can the ratio of the loose rotor radius to the hand-held guard radius separate the official switches (gain) from the T4 misfires (loss)?'" \
  --method "'rr0/rh of every switched mission of X108b on DEV, DEV2, CONF, STRESS against the misfire cases of AN17 (1.7-3.1)'" \
  --result "'MH_03 switches at rr0/rh 1.75-1.9 on every set with -10..-42 % ATE; the misfires lie at 1.7-3.1. MH_03 rr0 2.03-2.40 m also overlaps the misfire radii 1.74-2.06 m.'" \
  --decision "'No ratio or radius threshold separates them; X112 not run. Residual risk: strong accelerometer or gyro ramps (beyond the official script) can trigger the steady drag prior'" --files "'notes/make_batch_K.py (docstring)'"
