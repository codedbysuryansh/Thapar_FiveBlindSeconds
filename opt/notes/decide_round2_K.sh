#!/bin/zsh
# Round 2, batch K decisions (robustness of RC v1.1.0b) and the drag-offset analysis AN25.
cd ~/SamsungPRISM_FiveBlindSeconds/opt
D() { ../.venv/bin/python harness/fbsopt.py decide "$1" --status "$2" --reason "$3" > /dev/null; echo "$1 -> $2"; }
D DEV0-dev-identity-K validation "136/136 DEV outputs byte-identical to the champion (snapshot K = J + edits_13, flags at defaults)"
D X110b-guardloose-final12 rejected "12 s final hand-held gravity: DEV TUM-VI -0.02 % vs X110a (equal); T4 hand-held -0.2 % (= X110a), 16 vs 17 cases > 25 % worse - no robustness gain, the 24 s constant stays"
D X110c-guardloose-final8 rejected "8 s final hand-held gravity: DEV TUM-VI +0.19 % vs X110a (equal); T4 hand-held -0.1 %, 15 cases > 25 % worse - no material robustness gain"
D X111a-blend-1.5-2.0 rejected "Soft switch 1.5-2.0 m: DEV -0.03 % vs X108b (equal); T4 ATE -4.1 % vs -4.3 % (X110a), 17 cases > 25 % worse - removes the knife-edge but neither accuracy nor the T4 tail improves"
D X111b-blend-1.6-1.9 rejected "Soft switch 1.6-1.9 m: DEV +0.05 % vs X108b (equal); T4 ALL -4.96 % vs -5.18 % (X110a), 16 cases > 25 % worse - no benefit"
P="../.venv/bin/python harness/fbsopt.py analysis"
eval $P AN25-drag-offset-wander --title "'Does the wander of the loose drag-row offset tell when the steady prior helps?'" --status done \
  --question "'Can a statistic of the loose solution (rather than its radius) choose the steady drag prior, so that MH_01 (radius 1.3-1.5 m, never switched) also gains and the T4 misfires are avoided?'" \
  --method "'main_dev K, loose prior only (switch off) with the smoother states recorded vs steady prior forced, on DEV (88 EuRoC missions) and E46 seed 0 (319 rotor cases); S = RMS deviation of the 10 s moving average of the two drag-row offsets over flight blocks; rules compared by the mean ATE of the chosen solution (notes/an25b_rules.py)'" \
  --result "'Official DEV: S is large on Machine Hall flights (0.041-0.123) where the steady prior helps (MH_01 -5..-24 %, MH_03-05 -14..-41 %) and small in the Vicon rooms (0.014-0.052) where it is neutral or harmful (V1_03 +14..+60 %). Rule S >= 0.040-0.045: -14.8 % vs loose (radius rule -13.0 %, always steady -13.2 %, oracle -15.6 %), but MH_03 (0.045-0.077) and the V tail (up to 0.052) overlap, so the threshold is a knife-edge. E46 stress: accelerometer ramps make S large in the V rooms and the S rule switches them (acc_ramp_1 V1_03 +308 %, V2_02 +181 %, V2_01 +150 %): mean -1.3 % vs -3.2 % for the radius rule, 47 vs 30 bad choices; hybrids (radius or S with a radius floor) keep the acc_ramp failures.'" \
  --decision "'Not adopted: the statistic is environment-learned and brittle under unmodelled drifts; the radius rule stays. Documented for future work (a drift-aware statistic that tells real offset drift from absorbed velocity)'" \
  --files "'notes/an25_drag_constancy.py, notes/an25_dev.json/.txt, notes/an25_e46s0.json/.txt, notes/an25b_rules.py'"
