#!/bin/zsh
cd ~/SamsungPRISM_FiveBlindSeconds/opt
D() { ../.venv/bin/python harness/fbsopt.py decide "$1" --status "$2" --reason "$3" > /dev/null; echo "$1 -> $2"; }
for v in H I; do D DEV0-dev-identity-$v validation "136/136 DEV outputs byte-identical to the champion (snapshot $v, flags off)"; done
for c in P08c-sc-0.08 P15-sc0.04-tauc100 P15-sc0.04-tauc1000 P15-sc0.04-tauc300 P15-sc0.06-tauc100 P15-sc0.06-tauc1000 P15-sc0.06-tauc300 P15-sc0.08-tauc300 P15-sc0.1-tauc100 P16-sc0_0.15-sc0.05-tauc300; do
  D $c rejected "Fixed tighter drag-bias prior: EuRoC gains on DEV come from MH (-4..-23 %) while V1 loses (+1..+21 %); environment-level LOEO +1.5 % (worse than the champion); superseded by the radius-switched prior X044"; done
D P09h-matern_tau_b-40 rejected "Negligible: TUM-VI -0.24 % on DEV (below any practical effect); champion value kept"
D P11a-k2-0.15 rejected "Negligible: -0.05 % pooled / -0.23 % TUM-VI on DEV; champion value kept"
for c in X041a-rest_frac-0.15 X041b-rest_frac-0.30 X041c-rest_frac-0.30-gate-0.05 X041d-rest_frac-0.50-gate-0.05 X041e-rest30-g05-dragoff X041f-rest15-dragoff; do D $c rejected "Dominated by X041g (gate 0.06 admits the rotor-on ground windows; DEV EuRoC -2.31 % vs -0.57..-1.43 %)"; done
D X041j-rest80-g06 rejected "Identical outputs to X041g (the 50 % cap no longer binds at gate 0.06)"
D X044-switch-r1.6-p16a rejected "Radius threshold 1.6 m: DEV EuRoC -12.90 % vs -13.05 % at 1.7 m (chosen)"
D X044-switch-r1.8-p16a rejected "Radius threshold 1.8 m: DEV EuRoC -12.04 % vs -13.05 % at 1.7 m (chosen)"
D X044-switch-r1.7-p16b rejected "Milder steady prior (0.15/0.05/300): DEV EuRoC -12.63 % vs -13.05 % with 0.15/0.03/1000 (chosen)"
D X106-combo7 rejected "Heteroscedastic accelerometer rows add nothing (X005: -0.02 %); combination 7 = combination 5"
