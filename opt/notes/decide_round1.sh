#!/bin/zsh
# Decisions after the first promotion round (CONF, STRESS, DEV2 and the 935-case T4 robustness suite).
cd ~/SamsungPRISM_FiveBlindSeconds/opt
D() { ../.venv/bin/python harness/fbsopt.py decide "$1" --status "$2" --reason "$3" > /dev/null; echo "$1 -> $2"; }
D X104-combo5 rejected "CONF -11.35 % [-17.53, -3.73] (EuRoC -11.67 %, TUM-VI -8.25 %) and STRESS -9.19 %, but T4 robustness FAILS: the 24 s hand-held gravity (front end and final attitude) makes the hand-held stress cases +93 % ATE (gyro-bias ramps / random walks +100..+430 %) and short MH prefixes (hand-held path) up to x7 (E55 p25 MH_01 0.27 -> 1.93 m); superseded by X108b"
D RC-1.1.0a rejected "Byte-identical build of X104-combo5 (DEV 136/136, CONF 340/340, S42 34/34); fails T4 robustness as X104 (see X104-combo5); superseded by RC-1.1.0b"
D X105-combo6 rejected "No gain over X104 on CONF (TUM -8.04 vs -8.25 %) or STRESS, +20 % runtime (second hand-held smoother pass)"
D X107a-final24-const rejected "Robustness: T4 hand subset +4.7 % ATE, 38/338 cases > 25 % worse (heavy gyro noise +73 %, gyro bias constant/random walk +45 %, short MH prefixes +34 %); fixed by the clean-gyro conditions of X108b"
D X107b-final12-const rejected "Robustness as X107a (+3.9 %, 31/338 > 25 % worse); dominated by X108b"
D X107g-final8-const rejected "Robustness +2.9 %, 28/338 > 25 % worse (heavy gyro noise +52 %); dominated by X108b"
D X107c-taug6 rejected "Robustness: T4 hand subset +37 % ATE (gyro ramps +130..+200 %)"
D X107e-final24-always rejected "Robustness: T4 hand subset +30 % ATE (long gravity also in the drifting-bias mode)"
D X107d-acccomp rejected "Robust (T4 hand +4.9 %, worst condition +21 %) but smaller TUM-VI gain (DEV -4.67 %) and +20 % runtime; dominated by X108b"
D X107f-rotor-only accepted "Rotor part of X104 (rest 50 %/0.06 + no drag rows at rest + radius-switched drag prior): CONF EuRoC -11.67 % [-18.50, -2.89], STRESS -9.27 %, DEV2 -12.38 %; T4 935 cases: no errors, ATE -3.8 % (rotor -6.0 %); kept as the rotor half of X108b"
D X108a-final24-const-stats validation "Statistics run (= X107a on build I): official-degradation hand-held missions have |mu| <= 0.00163 rad/s and gyro sample noise <= 0.0295 rad/s on DEV; used to set the X108b thresholds"
D X108b-final24-safe-2e-3-0.04 accepted "CONF ALL -11.21 % [-17.41, -3.52] (EuRoC -11.67 % [-18.50, -2.89], TUM-VI -6.46 % [-10.50, +0.01], TUM ATE -7.60 %), STRESS -9.10 % [-13.93, -3.15] (both datasets significant), DEV2 -11.83 %; no component significantly worse; T4 935 cases: no errors or fallbacks, ATE -4.0 %, 8/833 cases > 25 % worse, worst condition acc_ramp_1 +15 %. Built as RC-1.1.0b"
D X108c-final24-safe-1.5e-3-0.035 rejected "Equivalent to X108b (DEV TUM -8.06 vs -7.95 %, T4 -1.1 vs -1.2 %); the stricter thresholds have no measurable advantage"
D X108d-final12-safe-2e-3-0.04 rejected "Equivalent to X108b (DEV TUM -7.98 %); no advantage of the 12 s final constant"
D X044-switch-r1.7-p16a accepted "Radius-switched drag prior: CONF EuRoC -11.67 %, STRESS -9.27 %, DEV2 -12.38 %; part of X107f / X108b"
D X041g-rest50-g06 accepted "Rest detector 50 %/0.06: DEV -2.31 %, DEV2 -1.67 % (EuRoC); CONF -0.32 % [-1.78, +0.94] alone (n.s.) but part of the accepted rotor package (it enables the ground handling of X042/X044)"
for c in X002-taug_hand-4.5 X002-taug_hand-6 X002-taug_hand-8 X002-taug_hand-12 X002-taug_hand-16 X002-taug_hand-24 X002-taug_hand-40 X003-taug_hand-6-acccomp1 X003-taug_hand-8-acccomp1 X003-taug_hand-12-acccomp1 X003-taug_hand-24-acccomp1 X100-combo1 X101-combo2 X102-combo3 X103-combo4; do
  D $c rejected "Long hand-held gravity in the front end: official-protocol gain (TUM-VI -6..-10 %) but fragile under gyro faults (T4: the 24 s variant +93 % on hand-held stress cases); superseded by the clean-gyro final gravity of X108b"
done
for c in X001a-acccomp-1-hand X001b-acccomp-2-hand X001c-acccomp-1-both X001d-acccomp-1-hand-g05; do D $c rejected "Acceleration-compensated gravity works (TUM-VI -2.7..-5.4 % on DEV) but costs a second smoother pass (+20 % runtime) and is dominated by X108b"; done
for c in X042a-rest-dragoff-loose X042f-rest-dragoff-loose-F; do D $c rejected "No drag rows at rest with the loose drag prior: -2.4 % EuRoC (= the rest detector alone); the rest-row exclusion only pays together with the switched prior (kept inside X044)"; done
for c in X042b-rest-dragoff-sc0.1-tau300 X042c-rest-dragoff-p16b X042d-rest-dragoff-p16a X042e-rest-dragoff-sc0.1-tau1000; do D $c rejected "Fixed tight drag prior: MH -13..-21 % but V1 +3..+9 % (environment-dependent); replaced by the radius switch of X044, which keeps the MH gain with V1 +0.5 %"; done
