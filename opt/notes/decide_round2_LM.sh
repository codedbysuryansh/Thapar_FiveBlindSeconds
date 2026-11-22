#!/bin/zsh
# Round 2 (guard fix) and round 3 (fine-tuning) decisions.
cd ~/SamsungPRISM_FiveBlindSeconds/opt
D() { ../.venv/bin/python harness/fbsopt.py decide "$1" --status "$2" --reason "$3" > /dev/null; echo "$1 -> $2"; }
D DEV0-dev-identity-L validation "136/136 DEV outputs byte-identical to the champion (snapshot L = K + edits_14, flags at defaults)"
D X110a-guardloose rejected "Guard on the loose solution without a fallback: identical to X108b on DEV; T4 935: ATE -5.18 % (X108b -4.89 %) but cases > 25 % worse 17 vs 8 - where the final-radius guard of X108b fired, the steady solution is now used even when it is bad (MH_02 stress cases +25..+54 %). Superseded by X110d"
D X110d-guard-revert accepted "Guard on the loose solution; a steady solution that fails the same guard reverts to the loose rotor solution. Identical to X108b on DEV (136/136); T4 935 cases: no errors or fallbacks, ATE -5.27 % (X108b -4.89 %, X110a -5.18 %), cases > 25 % worse 8 (= X108b), rotor/guard decision changes 8 (X108b 24). Built as RC-1.1.0c"
for c in X116a-steady-sc0.02 X116b-steady-sc0.04 X116c-steady-tau3000 X116d-steady-sc0_0.10 X116e-steady-sc0_0.25 X116f-steady-tau300; do
  D $c rejected "Steady-prior variant of the switched branch within +-0.3 % of X110d on DEV (EuRoC CIs straddle 0): the current prior (sd 0.03, tau 1000 s, initial sd 0.15) sits on a flat optimum"; done
D X117a-matern-ell1.25 rejected "Hand-held Matern ell 1.25 s with the 24 s final gravity: TUM-VI -0.41 % [-2.71, +2.74] on DEV - not significant"
D X117b-matern-ell0.8 rejected "Matern ell 0.8 s: TUM-VI +3.78 %, pooled +0.96 % [+0.04, +2.15] - worse"
D X117c-matern-sb0.07 rejected "Matern sigma_b 0.07: TUM-VI -0.26 % [-0.65, +0.27] - not significant"
D X117d-matern-sb0.14 rejected "Matern sigma_b 0.14: TUM-VI +0.46 % - not significant"
D X117e-matern-taub40 rejected "Matern tau_b 40 s: TUM-VI -0.42 % [-0.65, -0.09] on DEV - significant but negligible (below the 0.5 % practical threshold; round 1 P09h gave -0.24 % for the same change); not worth a re-validation of the release"
D X117f-matern-sp3 rejected "Matern sigma_p 3 m: TUM-VI +1.25 % [-3.16, +6.91] - not better"
D RC-1.1.0b rejected "Superseded by RC-1.1.0c: its own T4 run found 24 radius-guard flips on Machine Hall flights (AN15/AN16); the accuracy record (T3, CONF) carries over to RC-1.1.0c, which is byte-identical to it on every official set"
