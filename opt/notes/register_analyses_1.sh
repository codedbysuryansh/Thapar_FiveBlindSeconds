#!/bin/zsh
# Analyses and diagnostics of the first session of the Phase 5 programme (registered so negative results are kept).
cd ~/SamsungPRISM_FiveBlindSeconds/opt
P="../.venv/bin/python harness/fbsopt.py analysis"
eval $P AN00-error-decomposition --title "'Where the champion error lives (CONF baseline)'" --status done \
  --question "'Which directions and sequences dominate the error?'" --method "'CONF results of CHAMP-1.0.0: horizontal vs vertical ATE after the same SE(3) alignment, per-sequence ATE against motionless, error by thirds of the flight'" \
  --result "'Error is horizontal: EuRoC ate_h 2.60 vs ate_v 0.54 m; TUM-VI 0.44 vs 0.07. MH flights 4.2-5.1 m (motionless 4.3-6.9), V rooms 0.9-2.1, TUM rooms 0.31-0.81 (room4 worst). MH_01/02 have their largest error in the first third (ground phase).'" \
  --decision "'Vertical-prior candidate (X010) deprioritised (vertical is 4 % / 2 % of the MSE); MH and the ground phase targeted'" --files "'notes/diag/CHAMP-1.0.0/*.png'"
eval $P AN01-ground-segments --title "'Still (ground) segments and rest-detector coverage'" --status done \
  --question "'Are the drones still on the ground for long, and does the rest detector see it?'" --method "'Ground-truth speed < 0.03 m/s in 0.5 s windows vs the champion rest mask (DEV seed 1001 A)'" \
  --result "'MH flights sit still 7.5-24 s (MH_01 20.2-44.2 s); the rest mask covers 0-15 % of it (quietest 5 % of windows, gyro gate 0.035). Still gyro sd 0.012-0.034 vs 0.10-0.28 in flight.'" \
  --decision "'Led to X041 (wider rest detector) - accepted direction (X041g)'" --files "'notes/an01_ground_segments.py, .json'"
eval $P AN02-drag-model-structure --title "'Rotor-drag measurement model structure'" --status rejected \
  --question "'Does per-axis k, thrust-scaled k, quadratic drag, vertical-speed coupling or a full 2x2 matrix explain the body-frame specific force better than one linear k?'" \
  --method "'Clean EuRoC + ground-truth velocity/orientation, 0.5 and 2 Hz low-pass, shared coefficients leave-one-flight-out, per-flight offsets centred'" \
  --result "'No variant reduces the velocity-equivalent noise on the V flights by more than ~3 %; quadratic and vertical-coupling terms are worse out of sample. (MH numbers were contaminated by hand-carried phases; corrected flight mask in AN06.)'" \
  --decision "'Linear single-k drag kept'" --files "'notes/an02_drag_models.py, .json'"
eval $P AN04-X025-likelihood-selection --title "'Type-II maximum likelihood choice of the drag-bias prior'" --status rejected \
  --question "'Does the rotor filter innovation likelihood pick the better drag-bias prior per flight?'" --method "'Six drag-bias priors (all existing DEV candidates); forward-filter log-likelihood per DEV EuRoC mission; per-mission errors of the picked candidate'" \
  --result "'Anti-correlated: the likelihood picks the loose prior on every MH mission (where tight is 9-31 % better) and tight priors on V (where tight is worse); per-mission ML choice gives composite +1.34 % (worse than the champion).'" \
  --decision "'Rejected (likelihood rewards absorbing ground-phase misfit, not position accuracy)'" --files "'notes/an04_drag_ml.py, .json'"
eval $P AN05-07-drag-residuals --title "'Drag-row residual structure: angular rate, thrust coupling, slow offset wander'" --status rejected \
  --question "'Is the drag misfit explained by angular rate, thrust-axis misalignment (thrust-coupled offset) or a wandering offset?'" \
  --method "'Clean EuRoC + ground truth with a rotor-vibration flight mask: residual vs |w|, thrust, speed, jerk; per-axis thrust regressors (own and leave-one-flight-out); slow (0.05 Hz) residual and offsets by thirds'" \
  --result "'Residual 0.09-0.18 m/s^2, weakly correlated with |w| (0.06-0.25) and thrust (0.05-0.33); thrust regressors change residuals by < 4 % (slow part unchanged); slow offset wander 0.023-0.049 m/s^2 in every flight.'" \
  --decision "'No model change (X026 thrust coupling rejected); the drag offset is not constant in any flight'" --files "'notes/an05_drag_residuals.py, an06_drag_bias_drift.py, an07_thrust_coupling.py'"
eval $P AN08-drag-bias-tracks --title "'Smoother drag-bias tracks against the ground-truth-implied offset'" --status done \
  --question "'Why does a tight drag-bias prior help MH and hurt V1?'" --method "'Smoothed states 12/13 under loose and tight priors vs offset implied by ground-truth velocity in the estimator frame (V1_02, MH_04, DEV seed 1001 A)'" \
  --result "'MH_04: on the ground with rotors spinning the implied offset is -0.65 m/s^2 (ground reaction), +0.1 after take-off; the loose bias learns the ground value and needs ~20 s to unlearn it (velocity error up to 1 m/s); in flight the loose c2 also wanders by ~0.13 m/s^2 against the implied mean. V1_02: implied offset wanders +-0.1 at ~10 s scale, neither prior follows it.'" \
  --decision "'Led to X042 (no drag rows at rest) and X044 (workspace-switched prior)'" --files "'notes/an08_bias_tracks.py, notes/diag2/an08_*.png'"
eval $P AN09-radius-switch --title "'Offline test of a radius-switched drag prior'" --status done \
  --question "'Does the loose rotor solution radius separate flights where the tight prior helps?'" --method "'Per-mission mixing of DEV results of X042f (loose) and X042c/d/e (tight) by the loose solution radius threshold'" \
  --result "'Radius >= 1.7 m picks tight on 32/88 EuRoC missions: composite -13.05 % (always tight -13.20 %), MH -18.7 %, V1 +0.5 % (always tight +8.9 %), V2 +0.2 %. Guard radius rh (hand smoother on rotor data) does not separate the workspaces (0.26-1.24 m, MH_01 0.39 vs V1_02 0.84).'" \
  --decision "'X044 implemented on the loose solution radius rr; X043 (rh-based) rejected (never fires)'" --files "'notes/an09_radius_switch.py'"
eval $P AN10-ORACLE-k --title "'Per-flight drag coefficient oracle'" --status rejected \
  --question "'Does the leave-one-out k cost accuracy, and does it explain the V1 losses of the tight prior?'" --method "'Candidates ORACLE-k-champ / -P16tight / -combo2 with each flight ground-truth k (DEV)'" \
  --result "'No effect: champion -0.08 % [-0.48, +0.43]; tight prior with own k keeps the V1 loss (+8.4 % vs +8.2 %); combination 2 -12.4 % vs -13.1 %.'" \
  --decision "'Online k estimation (X021) not pursued: the drag coefficient is not a limiting factor'" --files "'candidates/ORACLE-k-*'"
eval $P AN11-x86-docker --title "'Genuine Linux x86-64 Docker test of v1.0.0'" --status done \
  --question "'Does the champion produce the same output in an x86-64 container, and how fast is it there?'" \
  --method "'Cloud Xeon 2.1 GHz; Docker 29.9.2; base image debootstrapped Ubuntu 24.04 minbase (python:3.12-slim cannot be pulled: registries blocked); NumPy 2.5.5 wheel; release Dockerfile otherwise; --network none'" \
  --result "'34/34 seed-42 outputs byte-identical to macOS arm64; own heap 2.62 MB; 7.41-7.78 ms per IMU-second (min). Found: importing main.py without a bytecode cache compiles it with a 10.57 MB transient traced peak (0.43 MB retained).'" \
  --decision "'Portability confirmed; compile-time peak recorded as a measurement risk for the memory workstream'" --files "'notes/x86_docker/'"
eval $P AN12-profile --title "'Runtime profile of v1.0.0'" --status done \
  --question "'Where does the runtime go?'" --method "'cProfile on MH_01_easy A42 and room2 A42; stage timers of tools/measure.py'" \
  --result "'MH_01 A (0.76 s on the M3): rotor smoother 0.22 s, front-end spline passes 0.19 s (quaternion prefix products ~0.1 s), guard hand smoother 0.09 s, output 0.07 s, measurements 0.06 s, parse 0.05 s; per-step NumPy call overhead dominates the smoothers (18k np.linalg.solve calls).'" \
  --decision "'Input to the runtime workstream'" --files "'notes/an03_profile.py, an03_profile.txt'"
