#!/bin/zsh
# Learned-model retest: train every fold (in-pool, leave-one-environment-out for EuRoC, leave-one-room-out for TUM-VI) and
# assemble the held-out predictions of all DEV missions into runs/<cand>__loeo for fusion.
cd ~/SamsungPRISM_FiveBlindSeconds/opt/learned
C=X108b-final24-safe-2e-3-0.04; PY=../../.venv/bin/python
for f in inpool MH V1 V2 room1 room2 room3 room4 room5 room6; do
  $PY train.py --fold $f --cand $C > runs_$f.log 2>&1; tail -3 runs_$f.log
done
mkdir -p runs/${C}__loeo
for f in MH V1 V2 room1 room2 room3 room4 room5 room6; do cp runs/${C}__$f/*_A*.npz runs/${C}__$f/*_B*.npz runs/${C}__loeo/ 2>/dev/null; done
ls runs/${C}__loeo | wc -l
