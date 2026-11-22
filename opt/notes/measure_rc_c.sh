#!/bin/zsh
# Sequential memory / runtime measurement (tools/measure.py, fresh processes) of the champion and the release candidate on the
# Mac (no other jobs running). Results: opt/notes/measure/<build>_<input>.json
cd ~/SamsungPRISM_FiveBlindSeconds
OUT=opt/notes/measure_c; mkdir -p $OUT
PY=~/SamsungPRISM_FiveBlindSeconds/.venv/bin/python
IN=data/phase4/inputs; LG=data/phase4/long
for inp in $IN/MH_01_easy_A42.zip $IN/MH_01_easy_B42.zip $IN/room2_A42.zip $IN/MH_04_difficult_A42.zip $LG/long_rotor_600s.zip $LG/long_hand_600s.zip; do
  b=$(basename $inp .zip)
  for build in rc; do
    tag=$( [ $build = rc ] && echo rc || echo champ )
    caffeinate -dimsu $PY $build/tools/measure.py $inp --repeats 5 --json $OUT/${tag}_${b}.json > $OUT/${tag}_${b}.log 2>&1
    echo "$tag $b done"
  done
done
