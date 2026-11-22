#!/bin/sh
# Crash-rate check of release/main.py under old NumPy wheels on this Mac (fresh venv per version, N runs each).
cd "$(dirname "$0")/../../.." || exit 1
N=${N:-12}; PYB=${PYB:-/opt/homebrew/bin/python3.10}; OUT=logs/phase4/p4_np_versions_mac.log
: > $OUT
for v in "$@"; do
  D=$(mktemp -d /tmp/fbs_npv.XXXX); $PYB -m venv $D/venv && $D/venv/bin/python -m pip install -q --disable-pip-version-check "numpy==$v" > /dev/null 2>&1
  cp release/main.py data/phase4/inputs/MH_01_easy_B42.zip $D/; ok=0; bad=""; same=0
  for i in $(seq 1 $N); do
    (cd $D && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 venv/bin/python main.py MH_01_easy_B42.zip est.txt > log.txt 2>&1); rc=$?
    if [ $rc -eq 0 ]; then ok=$((ok+1)); cmp -s $D/est.txt /tmp/fbs_t6_ref_MH01B.txt && same=$((same+1)); else bad="$bad $rc"; fi
    rm -f $D/est.txt
  done
  echo "numpy $v ($($D/venv/bin/python -c 'import sys; print(sys.version.split()[0])')): $ok/$N runs ok, $same identical to the reference; failures (exit codes):$bad" | tee -a $OUT
done
