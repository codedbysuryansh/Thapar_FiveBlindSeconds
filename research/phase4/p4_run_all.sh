#!/bin/sh
# Phase 4: run T1/T2, T7, render, T4, T3, T6 and finally T5 (alone) on the final release/main.py. Logs in logs/phase4/.
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python; L=logs/phase4; mkdir -p $L
md5 release/main.py > $L/p4_main_md5.txt
stamp() { date "+%Y-%m-%d %H:%M:%S"; }
echo "$(stamp) start" > $L/p4_run_all.status
P4_PROCS=6 $PY research/phase4/p4_t2_regression.py > $L/p4_t2_run.out 2>&1; echo "$(stamp) T2 exit $?" >> $L/p4_run_all.status
$PY release/tests/test_edge_cases.py > $L/p4_t7_edge_cases.log 2>&1; echo "$(stamp) T7 exit $?" >> $L/p4_run_all.status
$PY release/tests/test_render.py > $L/p4_render_test.log 2>&1; echo "$(stamp) render exit $?" >> $L/p4_run_all.status
P4_PROCS=10 $PY research/phase4/p4_t4_robustness.py > $L/p4_t4_run.out 2>&1; echo "$(stamp) T4 exit $?" >> $L/p4_run_all.status
P4_PROCS=8 $PY research/phase4/p4_t3_protocol.py > $L/p4_t3_run.out 2>&1; echo "$(stamp) T3 exit $?" >> $L/p4_run_all.status
$PY release/main.py data/phase4/inputs/MH_01_easy_B42.zip /tmp/fbs_t6_ref_MH01B.txt > /dev/null 2>&1
$PY research/phase4/p4_t6_cold_start.py data/phase4/inputs/MH_01_easy_B42.zip /tmp/fbs_t6_ref_MH01B.txt \
  "mac-cp313-latest=/opt/homebrew/bin/python3.13=-r" "mac-cp310-np1.22.4=/opt/homebrew/bin/python3.10=numpy==1.22.4" "mac-cp310-np1.26.4=/opt/homebrew/bin/python3.10=numpy==1.26.4" \
  --json research/phase4/results/p4_t6_cold_start_mac.json > $L/p4_t6_cold_start_mac.log 2>&1; echo "$(stamp) T6 exit $?" >> $L/p4_run_all.status
sleep 20
P4_REPEATS=5 $PY research/phase4/p4_t5_measure.py > $L/p4_t5_measure_mac.log 2>&1; echo "$(stamp) T5 exit $?" >> $L/p4_run_all.status
echo "$(stamp) done" >> $L/p4_run_all.status
