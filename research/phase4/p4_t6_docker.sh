#!/bin/sh
# Phase 4 test T6 in a real container: build release/Dockerfile (python:3.12-slim + pip install -r requirements.txt), run the
# README command on zipped official-format inputs, check the output contract (tools/check_output.py) and compare the bytes
# with the development build's output for the same input (data/phase4/out_t2). Needs Docker (here: colima + docker CLI).
cd "$(dirname "$0")/../.." || exit 1
L=logs/phase4/p4_t6_docker.log; W=data/phase4/t6_docker; mkdir -p $W; : > $L
say() { echo "$*" | tee -a $L; }
say "T6 (container) $(date '+%Y-%m-%d %H:%M:%S'); docker $(docker version --format '{{.Server.Version}} {{.Server.Os}}/{{.Server.Arch}}')"
docker build --no-cache -t fiveblindseconds:1.0 release > $W/build.log 2>&1 || { say "docker build FAILED (see $W/build.log)"; exit 1; }
say "image $(docker images -q fiveblindseconds:1.0)  base $(grep '^FROM' release/Dockerfile)"
say "inside: $(docker run --rm --entrypoint python fiveblindseconds:1.0 -c 'import sys, platform, numpy; print(platform.platform(), "Python", sys.version.split()[0], "NumPy", numpy.__version__)')"
say "main.py in the image: $(docker run --rm --entrypoint md5sum fiveblindseconds:1.0 /app/main.py | cut -d' ' -f1)   on disk: $(md5 -q release/main.py)"
ok=0; n=0
for tag in MH_01_easy_B42 room2_A42 V2_03_difficult_A42 room5_B42; do
  n=$((n+1)); cp data/phase4/inputs/$tag.zip $W/; rm -f $W/out_$tag.txt
  t0=$(python3 -c 'import time; print(time.time())')
  docker run --rm -v "$PWD/$W":/data fiveblindseconds:1.0 $tag.zip out_$tag.txt > $W/run_$tag.log 2>&1; rc=$?
  t1=$(python3 -c 'import time; print(time.time())')
  c=$(python3 release/tools/check_output.py data/phase4/inputs/$tag.zip $W/out_$tag.txt > $W/check_$tag.log 2>&1 && echo PASS || echo FAIL)
  same=$(cmp -s $W/out_$tag.txt data/phase4/out_t2/$tag.txt && echo identical || echo DIFFERENT)
  [ $rc -eq 0 ] && [ "$c" = PASS ] && [ "$same" = identical ] && ok=$((ok+1))
  say "$(printf '%-22s exit %s  T1 %s  output %s to the development build  %.1f s (container start included)  %s' $tag $rc $c $same $(python3 -c "print($t1-$t0)") "$(tail -2 $W/run_$tag.log | head -1)")"
done
say "T6 (container): $ok/$n pass -> $([ $ok -eq $n ] && echo PASS || echo FAIL)"
