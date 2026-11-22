#!/bin/zsh
# Promote RC-1.1.0c to champion v1.1.0: registry entry, then an immutable snapshot champion/v1.1.0/ (release folder, the
# regression logs of this exact main.py, the evidence), SHA256SUMS, read-only + user-immutable flag. v1.0.0 is not touched.
set -e
ROOT=~/SamsungPRISM_FiveBlindSeconds; O=$ROOT/opt; C=$ROOT/champion/v1.1.0; R=$O/regression; N=$O/notes
[ -e $C ] && { echo "champion/v1.1.0 already exists - refusing to overwrite"; exit 1; }
[ "$(shasum -a 256 $ROOT/rc/main.py | cut -d' ' -f1)" = c401d784ddc78e91ea6f76f394d5a93588c1d05e95c20a5482ab01cecac944e2 ] || { echo "rc/main.py changed"; exit 1; }
cd $O && ../.venv/bin/python harness/fbsopt.py decide RC-1.1.0c --status promoted --reason "Champion v1.1.0. = X110d byte for byte (DEV 136/136) and = X108b on S42/DEV2/CONF/STRESS (850/850); CONF pooled -11.21 % [-17.41, -3.52], STRESS -9.10 %; T3 official protocol EuRoC ATE -14.1 % [-21.2, -4.8], TUM-VI -7.3 % [-13.8, +1.5], no metric significantly worse; T1 34/34 (Mac, x86-64, identical); T4 935 + 799 held-out: 0 errors, 0 fallbacks, all family means better (rule 7b); T5 own heap 2.64 / 6.86 MB, Xeon worst 11.7 ms per IMU-second (+25 % on the 10-minute rotor input, justified by the accuracy gain); T6 container + 3 fresh macOS envs + x86-64 identical; T7 37/37; renderer 0 mismatches" > /dev/null
mkdir -p $C/release $C/logs/t5_m3 $C/logs/t5_xeon $C/evidence
rsync -a --exclude __pycache__ --exclude '*.pyc' --exclude .DS_Store $ROOT/rc/ $C/release/
cp $R/p5_t1_rcc.log $R/p5_t6_x86_rcc.log $R/p5_t3_protocol_rcc.log $R/p5_t3_protocol_rcc.json $R/p5_t4_rcc.log $R/p5_t4_rcc.json $R/p5_t4_rcc_verdict.txt \
   $R/t4fresh_rcc.txt $R/p5_t6_docker_arm64_c.log $R/p5_t6_mac_rcc.log $R/p5_t6_mac_rcc.json $R/p5_t7_rc_c.log $R/p5_render_rcc.log $C/logs/
cp $R/t1_rcc/SHA256_mac.txt $C/logs/p5_t1_rcc_SHA256_mac.txt
cp $N/measure_c/rc_*.json $N/measure_c/rc_*.log $C/logs/t5_m3/; cp $N/measure_xeon_rcc/*.json $C/logs/t5_xeon/
cp $O/README.md $C/evidence/PROGRAMME_RULES.md; cp $O/registry.json $O/REGISTRY.md $C/evidence/
cp $N/an10_t3_paired_v100_vs_v110.txt $N/an10_t3_paired_v100_vs_v110.json $N/an10b_t3_per_sequence_v100_vs_v110.txt $N/an15_t4_cases_rcb.txt \
   $N/an23_t4_compare_K.txt $N/an23_t4_compare_L.txt $N/an23_t4fresh_rcc.txt $C/evidence/
for f in $O/candidates/RC-1.1.0c/compare_*.txt; do cp $f $C/evidence/RC-1.1.0c_$(basename $f); done
cp $N/CHAMPION_v1.1.0.md $C/CHAMPION.md
cd $C && find . -type f ! -name SHA256SUMS | sort | xargs shasum -a 256 > SHA256SUMS
chmod -R a-w $C && chflags -R uchg $C
shasum -a 256 -c SHA256SUMS | grep -vc ": OK$" || true
echo "champion v1.1.0: $(wc -l < SHA256SUMS | tr -d ' ') files, read-only, immutable"
