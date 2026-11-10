#!/bin/bash
# occ.sh <wgsl> <entry> <constants.json> -> maxTotalThreadsPerThreadgroup of the uncapped Metal pipeline.
# Compile only: Dawn pipeline creation dumps the MSL, the 64-thread attribute is stripped,
# and Metal's pipeline state reports its register-limited thread cap (no dispatch).
# Run under tools/gpu/fluid-gpu-run.sh check. Uses lane T's mslprobe (Swift, Metal API).
W=$1; E=$2; C=$3; S=/Users/petersuggate/code/me/fluid-scratch/lane-n; T=$(mktemp -d $S/occ.XXXX)
cd $S/tree
timeout 600 node --import tsx /Users/petersuggate/code/me/fluid/tools/experiments/lane-n/dump-msl.mts $W $E $C x > $T/log 2>&1
grep -q "^ERR" $T/log && { echo "COMPILE-ERR $(grep '^ERR' $T/log | head -2)"; rm -rf $T; exit 1; }
s=$(grep -n "Dumped generated MSL" $T/log | tail -1 | cut -d: -f1); e=$(grep -n "^pipeline ms" $T/log | cut -d: -f1)
[ -z "$s" ] && { echo "NO-MSL $(tail -3 $T/log)"; rm -rf $T; exit 1; }
sed -n "$((s+1)),$((e-1))p" $T/log | sed 's/^\[\[max_total_threads_per_threadgroup([0-9]*)\]\]//' > $T/k.metal
[ -n "${KEEP:-}" ] && cp $T/k.metal $KEEP
/Users/petersuggate/code/me/fluid-scratch/lane-t/tools/mslprobe/mslprobe $T/k.metal | grep -o 'maxTotalThreadsPerThreadgroup [0-9]*\|ERROR.*' | head -1
rm -rf $T
