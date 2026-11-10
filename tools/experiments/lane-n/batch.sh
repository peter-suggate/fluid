#!/bin/bash
# batch.sh <base.wgsl> <entry> <constants.json> <list> : each list line "name|edit edit ..." -> name, threads.
# Serial compile-only probes; run the whole batch inside one `tools/gpu/fluid-gpu-run.sh check`.
B=$1; E=$2; C=$3; L=$4; D=/Users/petersuggate/code/me/fluid/tools/experiments/lane-n; V=/Users/petersuggate/code/me/fluid-scratch/lane-n/variants
mkdir -p $V
while IFS='|' read -r name edits; do
  [ -z "$name" ] && continue; case "$name" in \#*) continue;; esac
  eval "set -- $edits"
  python3 $D/variant.py $B $V/$name.wgsl "$@" || { echo "$name GEN-ERR"; continue; }
  printf '%-34s %s\n' "$name" "$($D/occ.sh $V/$name.wgsl $E $C 2>&1 | tail -1)"
done < $L
