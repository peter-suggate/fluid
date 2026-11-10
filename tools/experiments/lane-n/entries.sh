#!/bin/bash
# entries.sh <wgsldir> <list of "module entry"> [variant edits...]: baseline (or edited) cap of each entry.
W=$1; L=$2; shift 2; D=/Users/petersuggate/code/me/fluid/tools/experiments/lane-n; V=/Users/petersuggate/code/me/fluid-scratch/lane-n/variants; mkdir -p $V
while read -r m e; do
  [ -z "$m" ] && continue; f=$W/$(printf '%03d' $m).wgsl
  if [ $# -gt 0 ]; then python3 $D/variant.py $f $V/e-$m-$e.wgsl "$@" || { echo "$m $e GEN-ERR"; continue; }; f=$V/e-$m-$e.wgsl; fi
  printf '%-4s %-26s %s\n' $m $e "$($D/occ.sh $f $e $W/c-$m-$e.json 2>&1 | tail -1)"
done < $L
