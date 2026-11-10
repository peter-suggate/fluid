#!/bin/bash
# entries-idx.sh <wgsldir> <list of "descriptor module entry tag"> [variant edits...]: cap of each pipeline descriptor.
W=$1; L=$2; shift 2; D=/Users/petersuggate/code/me/fluid/tools/experiments/lane-n; V=/Users/petersuggate/code/me/fluid-scratch/lane-n/variants; mkdir -p $V
while read -r i m e tag; do
  [ -z "$i" ] && continue; f=$W/$(printf '%03d' $m).wgsl
  if [ $# -gt 0 ]; then python3 $D/variant.py $f $V/x-$m.wgsl "$@" || { echo "$i GEN-ERR"; continue; }; f=$V/x-$m.wgsl; fi
  printf '%-3s %-20s %-46s %s\n' $i $e "$tag" "$($D/occ.sh $f $e $W/c-idx-$i.json 2>&1 | tail -1 | sed 's/maxTotalThreadsPerThreadgroup //')"
done < $L
