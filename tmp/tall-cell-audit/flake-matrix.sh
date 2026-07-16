#!/bin/zsh
export WEBGPU_NODE_MODULE=/private/tmp/fluid-webgpu/node_modules/webgpu/index.js
cd /Users/petersuggate/code/me/fluid
run() {
  local label=$1; shift
  for i in 1 2; do
    env "$@" FLUID_SCENE=dam-break-ui FLUID_METHOD=tall-cell FLUID_TARGET_S=5 FLUID_REGULAR_LAYERS=24 FLUID_CPU_ORACLE=0 FLUID_FIELD_STATS=1 FLUID_SURFACE_TENSION=0 FLUID_DISABLE_TIMESTAMPS=1 node --import tsx tools/run-webgpu-smoke.ts > /tmp/flake-$label-$i.jsonl 2>&1
    local ok=$(grep -c '"nonFiniteCount":0' /tmp/flake-$label-$i.jsonl 2>/dev/null | head -1)
    local nf=$(grep -o '"nonFiniteCount":[0-9]*' /tmp/flake-$label-$i.jsonl | head -1)
    echo "$label run$i: exit-relevant $nf"
  done
}
run baseline
run nosharpen FLUID_SHARPENING=0
run nohier FLUID_HIERARCHY=0
run dt2 FLUID_MAX_DT=0.002
echo "matrix done"
