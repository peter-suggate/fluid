#!/bin/zsh
export WEBGPU_NODE_MODULE=/private/tmp/fluid-webgpu/node_modules/webgpu/index.js
cd /Users/petersuggate/code/me/fluid
for i in 1 2 3; do
  FLUID_SCENE=dam-break-ui FLUID_METHOD=tall-cell FLUID_TARGET_S=5 FLUID_REGULAR_LAYERS=24 FLUID_CPU_ORACLE=0 FLUID_FIELD_STATS=1 FLUID_SURFACE_TENSION=0 FLUID_DISABLE_TIMESTAMPS=1 node --import tsx tools/run-webgpu-smoke.ts > /tmp/soak-$i.jsonl 2>&1
  echo "soak$i: $(grep -o '\"nonFiniteCount\":[0-9]*' /tmp/soak-$i.jsonl | head -1) $(grep -o '\"maxSpeed_m_s\":[0-9.e-]*' /tmp/soak-$i.jsonl | tail -1)"
done
echo "=== 2s gate:"
npm run -s test:webgpu:dam-tall-active > /tmp/soak-gate.jsonl 2>&1
echo "gate exit=$?"
