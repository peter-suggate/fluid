#!/bin/zsh
export WEBGPU_NODE_MODULE=/private/tmp/fluid-webgpu/node_modules/webgpu/index.js
export FLUID_SCENE=dam-break-ui FLUID_CPU_ORACLE=0 FLUID_REPORT_EVERY=1 FLUID_FIELD_STATS=1
export FLUID_STABILITY_ENVELOPE=1 FLUID_CHECKPOINT_EVERY_S=0.1 FLUID_DISABLE_TIMESTAMPS=1
export FLUID_SURFACE_TENSION=0 FLUID_TARGET_S=2
cd /Users/petersuggate/code/me/fluid
echo "=== run d24 (failing config + uniform baseline)"
FLUID_METHOD=tall-cell,uniform FLUID_REGULAR_LAYERS=24 node --import tsx tools/run-webgpu-smoke.ts > tmp/tall-cell-audit/d24.jsonl 2>&1
echo "d24 exit=$?"
echo "=== run c44 (restricted kernels, minimal tall + uniform baseline)"
FLUID_METHOD=tall-cell,uniform FLUID_REGULAR_LAYERS=44 node --import tsx tools/run-webgpu-smoke.ts > tmp/tall-cell-audit/c44.jsonl 2>&1
echo "c44 exit=$?"
echo "=== run b46 (fallback confirmation)"
FLUID_METHOD=tall-cell FLUID_REGULAR_LAYERS=46 FLUID_TARGET_S=0.1 node --import tsx tools/run-webgpu-smoke.ts > tmp/tall-cell-audit/b46.jsonl 2>&1
echo "b46 exit=$?"
echo "=== run e24 (semi-lagrangian variant)"
FLUID_METHOD=tall-cell FLUID_REGULAR_LAYERS=24 FLUID_VELOCITY_TRANSPORT=semi-lagrangian node --import tsx tools/run-webgpu-smoke.ts > tmp/tall-cell-audit/e24-sl.jsonl 2>&1
echo "e24 exit=$?"
echo "=== matrix complete"
