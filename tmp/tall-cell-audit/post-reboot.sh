#!/bin/zsh
export WEBGPU_NODE_MODULE=/private/tmp/fluid-webgpu/node_modules/webgpu/index.js
cd /Users/petersuggate/code/me/fluid
echo "=== 1. dam-tall-active (calibrated gates)"
npm run -s test:webgpu:dam-tall-active > tmp/tall-cell-audit/pr-tallactive.jsonl 2>&1
echo "tallactive exit=$?"
echo "=== 2. all scenes, all methods"
FLUID_SCENE=all node --import tsx tools/run-webgpu-smoke.ts > tmp/tall-cell-audit/pr-all.jsonl 2>&1
echo "all exit=$?"
echo "=== 3a. dam-conservation"
npm run -s test:webgpu:dam-conservation > tmp/tall-cell-audit/pr-damcons.jsonl 2>&1
echo "damcons exit=$?"
echo "=== 3b. dam-break-regression (quadtree)"
npm run -s test:webgpu:dam-break-regression > tmp/tall-cell-audit/pr-quadtree.jsonl 2>&1
echo "quadtree exit=$?"
echo "=== 4a. A/B sharpening off (dam 1s)"
FLUID_SCENE=dam-break-ui FLUID_METHOD=tall-cell FLUID_TARGET_S=1 FLUID_REGULAR_LAYERS=24 FLUID_SHARPENING=0 FLUID_STABILITY_ENVELOPE=1 FLUID_CPU_ORACLE=0 FLUID_SURFACE_TENSION=0 FLUID_DISABLE_TIMESTAMPS=1 node --import tsx tools/run-webgpu-smoke.ts > tmp/tall-cell-audit/pr-nosharpen.jsonl 2>&1
echo "nosharpen exit=$?"
echo "=== 4b. A/B hierarchy off (dam 1s)"
FLUID_SCENE=dam-break-ui FLUID_METHOD=tall-cell FLUID_TARGET_S=1 FLUID_REGULAR_LAYERS=24 FLUID_HIERARCHY=0 FLUID_STABILITY_ENVELOPE=1 FLUID_CPU_ORACLE=0 FLUID_SURFACE_TENSION=0 FLUID_DISABLE_TIMESTAMPS=1 node --import tsx tools/run-webgpu-smoke.ts > tmp/tall-cell-audit/pr-nohierarchy.jsonl 2>&1
echo "nohierarchy exit=$?"
echo "=== 4c. baseline both on (dam 1s)"
FLUID_SCENE=dam-break-ui FLUID_METHOD=tall-cell FLUID_TARGET_S=1 FLUID_REGULAR_LAYERS=24 FLUID_STABILITY_ENVELOPE=1 FLUID_CPU_ORACLE=0 FLUID_SURFACE_TENSION=0 FLUID_DISABLE_TIMESTAMPS=1 node --import tsx tools/run-webgpu-smoke.ts > tmp/tall-cell-audit/pr-baseline.jsonl 2>&1
echo "baseline exit=$?"
echo "=== 5. timestamp quirk recheck (2 steps, ts on)"
FLUID_SCENE=dam-break-ui FLUID_METHOD=tall-cell FLUID_TARGET_S=0.008 FLUID_REGULAR_LAYERS=24 FLUID_REPORT_EVERY=1 FLUID_CPU_ORACLE=0 FLUID_FIELD_STATS=0 FLUID_SURFACE_TENSION=0 node --import tsx tools/run-webgpu-smoke.ts > tmp/tall-cell-audit/pr-tson.jsonl 2>&1
echo "tson exit=$?"
echo "=== post-reboot validation complete"
