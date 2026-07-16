#!/bin/zsh
export WEBGPU_NODE_MODULE=/private/tmp/fluid-webgpu/node_modules/webgpu/index.js
cd /Users/petersuggate/code/me/fluid
echo "=== 1. dam-tall-active"
npm run -s test:webgpu:dam-tall-active > tmp/tall-cell-audit/pf-tallactive.jsonl 2>&1
echo "tallactive exit=$?"
echo "=== 2. all scenes"
FLUID_SCENE=all node --import tsx tools/run-webgpu-smoke.ts > tmp/tall-cell-audit/pf-all.jsonl 2>&1
echo "all exit=$?"
echo "=== 3a. dam-conservation"
npm run -s test:webgpu:dam-conservation > tmp/tall-cell-audit/pf-damcons.jsonl 2>&1
echo "damcons exit=$?"
echo "=== 3b. dam-break-regression"
npm run -s test:webgpu:dam-break-regression > tmp/tall-cell-audit/pf-quadtree.jsonl 2>&1
echo "quadtree exit=$?"
echo "=== complete"
