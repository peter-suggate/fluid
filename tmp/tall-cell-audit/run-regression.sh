#!/bin/zsh
export WEBGPU_NODE_MODULE=/private/tmp/fluid-webgpu/node_modules/webgpu/index.js
cd /Users/petersuggate/code/me/fluid
echo "=== test:webgpu (all scenes, all methods)"
FLUID_SCENE=all node --import tsx tools/run-webgpu-smoke.ts > tmp/tall-cell-audit/regress-all.jsonl 2>&1
echo "all exit=$?"
echo "=== test:webgpu:dam-conservation"
npm run -s test:webgpu:dam-conservation > tmp/tall-cell-audit/regress-damcons.jsonl 2>&1
echo "damcons exit=$?"
echo "=== test:webgpu:dam-break-regression (quadtree)"
npm run -s test:webgpu:dam-break-regression > tmp/tall-cell-audit/regress-quadtree.jsonl 2>&1
echo "quadtree exit=$?"
echo "=== test:webgpu:dam-tall-active (new)"
npm run -s test:webgpu:dam-tall-active > tmp/tall-cell-audit/regress-tallactive.jsonl 2>&1
echo "tallactive exit=$?"
echo "=== regression complete"
