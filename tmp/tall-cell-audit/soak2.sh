#!/bin/zsh
export WEBGPU_NODE_MODULE=/private/tmp/fluid-webgpu/node_modules/webgpu/index.js
cd /Users/petersuggate/code/me/fluid
for i in 1 2 3; do
  npm run -s test:webgpu:dam-tall-soak > /tmp/soak2-$i.jsonl 2>&1
  echo "soak$i exit=$? $(grep -o '\"nonFiniteCount\":[0-9]*' /tmp/soak2-$i.jsonl | head -1) peakCFL=$(grep -o '\"peakComponentCfl\":[0-9.]*' /tmp/soak2-$i.jsonl | tail -1)"
done
