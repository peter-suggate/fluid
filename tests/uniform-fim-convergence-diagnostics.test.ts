import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const extrapolator = readFileSync(
  new URL("../lib/webgpu-uniform-velocity-extrapolation.ts", import.meta.url), "utf8");
const host = readFileSync(new URL("../lib/webgpu-uniform-reference.ts", import.meta.url), "utf8");

test("uniform FIM publishes and checks the source-stated empty active-list termination", () => {
  assert.match(extrapolator, /this\.activeFrontPasses = Math\.max\(\.\.\.dims\)/);
  assert.match(extrapolator, /GPUBufferUsage\.COPY_SRC/);
  assert.match(extrapolator, /get convergenceDiagnostics\(\): GPUBuffer/);
  assert.match(host, /copyBufferToBuffer\(this\.velocityExtrapolator\.convergenceDiagnostics/);
  assert.match(host, /uniformFIMTerminalActiveFaces: words\[24\] \+ words\[25\]/);
  assert.match(host, /uniformFIMConverged: words\[24\] \+ words\[25\] === 0/);
});
