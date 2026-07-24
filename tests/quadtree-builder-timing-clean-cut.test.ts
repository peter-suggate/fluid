import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL("../lib/webgpu-quadtree-builder.ts", import.meta.url), "utf8");

test("quadtree builder contains no legacy timing or timestamp instrumentation", () => {
  assert.doesNotMatch(source,
    /finalTimestampWrites|timestampWrites|timestamp-query|createQuerySet|resolveQuerySet|queryResolve|gpuKernel_ms|gpuWall_ms|performance\.now/);
});

test("quadtree builder preserves functional topology, diagnostic, profile, and surface readbacks", () => {
  assert.match(source, /copyBufferToBuffer\(finalTopology,\s*0,\s*readback,\s*topologyOffset,\s*topologyBytes\)/);
  assert.match(source, /copyBufferToBuffer\(inputs\.diagnosticBuffer,\s*0,\s*readback,\s*diagnosticOffset,\s*inputs\.diagnosticBytes\)/);
  assert.match(source, /copyBufferToBuffer\(columnProfiles,\s*0,\s*profileReadback,\s*0,\s*profileBytes\)/);
  assert.match(source, /copyBufferToBuffer\(this\.reductions,\s*0,\s*readback,\s*0,\s*16\)/);
  assert.match(source, /await readback\.mapAsync\(GPUMapMode\.READ\)/);
  assert.match(source, /await profileReadback\.mapAsync\(GPUMapMode\.READ,\s*0,\s*profileBytes\)/);
});
