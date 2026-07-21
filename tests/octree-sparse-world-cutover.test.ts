import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { octreeSparseWorldRequired } from "../lib/webgpu-octree";

const source = readFileSync(fileURLToPath(new URL("../lib/webgpu-octree.ts", import.meta.url)), "utf8");

test("body-free direct-page authority omits the compatibility sparse world", () => {
  assert.equal(octreeSparseWorldRequired(true, true, false, 0), false);
  assert.equal(octreeSparseWorldRequired(false, true, false, 0), true, "dense topology still needs bootstrap publication");
  assert.equal(octreeSparseWorldRequired(true, false, false, 0), true, "no page authority keeps compatibility");
  assert.equal(octreeSparseWorldRequired(true, true, true, 0), true, "terrain remains unsupported");
  assert.equal(octreeSparseWorldRequired(true, true, false, 1), true, "body scenes remain unsupported");
  assert.equal(octreeSparseWorldRequired(true, true, false, 0, true), true, "raw/compatibility mode is explicit");
});

test("worldless allocation owns one scheduler and exposes page-native inspection instead of a stale renderer source", () => {
  assert.match(source, /if \(allocateSparseWorld\) this\.sparseBrickWorld = new OctreeSparseBrickWorld/);
  assert.match(source, /this\.topologyResidency = this\.sparseBrickWorld\?\.topologyResidency \?\? new GPUFluidBrickResidency/);
  assert.match(source, /\(this\.sparseBrickWorld\?\.allocatedBytes \?\? this\.topologyResidency\.allocatedBytes\)/);
  assert.match(source, /get sparseVoxelSceneSource\(\) \{ return this\.sparseBrickWorld\?\.sceneSource; \}/);
  assert.match(source, /new CompactOctreeVoxelInspection\([\s\S]*leafHeaders: \{ buffer: this\.leafHeaders \}[\s\S]*rowCount: \{ buffer: this\.compaction \}/,
    "raw inspection follows the compact pressure-grid authority when the compatibility world is absent");
  assert.match(source, /if \(this\.sparseBrickWorld\) this\.sparseBrickWorld\.destroy\(\); else this\.topologyResidency\.destroy\(\);/);
});

test("post-bootstrap topology scheduling comes directly from adaptive candidates", () => {
  assert.match(source, /this\.topologyResidency\.encodeSurfaceCandidates\([\s\S]*source\.leaves, source\.candidates\.candidates, source\.candidates\.countAndDispatch/);
  assert.match(source, /if \(!this\.sparseBrickWorld\) \{[\s\S]*this\.topologyResidency\.encode\(encoder, this\.levelSetTexture/);
});
