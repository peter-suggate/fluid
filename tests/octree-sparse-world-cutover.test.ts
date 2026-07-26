import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { octreeSparseWorldRequired } from "../lib/webgpu-octree";

const source = readFileSync(fileURLToPath(new URL("../lib/webgpu-octree.ts", import.meta.url)), "utf8");
const sparseWorldSource = readFileSync(
  fileURLToPath(new URL("../lib/webgpu-octree-sparse-bricks.ts", import.meta.url)),
  "utf8",
);
const staticSvoSource = readFileSync(
  fileURLToPath(new URL("../lib/webgpu-static-svo-scene.ts", import.meta.url)),
  "utf8",
);

test("body-free direct-page authority omits the compatibility sparse world", () => {
  assert.equal(octreeSparseWorldRequired(false, 0), false);
  assert.equal(octreeSparseWorldRequired(true, 0), true, "terrain requires sparse solid/world support");
  assert.equal(octreeSparseWorldRequired(false, 1), true, "body scenes require sparse solid/world support");
});

test("worldless allocation owns one scheduler and exposes compact inspection instead of a stale renderer source", () => {
  assert.match(source, /if \(allocateSparseWorld\) this\.sparseBrickWorld = new OctreeSparseBrickWorld/);
  assert.match(source, /this\.topologyResidency = this\.sparseBrickWorld\?\.topologyResidency \?\? new GPUFluidBrickResidency/);
  assert.match(source, /\(this\.sparseBrickWorld\?\.allocatedBytes \?\? this\.topologyResidency\.allocatedBytes\)/);
  assert.match(source, /get sparseVoxelSceneSource\(\) \{ return this\.sparseBrickWorld\?\.sceneSource; \}/);
  assert.match(source, /new CompactOctreeVoxelInspection\([\s\S]*leafHeaders: \{ buffer: this\.leafHeaders \}[\s\S]*rowCount: \{ buffer: this\.compaction \}/,
    "raw inspection follows the compact pressure-grid authority when the compatibility world is absent");
  assert.match(source, /if \(this\.sparseBrickWorld\) this\.sparseBrickWorld\.destroy\(\); else this\.topologyResidency\.destroy\(\);/);
});

test("post-bootstrap topology scheduling comes only from adaptive candidates", () => {
  assert.match(source, /this\.topologyResidency\.encodeFineSeedCandidates\([\s\S]*source\.leaves, source\.candidates\.candidates, source\.candidates\.countAndDispatch/);
  const recurring = source.slice(source.indexOf("encodeSparseBrickWorld"), source.indexOf("destroy()", source.indexOf("encodeSparseBrickWorld")));
  assert.doesNotMatch(recurring, /topologyResidency\.encode\(encoder/,
    "the dense bootstrap classifier must not survive as a recurring fallback");
});

test("deep-liquid residency is explicit and the retired dense atlas cannot be selected", () => {
  assert.match(source, /bulkResidency: true/);
  assert.match(sparseWorldSource, /bulkResidency\?: boolean/);
  assert.match(sparseWorldSource, /if \(options\.bulkResidency\)/);
  assert.doesNotMatch(source + sparseWorldSource + staticSvoSource,
    /brickAtlas|bulkResidencyOnly|WebGPUFluidBrickAtlas|atlasSamplingSource|readAtlasStats/);
  assert.match(staticSvoSource, /brickPreActivation: false/,
    "static SVO publication retains its non-advecting residency behavior");
});
