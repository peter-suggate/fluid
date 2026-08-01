import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { planOctreePressureCapacity } from "../lib/webgpu-octree";
import {
  OCTREE_PERSISTENT_MGPCG_ROW_THRESHOLD,
  WebGPUOctreePersistentMGPCG,
} from "../lib/webgpu-octree-persistent-mgpcg";
import {
  OCTREE_PERSISTENT_MGPCG_CHANNEL_COUNT,
  OCTREE_PERSISTENT_MGPCG_HEADER,
  OCTREE_PERSISTENT_MGPCG_PARTIAL_WORDS,
  OCTREE_PERSISTENT_MGPCG_REDUCTION_LANES,
  octreePersistentMGPCGArenaWords,
  octreePersistentMGPCGWGSL,
} from "../lib/webgpu-octree-persistent-mgpcg.wgsl";

test("persistent MGPCG packs hot channels by live rows after stable input staging", () => {
  const compact = octreePersistentMGPCGWGSL({
    maximumIterations: 10,
    compactLiveRows: true,
  });
  assert.match(compact, /fn rowStride\(\)->u32\{return wRows;\}/);
  assert.match(compact,
    /let seed=stagedVload\(CH_SEED,row\);\s*let rhs=stagedVload\(CH_RHS,row\);/);
  assert.match(compact,
    /vstore\(CH_RHS,row,rhs\);\s*vstore\(CH_SEED,row,seed\);/);
  assert.match(compact,
    /fn partialBase\(\)->u32\{return ARENA_HEADER\+CHANNELS\*rowStride\(\);\}/);
  assert.match(compact,
    /fn stagedCh\(c:u32,r:u32\)->u32\{return p\.sizes\.w\+\(c-CH_RHS\)\*capacity\(\)\+r;\}/);
  assert.match(compact,
    /else if\(acc\(2u\)>MAX_LIVE_ROWS\)\{reportAt\(ERR_ROW,1u,acc\(2u\)\);\}/);
});

test("persistent MGPCG keeps staged inputs disjoint from compact live channels", () => {
  const capacity = 148_480;
  const partials = Math.ceil(capacity / OCTREE_PERSISTENT_MGPCG_REDUCTION_LANES);
  assert.equal(octreePersistentMGPCGArenaWords(capacity),
    OCTREE_PERSISTENT_MGPCG_HEADER.totalWords
      + OCTREE_PERSISTENT_MGPCG_CHANNEL_COUNT * capacity
      + OCTREE_PERSISTENT_MGPCG_PARTIAL_WORDS * partials
      + 2 * capacity);
});

test("persistent MGPCG retains the capacity-strided arena as an A/B oracle", () => {
  const legacy = octreePersistentMGPCGWGSL({
    maximumIterations: 10,
    compactLiveRows: false,
  });
  assert.match(legacy, /fn rowStride\(\)->u32\{return capacity\(\);\}/);
});

test("persistent MGPCG selects exact live rows, not provisioned capacity", () => {
  assert.equal(OCTREE_PERSISTENT_MGPCG_ROW_THRESHOLD, 65_536);
  assert.equal(WebGPUOctreePersistentMGPCG.acceptsLiveRows(4_096), true);
  assert.equal(WebGPUOctreePersistentMGPCG.acceptsLiveRows(9_216), true);
  assert.equal(WebGPUOctreePersistentMGPCG.acceptsLiveRows(65_537), false);
});

test("large provisioned pressure headroom does not reject adaptive construction", () => {
  const capacity = planOctreePressureCapacity(
    { nx: 96, ny: 64, nz: 96 }, 32, 4, undefined, false, 0.22,
  ).rowCapacity;
  assert.ok(capacity > OCTREE_PERSISTENT_MGPCG_ROW_THRESHOLD,
    "fixture must distinguish conservative capacity from exact adaptive rows");
  const projectionSource = readFileSync(
    new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8",
  );
  assert.doesNotMatch(projectionSource,
    /acceptsLiveRows\(this\.pressureCapacity\.rowCapacity\)/,
    "provisioned capacity exists before the GPU publishes adaptive rows");
});
