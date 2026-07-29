import assert from "node:assert/strict";
import test from "node:test";
import {
  OCTREE_PERSISTENT_MGPCG_ROW_THRESHOLD,
  WebGPUOctreePersistentMGPCG,
} from "../lib/webgpu-octree-persistent-mgpcg";
import {
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
});

test("persistent MGPCG retains the capacity-strided arena as an A/B oracle", () => {
  const legacy = octreePersistentMGPCGWGSL({
    maximumIterations: 10,
    compactLiveRows: false,
  });
  assert.match(legacy, /fn rowStride\(\)->u32\{return capacity\(\);\}/);
});

test("persistent MGPCG admits only the measured-beneficial constructed capacity", () => {
  assert.equal(OCTREE_PERSISTENT_MGPCG_ROW_THRESHOLD, 4_096);
  assert.equal(WebGPUOctreePersistentMGPCG.selects(4_096), true);
  assert.equal(WebGPUOctreePersistentMGPCG.selects(4_097), false);
  assert.equal(WebGPUOctreePersistentMGPCG.selects(9_216), false,
    "ceiling-drop must use the measured-faster row-parallel solver");
});
