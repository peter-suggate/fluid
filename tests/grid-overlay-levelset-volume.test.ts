import assert from "node:assert/strict";
import test from "node:test";
import {
  GRID_OVERLAY_LSV_UNIFORM_WORDS,
  gridOverlayLevelSetVolumeUniform,
  gridOverlayLevelSetVolumeWGSL,
} from "../lib/core/grid-overlay-levelset-volume.wgsl";
import type { SparseLevelSetVolumeConsumerLayout } from "../lib/core/levelset-consumer-abi";

const layout: SparseLevelSetVolumeConsumerLayout = {
  globalHeaderBaseWords: 101,
  slot0BaseWords: 201,
  slotStrideWords: 301,
  slotHeaderOffsetWords: 1,
  cornerRefsOffsetWords: 2,
  cellRecordsOffsetWords: 3,
  cellHashOffsetWords: 4,
  phi0OffsetWords: 5,
  phi1OffsetWords: 6,
  support0OffsetWords: 7,
  support1OffsetWords: 8,
  cellCapacity: 401,
  vertexCapacity: 501,
  cellHashCapacity: 1024,
  hashProbeLimit: 32,
  solidCellOpenOffsetFloats: 701,
  solidVoxelCellOpenOffsetFloats: 801,
};

test("slice LSV uniform has a fixed fail-closed ABI", () => {
  assert.equal(GRID_OVERLAY_LSV_UNIFORM_WORDS, 24);
  assert.deepEqual([...gridOverlayLevelSetVolumeUniform(undefined)], Array(24).fill(0));
  assert.deepEqual([...gridOverlayLevelSetVolumeUniform(layout)], [
    1, 101, 201, 301,
    1, 2, 3, 4,
    5, 6, 7, 8,
    401, 501, 1023, 32,
    0, 701, 801, 3,
    0, 0, 0, 0,
  ]);
});

test("slice LSV shader follows the accepted publication and direct corner references", () => {
  assert.match(gridOverlayLevelSetVolumeWGSL,
    /sliceLsvHeader\(slot,2u\)==sliceLsvGlobal\(4u\)/);
  assert.match(gridOverlayLevelSetVolumeWGSL,
    /let vertex=sparseTopologyArena\[sliceLsvSlotBase\(slot\)\+sliceLsvP\.offsets0\.y\s*\+8u\*ordinal\+corner\]/);
  assert.match(gridOverlayLevelSetVolumeWGSL,
    /fn sliceLsvMix\(hash:u32,value:u32\).*0x9e3779b1u/s);
  assert.doesNotMatch(gridOverlayLevelSetVolumeWGSL, /\bisFinite\(/);
  assert.match(gridOverlayLevelSetVolumeWGSL,
    /let density=sparseState\[sparseDensityOffset\(\)\+owner\.x\]/);
});
