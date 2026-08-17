#!/usr/bin/env node
/** Assert and report the standalone FPL1 B16-native scheduling ABI. */
import {
  SPARSE_CM12_FRAME_PLAN_BRICK_WORDS,
  SPARSE_CM12_FRAME_PLAN_HEADER,
  SPARSE_CM12_FRAME_PLAN_HEADER_WORDS,
  SPARSE_CM12_FRAME_PLAN_INDIRECT_WORDS,
  SPARSE_CM12_FRAME_PLAN_SLOT_HEADER_WORDS,
  SPARSE_CM12_FRAME_PLAN_STAGE_COUNT,
  SPARSE_CM12_FRAME_PLAN_TILE_WORDS,
  createSparseCM12FramePlanInitialWords,
  createSparseCM12FramePlanLayout,
  sparseCM12FramePlanValidTileMask,
} from "../lib/core/sparse-cm12-frame-plan";

const fail = (condition: boolean, message: string): void => {
  if (!condition) throw new Error(message);
};

const brickCapacity = 257;
const variants = ([4, 8, 16] as const).map((brickFineResolution) => {
  const layout = createSparseCM12FramePlanLayout({ brickCapacity,
    brickFineResolution, packetCount: SPARSE_CM12_FRAME_PLAN_STAGE_COUNT });
  const initial = createSparseCM12FramePlanInitialWords(layout);
  fail(SPARSE_CM12_FRAME_PLAN_TILE_WORDS * 4 <= 16,
    "FPL1 hot tile record exceeds 16 bytes");
  fail(layout.slot0BaseWords + layout.slotWords === layout.slot1BaseWords,
    "FPL1 Current/Next slots overlap or have a gap");
  fail(layout.slot1BaseWords + layout.slotWords === layout.totalWords,
    "FPL1 second slot does not end at totalWords");
  fail(initial.length === layout.totalWords - layout.baseWords,
    "FPL1 initializer does not cover the configured arena");
  fail(initial[SPARSE_CM12_FRAME_PLAN_HEADER.headerWords]
    === SPARSE_CM12_FRAME_PLAN_HEADER_WORDS, "FPL1 header size mismatch");
  fail(initial[SPARSE_CM12_FRAME_PLAN_HEADER.acceptedFrameGeneration] === 0
    && initial[SPARSE_CM12_FRAME_PLAN_HEADER.acceptedParity] === 0,
  "FPL1 frame-control authority does not start at generation/parity zero");
  const [validMaskLow, validMaskHigh] = sparseCM12FramePlanValidTileMask(
    brickFineResolution,
  );
  const bytesPerBrickPerSlot = 4 * (SPARSE_CM12_FRAME_PLAN_BRICK_WORDS
    + SPARSE_CM12_FRAME_PLAN_TILE_WORDS * layout.tilesPerBrick
    + layout.packetCount);
  return {
    brickFineResolution,
    tilesPerAxis: layout.tilesPerAxis,
    tilesPerBrick: layout.tilesPerBrick,
    validMask: [validMaskLow.toString(16).padStart(8, "0"),
      validMaskHigh.toString(16).padStart(8, "0")],
    hotTileRecordBytes: 4 * SPARSE_CM12_FRAME_PLAN_TILE_WORDS,
    bytesPerBrickPerSlot,
    slotBytes: 4 * layout.slotWords,
    totalBytes: layout.totalBytes,
  };
});

const receipt = {
  abi: "FPL1/v1",
  native: "B16/P16",
  stages: SPARSE_CM12_FRAME_PLAN_STAGE_COUNT,
  headerBytes: 4 * SPARSE_CM12_FRAME_PLAN_HEADER_WORDS,
  slotHeaderBytes: 4 * SPARSE_CM12_FRAME_PLAN_SLOT_HEADER_WORDS,
  maximumIndirectSnapshotBytes: 4 * SPARSE_CM12_FRAME_PLAN_INDIRECT_WORDS,
  variants,
  ownership: {
    dispatch: "one workgroup per physical brick",
    tile: "lane owns one 4^3 tile; B4 uses lane 0, B8 lanes 0..7, B16 lanes 0..63",
    lists: "deterministic brick-indexed packet slots; no per-tile append journal",
    closure: "bounded cross-brick mask OR, then fixed owner-lane resolution",
  },
  generations: {
    slots: "double-buffered Current/Next",
    frameControl: "accepted frame generation and parity are GPU-authored on commit",
    fault: "global ABI/transition faults retain Current; local brick faults publish magenta and omit only that brick",
  },
  overlay: {
    authority: "same 16-byte tile and brick packet records that schedule physics",
    modes: ["logical stage mask", "physical packet", "cause", "closure depth",
      "generation", "executed/skipped", "local fault"],
  },
};

console.log(JSON.stringify(receipt, null, 2));
