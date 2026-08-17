#!/usr/bin/env node
/** Assert and report the standalone FPP1 packet/execution ABI. */
import {
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER_WORDS,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_WORDS,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_SAMPLES_PER_TILE,
  createSparseCM12FramePlanPresentationInitialWords,
  createSparseCM12FramePlanPresentationLayout,
} from "../lib/methods/adaptive-mass/sparse-cm12-frame-plan-presentation";

const assert = (condition: boolean, message: string): void => {
  if (!condition) throw new Error(message);
};
const capacity = 257;
const variants = ([4, 8, 16] as const).map((brickFineResolution) => {
  const layout = createSparseCM12FramePlanPresentationLayout({
    pageCapacity: capacity, brickFineResolution,
  });
  const initial = createSparseCM12FramePlanPresentationInitialWords(layout);
  assert(initial.length === layout.totalWords - layout.baseWords,
    "FPP1 initializer/layout length mismatch");
  assert(initial[SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.headerWords]
    === SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER_WORDS,
  "FPP1 header mismatch");
  assert(layout.indirectBinding.offset % 4 === 0
    && layout.indirectBinding.size === 12, "FPP1 indirect triplet mismatch");
  assert(layout.tilesPerPage * SPARSE_CM12_FRAME_PLAN_PRESENTATION_SAMPLES_PER_TILE
    === brickFineResolution ** 3, "FPP1 tile/sample enumeration is not bijective");
  return {
    specialization: `B${brickFineResolution}/P${brickFineResolution}`,
    lanesOwningTiles: layout.tilesPerPage,
    samplesPerTile: SPARSE_CM12_FRAME_PLAN_PRESENTATION_SAMPLES_PER_TILE,
    samplesPerPage: brickFineResolution ** 3,
    pageRecordBytes: 4 * SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_WORDS,
    arenaBytes: layout.totalBytes,
    indirectByteOffset: layout.indirectBinding.offset,
  };
});

console.log(JSON.stringify({
  abi: "FPP1/v1",
  native: "B16/P16",
  variants,
  scheduling: {
    build: "one validation workgroup per physical brick; one atomic append per dirty page",
    execute: "GPU-authored compact indirect count; no presentation dispatch for clean pages",
    authority: "FPL1 Current generation/topology/packet; no CPU count or parity",
  },
  transaction: {
    samples: "non-visible candidate page bank",
    commit: "dirty-tile copy to accepted prefix only after exact mask equality",
    localFault: "omit the affected page; unrelated pages proceed",
    globalFault: "zero indirect dispatch",
  },
  causes: ["density", "phase", "topology-created", "topology-retired",
    "page-activated", "page-retired", "boundary", "dependency-closure"],
}, null, 2));
