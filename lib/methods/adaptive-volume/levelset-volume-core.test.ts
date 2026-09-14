import assert from "node:assert/strict";
import test from "node:test";
import {
  LEVELSET_VOLUME_GLOBAL_HEADER,
  LEVELSET_VOLUME_INVALID,
  LEVELSET_VOLUME_WORKGROUP_SIZE,
  createLevelSetVolumeInitialWords,
  createLevelSetVolumeLayout,
  levelSetVolumeClearInvocationCount,
  levelSetVolumeDispatchCount,
} from "./levelset-volume-layout";
import { createLevelSetVolumeWGSL, extendSignedPhiAlongGradient } from
  "./levelset-volume-core.wgsl";

test("frontier phi extension follows the signed outgoing normal", () => {
  assert.ok((extendSignedPhiAlongGradient(-1.49e-7, [1, 0, 0], [1, 0, 0]) ?? -1) > 0,
    "an outward step from a near-zero liquid sample becomes air");
  assert.ok((extendSignedPhiAlongGradient(1.49e-7, [-1, 0, 0], [1, 0, 0]) ?? 1) < 0,
    "the opposite oriented interface becomes liquid on its outgoing side");
  assert.equal(extendSignedPhiAlongGradient(0, [0, 0, 0], [1, 0, 0]), undefined,
    "a degenerate local gradient cannot authorize an extension");
});

test("adaptive phi allocation scales with active budgets and has disjoint generation slots", () => {
  const layout = createLevelSetVolumeLayout({ baseWords: 70, activeCellCapacity: 96,
    vertexCapacity: 180, maximumArenaWords: 100_000 });
  assert.equal(layout.headerBaseWords % 64, 0);
  assert.equal(layout.hashCapacity, 512);
  assert.equal(layout.slots[0].totalWords, layout.slots[1].baseWords);
  assert.equal(layout.slots[1].baseWords - layout.slots[0].baseWords,
    layout.slotStrideWords);
  assert.ok(layout.slots[0].cornerRefsBaseWords + 8 * layout.activeCellCapacity
    <= layout.slots[0].hashBaseWords);
  // Stable/template capacity is deliberately absent from this contract.
  assert.equal(levelSetVolumeClearInvocationCount(layout), 8 * 96);
  assert.equal(levelSetVolumeDispatchCount(65), 2);
  assert.equal(LEVELSET_VOLUME_WORKGROUP_SIZE, 64);
});

test("adaptive phi storage scales with resolution for identical brick coverage", () => {
  const samples = [1, 2, 4, 8].map(resolution => {
    const layout = createLevelSetVolumeLayout({
      activeCellCapacity: resolution ** 3,
      vertexCapacity: (resolution + 1) ** 3,
    });
    return { resolution, words: layout.totalWords - layout.baseWords };
  });
  for (let index = 1; index < samples.length; index += 1) {
    assert.ok(samples[index]!.words > samples[index - 1]!.words,
      `r${samples[index]!.resolution} must allocate more than r${samples[index - 1]!.resolution}`);
  }
  assert.ok(samples[0]!.words * 20 < samples.at(-1)!.words,
    "coarse coverage must not inherit the finest-rung storage footprint");
});

test("bootstrap is a small global header and leaves publication invalid", () => {
  const layout = createLevelSetVolumeLayout({ activeCellCapacity: 8, vertexCapacity: 27 });
  const words = createLevelSetVolumeInitialWords(layout);
  assert.equal(words.length, 32);
  assert.equal(words[LEVELSET_VOLUME_GLOBAL_HEADER.acceptedSlot], LEVELSET_VOLUME_INVALID);
  assert.equal(words[LEVELSET_VOLUME_GLOBAL_HEADER.slot0Base], layout.slots[0].baseWords);
  assert.equal(words[LEVELSET_VOLUME_GLOBAL_HEADER.slotStride], layout.slotStrideWords);
});

test("layout rejects an underprovisioned or non-power-of-two vertex hash", () => {
  assert.throws(() => createLevelSetVolumeLayout({ activeCellCapacity: 8,
    vertexCapacity: 27, hashCapacity: 16 }), /at least vertexCapacity/);
  assert.throws(() => createLevelSetVolumeLayout({ activeCellCapacity: 8,
    vertexCapacity: 27, hashCapacity: 63 }), /power of two/);
  assert.throws(() => createLevelSetVolumeLayout({ activeCellCapacity: 8,
    vertexCapacity: 27, maximumArenaWords: 100 }), /exceeding/);
});

test("WGSL exposes fenced sampling and the complete production lifecycle", () => {
  const layout = createLevelSetVolumeLayout({ activeCellCapacity: 8, vertexCapacity: 27 });
  const wgsl = createLevelSetVolumeWGSL({ layout,
    acceptedGenerationExpression: "cnxSourceGeneration()",
    buildGenerationExpression: "candidateGeneration()",
    buildSlotExpression: "candidateSlot()",
    buildCellCountExpression: "candidateCellCount()",
    buildCellAtOrdinal: ordinal => `candidateCell(${ordinal})`,
    acceptedCellOrdinal: cell => `cnxCellOrdinalUnchecked(${cell})`,
    acceptedOwnerCellAt: q => `ownerCellAt(${q})`,
    buildOwnerCellAt: q => `candidateOwnerCellAt(${q})`,
    authoredSample: p => `vec2f(lsvAuthoredPhi(${p}),3.0)`,
    velocitySample: p => `vec4f(effectiveVelocity(${p}),1.0)`,
    dtExpression: "frameDt()", constraintWidthExpression: "constraintWidth()",
  });
  for (const symbol of ["fn lsvPhiAt(", "fn lsvCellPhi(", "fn lsvGradientAt(",
    "fn lsvCellLiquid(", "fn lsvPhiMetricAt(", "fn lsvCatalogCellCorners(",
    "fn lsvCompileConstraints(", "fn lsvTransferPhi(", "fn lsvAdvectPhi(",
    "fn lsvApplyConstraints(",
    "fn lsvSealTopology(", "fn lsvPublishTopology("]) assert.match(wgsl, new RegExp(symbol.replace("(", "\\(")));
  assert.match(wgsl, /fn lsvInvalidPhi\(\)->f32\{var bits=0x7fc00000u/);
  assert.match(wgsl, /return LsvPhiSample\(lsvInvalidPhi\(\),false,false/);
  assert.match(wgsl, /let shapeWins=select\(\(shapePhi>oldPhi\),\(shapePhi<oldPhi\),\(shape\.y>0\.0\)\)/);
  assert.match(wgsl, /cnxSourceGeneration\(\)/);
  assert.doesNotMatch(wgsl, /for\s*\([^)]*LSV_CELL_CAPACITY/);
});
