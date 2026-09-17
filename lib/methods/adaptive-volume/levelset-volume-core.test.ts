import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
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
import { consumeSignedPhaseClearance, createLevelSetVolumeWGSL, extendSignedPhiAlongGradient } from
  "./levelset-volume-core.wgsl";

test("normalized-gradient extrapolation follows the signed outgoing normal", () => {
  assert.ok((extendSignedPhiAlongGradient(-1.49e-7, [1, 0, 0], [1, 0, 0]) ?? -1) > 0,
    "an outward step from a near-zero liquid sample becomes air");
  assert.ok((extendSignedPhiAlongGradient(1.49e-7, [-1, 0, 0], [1, 0, 0]) ?? 1) < 0,
    "the opposite oriented interface becomes liquid on its outgoing side");
  assert.equal(extendSignedPhiAlongGradient(0, [0, 0, 0], [1, 0, 0]), undefined,
    "a degenerate local gradient cannot authorize an extension");
});

test("unsampled advection consumes only certified phase clearance", () => {
  assert.ok(Math.abs(consumeSignedPhaseClearance(3.3166249, 1.0602026)! - 2.2564223) < 1e-6,
    "the measured sparse-frontier failure remains certified air");
  assert.equal(consumeSignedPhaseClearance(-2, 0.75), -1.25,
    "liquid clearance preserves its sign while consuming travel");
  assert.equal(consumeSignedPhaseClearance(.5, .5), undefined,
    "a path that can reach the contour cannot receive phase-only support");
  assert.equal(consumeSignedPhaseClearance(1, -1), undefined,
    "travel must be a finite nonnegative length");
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
    releasedWallPhi: p => `releasedWallPhiAt(${p})`,
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
  assert.match(wgsl, /if\(\(frameDt\(\)\)>0\.0\)\{releasedWall=releasedWallPhiAt\(position\);\}/,
    "dt zero must bypass wall-gap publication exactly");
  assert.equal((wgsl.match(/lsvStoreAdvectedPhi\(slot,destination,vertex/g) ?? []).length, 3,
    "deep, sparse fallback, and ordinary advection must all publish through the wall carve");
  assert.match(wgsl, /if\(wallWins\)\{support=select\(select\(LSV_SUPPORT_DEEP_AIR,LSV_SUPPORT_DEEP_LIQUID,phi<0\.0\),\s*LSV_SUPPORT_METRIC,abs\(phi\)<=4\.0\);\}/,
    "a released-wall contour must become a metric seed for redistance");
  assert.doesNotMatch(wgsl, /for\s*\([^)]*LSV_CELL_CAPACITY/);
});

test("a brick plane is per slot and absent without a brick roster", () => {
  const without = createLevelSetVolumeLayout({ activeCellCapacity: 8, vertexCapacity: 27 });
  assert.equal(without.brickCapacity, 0);
  assert.equal(without.slots[0].cornerRefsBaseWords, without.slots[0].brickPlaneBaseWords,
    "an absent roster must consume no words");
  const layout = createLevelSetVolumeLayout({ activeCellCapacity: 64,
    vertexCapacity: 125, brickCapacity: 40 });
  assert.equal(layout.brickCapacity, 40);
  // Base ordinal + phi resolution per brick, and nothing per phi cell: the
  // plane is bisected for the inverse map instead of scattering an id array.
  assert.ok(layout.slots[0].brickPlaneBaseWords + 2 * 40
    <= layout.slots[0].cornerRefsBaseWords);
  assert.ok(layout.slots[0].cornerRefsBaseWords
    - layout.slots[0].brickPlaneBaseWords < 2 * 40 + 64);
  for (const slot of layout.slots) {
    assert.ok(slot.brickPlaneBaseWords >= slot.headerBaseWords);
    assert.ok(slot.brickPlaneBaseWords < slot.totalWords);
  }
  assert.notEqual(layout.slots[0].brickPlaneBaseWords, layout.slots[1].brickPlaneBaseWords,
    "the candidate plan must not overwrite the accepted one");
});

test("the phi-cell plan is emitted only with a roster and degrades inside one scan", () => {
  const layout = createLevelSetVolumeLayout({ activeCellCapacity: 64,
    vertexCapacity: 125, brickCapacity: 40 });
  const common = {
    acceptedGenerationExpression: "cnxSourceGeneration()",
    buildGenerationExpression: "cnxSourceGeneration()",
    buildSlotExpression: "lsvBuildSlot()",
    buildCellCountExpression: "lsvPlannedCellCount()",
    buildCellAtOrdinal: (ordinal: string) => `phiCellAt(${ordinal})`,
    acceptedCellOrdinal: (cell: string) => `phiOrdinalAccepted(${cell})`,
    acceptedOwnerCellAt: (q: string) => `phiOwnerAccepted(${q})`,
    buildOwnerCellAt: (q: string) => `phiOwnerBuild(${q})`,
    authoredSample: (p: string) => `vec2f(lsvAuthoredPhi(${p}),3.0)`,
    velocitySample: (p: string) => `vec4f(effectiveVelocity(${p}),1.0)`,
    dtExpression: "frameDt()", constraintWidthExpression: "0.0",
  } as const;
  const plain = createLevelSetVolumeWGSL({ layout, ...common });
  assert.doesNotMatch(plain, /fn lsvPlanPhiCells\(/,
    "no roster means no plan kernels at all");
  assert.doesNotMatch(plain, /fn lsvScatterPhiCells\(/);

  const planned = createLevelSetVolumeWGSL({ layout, ...common,
    buildCellMemberOrdinal: cell => `phiOrdinalBuild(${cell})`,
    buildOwnerSolverCell: cell => `solverCellOf(${cell})`,
    phiCellPlan: {
      brickCountExpression: "brickCount()",
      bandedBrickPlan: brick => `bandedPlan(${brick})`,
      acceptedBrickPlan: brick => `acceptedPlan(${brick})`,
      brickCellRange: (brick, resolution) => `cellRange(${brick},${resolution})`,
    },
  });
  assert.match(planned, /fn lsvPlanPhiCells\(/);
  // The inverse map is a bisection of the same plane, not a second kernel and
  // not an arena word per phi cell: no new pass, no capacity-shaped dispatch.
  assert.doesNotMatch(planned, /fn lsvScatterPhiCells\(/);
  assert.doesNotMatch(planned, /LSV_PHI_CELL_IDS/);
  assert.match(planned, /while\(low<high\)\{/);
  assert.match(planned, /cellRange\(low,resolution\)/);
  assert.match(planned, /bandedPlan\(brick\)/);
  assert.match(planned, /acceptedPlan\(brick\)/);
  // The band arm is attempted first and the same scan replans at the accepted
  // rungs when it overflows, so an oversized band costs accuracy, not a fault.
  assert.match(planned, /attempt<2u/);
  assert.match(planned, /total>LSV_CELL_CAPACITY/);
  // Membership during a build reads the slot being built, never the accepted
  // one: the accepted plan still describes the previous generation.
  assert.match(planned, /if\(phiOrdinalBuild\(owner\)==INVALID\)\{continue;\}/);
  assert.match(planned, /if\(phiOrdinalBuild\(cell\)==INVALID\)\{continue;\}/);
  // Record word 7 is the owning solver cell, so transport spans and wall
  // clipping keep reading the coarse cell under a finer phi lattice.
  assert.match(planned, /lsvStore\(record\+7u,solverCellOf\(cell\)\);/);
  assert.match(planned, /fn lsvStencilAtPosition\(/);
  assert.match(planned, /fn lsvStencilContains\(/);
  assert.doesNotMatch(planned, /for\s*\([^)]*LSV_CELL_CAPACITY/);
});

test("the resident binds every level-set domain hook to the phi-cell space", () => {
  const source = readFileSync(new URL("./webgpu-sparse-cm12-resident.wgsl.ts",
    import.meta.url), "utf8");
  // Mixing the two spaces would address a phi record by a solver ordinal, so
  // the five hooks must move together.
  assert.match(source, /buildCellCountExpression: "lsvPlannedCellCount\(\)"/);
  assert.match(source, /lsvPhiCellAtOrdinalInSlot\(lsvBuildSlot\(\),\$\{ordinal\}\)/);
  assert.match(source, /cm12PhiCellOrdinalInSlot\(lsvAcceptedSlot\(\),\$\{cell\}\)/);
  assert.match(source, /cm12PhiCellOrdinalInSlot\(lsvBuildSlot\(\),\$\{cell\}\)/);
  assert.match(source, /cm12PhiOwnerCellAtSlot\(lsvAcceptedSlot\(\),\$\{lattice\}\)/);
  assert.match(source, /cm12PhiOwnerCellAtSlot\(lsvBuildSlot\(\),\$\{lattice\}\)/);
  assert.doesNotMatch(source, /acceptedCellOrdinal: cell => `cnxCellOrdinalUnchecked/);
  assert.match(source, /releasedWallPhi: position => `cm12ReleasedWallPhi\(\$\{position\}\)`/);
  const releasedWall = source.slice(source.indexOf("fn cm12ReleasedWallPhi"),
    source.indexOf("` + createLevelSetVolumeWGSL"));
  assert.match(releasedWall, /for\(var face=0u;face<6u;face\+=1u\)/,
    "every physical ClosedWorld plane must be considered regardless of normal distance");
  assert.match(releasedWall, /if\(gravityWeight<=1e-6\)\{return vec2f\(carved,0\.0\);\}/,
    "zero gravity cannot have a released row and must avoid boundary lookups");
  assert.match(releasedWall,
    /if\(expectedInward\*p\.acceleration\[axis\]<=0\.5\*gravityWeight\)\{continue;\}/,
    "plane lookup must retain the pressure release predicate exactly");
  assert.match(releasedWall, /boundary=select\(0\.0,f32\(p\.dimensions\[axis\]\),upper\)/);
  assert.match(releasedWall, /probe\[axis\]=boundary\+epsilon\*expectedInward/,
    "wall discovery must project onto the plane rather than stop after one normal cell");
  assert.match(releasedWall, /for\(var quadrant=0u;quadrant<4u;quadrant\+=1u\)/);
  assert.match(releasedWall, /probe\[tangent0\]\+=epsilon\*select\(-1\.0,1\.0,\(quadrant&1u\)!=0u\)/,
    "four tangential probes must cover coarse/fine patch seams");
  assert.match(releasedWall, /cnxCellIncidenceRangeUnchecked\(cell\)/,
    "wall lookup must use the accepted post-transition CNX image");
  assert.match(releasedWall,
    /state\[destinationFaceVelocity\(\)\+row\]-rowSolidVelocity\(row\)/,
    "gap speed must use the final projected MAC face, not a collocated trace velocity");
  assert.match(releasedWall, /if\(away<=1e-6\)\{continue;\}/);
  assert.match(releasedWall, /p\.frame\.x\*away-interiorDistance/);
  assert.match(releasedWall, /abs\(positionFine\[tangent\]-center\[tangent\]\)<=0\.5\*widths\[tangent\]\+epsilon/,
    "the continuation is limited to the released face's tangential footprint");
  // The band is the presentation surface apron, which is wider than the
  // four-fine-cell metric band the redistance pass maintains.
  assert.match(source, /brickHasPresentationSurfaceSupport\(brick\)/);
  assert.match(source, /bandWidthExpression: "4\.0"/);
});

