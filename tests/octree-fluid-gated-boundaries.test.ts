import assert from "node:assert/strict";
import test from "node:test";
import {
  censusOctreeTopologyLeaves,
  octreeFluidGatedBoundaryWouldRefine,
  octreeProjectionShader,
} from "../lib/webgpu-octree";
import {
  packOctreeOwnerPageWord,
  planOctreeOwnerPages,
} from "../lib/webgpu-octree-owner-pages";

test("fluid protection arrives before contact while dry boundaries may stay coarse", () => {
  const gate = (overrides: Partial<Parameters<typeof octreeFluidGatedBoundaryWouldRefine>[0]>) =>
    octreeFluidGatedBoundaryWouldRefine({
      boundaryIntersects: true,
      liquidProximityProtected: false,
      minimumPhi: 0,
      maximumPhi: 0,
      boundedInterval: true,
      protectionWidth: 1,
      fluidGated: true,
      ...overrides,
    });

  assert.equal(gate({ minimumPhi: 2, maximumPhi: 3 }), false,
    "a dry terrain crossing is the intended coarse case");
  assert.equal(gate({ minimumPhi: 2, maximumPhi: 3, liquidProximityProtected: true }), true,
    "the existing interface band must split before liquid contact");
  assert.equal(gate({ minimumPhi: 0.75, maximumPhi: 1.5 }), true,
    "cold bootstrap must apply the same pre-contact protection width");
  assert.equal(gate({ minimumPhi: -0.25, maximumPhi: 0.5 }), true,
    "a boundary leaf just under the surface remains refined");
  assert.equal(gate({ minimumPhi: 2, maximumPhi: 3, fluidGated: false }), true,
    "the control arm preserves unconditional boundary refinement");

  // The band is a distance to the SURFACE, not a test for liquid. A leaf wholly
  // below the free surface is as uninteresting as one wholly above it, and the
  // signed form used to split every submerged wall leaf at any depth -- 81.5%
  // of the 24x18x16 water box, which is why its interior never coarsened.
  assert.equal(gate({ minimumPhi: -4, maximumPhi: -3 }), false,
    "a leaf wholly deeper than the band may coarsen");
  assert.equal(gate({ minimumPhi: -4, maximumPhi: -3, liquidProximityProtected: true }), true,
    "interface protection still overrides depth");

  // And the reason the deep half must read the MAXIMUM rather than abs(minimum):
  // a leaf can run from deep liquid up through the surface into air. Its minimum
  // is deep and it holds the interface anyway. symmetric-expansion's 32x16x32
  // lattice is tiled by exactly four full-height size-16 leaves of this shape,
  // and abs(minimum) left all four coarse -- four pressure cells for the tank.
  assert.equal(gate({ minimumPhi: -4, maximumPhi: 4 }), true,
    "a leaf spanning liquid, surface and air must stay refined however deep it reaches");

  // An interval that is not proven to cover the candidate may not be used to
  // reject: the centre-sample fallbacks bound nothing, so they keep the
  // original one-sided answer.
  assert.equal(gate({ minimumPhi: -4, maximumPhi: -3, boundedInterval: false }), true,
    "an unbounded interval over-refines rather than under-refines");
  assert.equal(gate({ minimumPhi: 2, maximumPhi: 3, boundedInterval: false }), false,
    "the dry half never needed the interval and is unchanged");

  assert.throws(() => gate({ minimumPhi: 1, maximumPhi: -1 }), RangeError,
    "an inverted interval is a producer bug, not a coarsening decision");
});

test("both fine and cooperative coarse refinement use the same fluid gate", () => {
  assert.match(octreeProjectionShader,
    /override fluidGatedBoundaryRefinement: bool = true/);
  const fine = octreeProjectionShader.slice(
    octreeProjectionShader.indexOf("fn leafNeedsRefinement"),
    octreeProjectionShader.indexOf("fn splitLeaf"),
  );
  assert.match(fine,
    /crossesBoundary[\s\S]*if \(!fluidGatedBoundaryRefinement\) \{ return true; \}[\s\S]*boundaryLiquidWouldRefine\([\s\S]*boundaryLiquidPhiInterval\(origin, size, minimumPhi, maximumPhi\)/);
  const coarse = octreeProjectionShader.slice(
    octreeProjectionShader.indexOf("fn refineCoarseBlock"),
    octreeProjectionShader.indexOf("fn ownerAtIsTooFine"),
  );
  assert.match(coarse,
    /boundaryDecision = crossesBoundary[\s\S]*fluidGatedBoundaryRefinement && crossesBoundary[\s\S]*boundaryLiquidWouldRefine\([\s\S]*boundaryLiquidPhiInterval\(origin, size, range\.z, range\.w\)/);
  // Authored refinement regions bound both paths — see
  // tests/octree-refinement-regions.test.ts — while the fluid gate the two
  // paths share remains unchanged beneath those bounds.
  assert.match(coarse,
    /pressureEvidence = pressureRefinementEvidence\(origin, size\)[\s\S]*forcedByRegion = refinementRegionForcesSplit\(origin, size\)[\s\S]*decision = forcedByRegion \|\| \(!refinementRegionHoldsLeaf\(origin, size\)\s*&& \(pressureEvidence \|\| adaptivity <= 0\.0 \|\| boundaryDecision\)\)/);
});

test("adaptive Losasso topology consumes the published leaf interval", () => {
  const corrected = octreeProjectionShader.slice(
    octreeProjectionShader.indexOf("fn correctedCoarsePhi"),
    octreeProjectionShader.indexOf("fn coarseClassificationPhi"),
  );
  assert.match(corrected,
    /let minimum=bitcast<f32>\(coarseWord\(entry\+3u\)\);let maximum=bitcast<f32>\(coarseWord\(entry\+4u\)\)/,
    "topology must read the graph publisher's corner minimum and maximum");
  assert.match(corrected,
    /crossing=minimum<=0\.0&&maximum>=0\.0[\s\S]*crossingFlag=\(flags&4u\)!=0u[\s\S]*CorrectedCoarsePhi\(true,value,minimum,maximum,size\)/,
    "a positive centre with a corner zero crossing must retain interface evidence");
  assert.doesNotMatch(corrected,
    /CorrectedCoarsePhi\(true,value,value,value,size\)/,
    "adaptive interval evidence must not collapse back to the centre sample");
});

test("fine boundary gating stays candidate-local", () => {
  const evidence = octreeProjectionShader.slice(
    octreeProjectionShader.indexOf("fn pressureRefinementEvidence"),
    octreeProjectionShader.indexOf("fn boundaryLiquidMinimumPhi"),
  );
  assert.match(evidence,
    /compactProtectionWidth[\s\S]*fineSummaryFactor == 1u/,
    "factor-4\/8 must keep compact pressure reach independently of boundary policy");
  assert.doesNotMatch(octreeProjectionShader, /pressureRetentionAt|PRESSURE_RETENTION_GENERATIONS/,
    "one surface leaf must not force its entire topology tile to unit size");
  const fine = octreeProjectionShader.slice(
    octreeProjectionShader.indexOf("fn leafNeedsRefinement"),
    octreeProjectionShader.indexOf("fn splitLeaf"),
  );
  assert.match(fine,
    /pressureRefinementEvidence\(origin, size\)[\s\S]*if \(crossesBoundary\)[\s\S]*return boundaryLiquidWouldRefine\(/,
    "current local evidence wins before the boundary decision");
});

test("factor-one band one admits size-two interface and closed-wall cuts", () => {
  assert.match(octreeProjectionShader,
    /fineSummaryFactor == 1u && size <= finestSurfaceCellSize\(\)[\s\S]*params\.solve\.w <= 1\.0[\s\S]*!summary\.sizingRefinement[\s\S]*return false/,
    "the live cut floor should keep ordinary factor-one crossings on size-two rows at band one");
  const fine = octreeProjectionShader.slice(
    octreeProjectionShader.indexOf("fn leafNeedsRefinement"),
    octreeProjectionShader.indexOf("fn splitLeaf"),
  );
  assert.match(fine,
    /crossesBoundary[\s\S]*fineSummaryFactor == 1u && size <= finestSurfaceCellSize\(\)[\s\S]*params\.solve\.w <= 1\.0[\s\S]*return false/,
    "a closed wall must preserve the same supported size-two cut");
  assert.match(octreeProjectionShader,
    /fn wallBandCells\(\)[\s\S]*topologyDialByte\(8u\)[\s\S]*let width = wallBandCells\(\)/,
    "the closed-wall strip must use its independent live UI control");
  assert.doesNotMatch(octreeProjectionShader, /max\(3\.0, params\.solve\.w\)/,
    "wall reach must no longer be silently hard-floored behind the UI");
});

test("topology census deduplicates one coarse leaf spanning owner pages", () => {
  const plan = planOctreeOwnerPages([16, 16, 16], { maximumPages: 8 });
  const arena = new Uint32Array(plan.allocatedWords);
  arena[1] = 8;
  arena[3] = 8;
  arena[7] = 9;
  for (let logical = 0; logical < 8; logical += 1) {
    arena[plan.ownerDirectoryOffsetWords + logical] = logical + 1;
  }
  for (let z = 0; z < 16; z += 1) {
    for (let y = 0; y < 16; y += 1) {
      for (let x = 0; x < 16; x += 1) {
        const logical = Math.floor(x / 8)
          + 2 * (Math.floor(y / 8) + 2 * Math.floor(z / 8));
        const local = (x & 7) + 8 * (y + 8 * z);
        arena[plan.ownerPagesOffsetWords + logical * plan.pageVoxels
          + (local % plan.pageVoxels)] =
          packOctreeOwnerPageWord([x, y, z], [0, 0, 0], 16);
      }
    }
  }
  assert.deepEqual(censusOctreeTopologyLeaves(arena, plan, 16), {
    generation: 9,
    residentOwnerPages: 8,
    topologyLeaves: 1,
    representedCells: 4096,
    leafCountsBySize: { 16: 1 },
    coarseLeafCountsByOriginY: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    boundaryStripLeafCountsBySize: {
      xLow: { 16: 1 }, xHigh: { 16: 1 },
      yLow: { 16: 1 }, yHigh: { 16: 1 },
      zLow: { 16: 1 }, zHigh: { 16: 1 },
    },
  });
});
