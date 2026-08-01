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
  assert.equal(octreeFluidGatedBoundaryWouldRefine({
    boundaryIntersects: true,
    liquidProximityProtected: false,
    minimumPhi: 2,
    protectionWidth: 1,
    fluidGated: true,
  }), false, "a dry terrain crossing is the intended coarse case");
  assert.equal(octreeFluidGatedBoundaryWouldRefine({
    boundaryIntersects: true,
    liquidProximityProtected: true,
    minimumPhi: 2,
    protectionWidth: 1,
    fluidGated: true,
  }), true, "the existing interface band must split before liquid contact");
  assert.equal(octreeFluidGatedBoundaryWouldRefine({
    boundaryIntersects: true,
    liquidProximityProtected: false,
    minimumPhi: 0.75,
    protectionWidth: 1,
    fluidGated: true,
  }), true, "cold bootstrap must apply the same pre-contact protection width");
  assert.equal(octreeFluidGatedBoundaryWouldRefine({
    boundaryIntersects: true,
    liquidProximityProtected: false,
    minimumPhi: -0.25,
    protectionWidth: 1,
    fluidGated: true,
  }), true, "a boundary leaf containing liquid remains refined");
  assert.equal(octreeFluidGatedBoundaryWouldRefine({
    boundaryIntersects: true,
    liquidProximityProtected: false,
    minimumPhi: 2,
    protectionWidth: 1,
    fluidGated: false,
  }), true, "the control arm preserves unconditional boundary refinement");
});

test("both fine and cooperative coarse refinement use the same fluid gate", () => {
  assert.match(octreeProjectionShader,
    /override fluidGatedBoundaryRefinement: bool = true/);
  const fine = octreeProjectionShader.slice(
    octreeProjectionShader.indexOf("fn leafNeedsRefinement"),
    octreeProjectionShader.indexOf("fn splitLeaf"),
  );
  assert.match(fine,
    /crossesBoundary[\s\S]*if \(!fluidGatedBoundaryRefinement\) \{ return true; \}[\s\S]*boundaryLiquidMinimumPhi[\s\S]*minimumPhi <= params\.solve\.w \* params\.cellRelax\.x/);
  const coarse = octreeProjectionShader.slice(
    octreeProjectionShader.indexOf("fn refineCoarseBlock"),
    octreeProjectionShader.indexOf("fn ownerAtIsTooFine"),
  );
  assert.match(coarse,
    /boundaryDecision = crossesBoundary[\s\S]*fluidGatedBoundaryRefinement && crossesBoundary[\s\S]*boundaryLiquidMinimumPhi[\s\S]*params\.solve\.w \* params\.cellRelax\.x/);
  assert.match(coarse,
    /pressureEvidence = pressureRefinementEvidence\(origin, size\)[\s\S]*pressureRetained = pressureRetentionAt\(origin\) > 0u[\s\S]*fineSummaryFactor == 1u[\s\S]*pressureRetained && \(!fluidGatedBoundaryRefinement \|\| !crossesBoundary\)[\s\S]*boundaryDecision/);
});

test("fine boundary gating stays compact while factor one retains pressure support", () => {
  const evidence = octreeProjectionShader.slice(
    octreeProjectionShader.indexOf("fn pressureRefinementEvidence"),
    octreeProjectionShader.indexOf("fn pressureRetentionAt"),
  );
  assert.match(evidence,
    /compactProtectionWidth[\s\S]*fineSummaryFactor == 1u/,
    "factor-4\/8 must keep compact pressure reach independently of boundary policy");
  assert.match(octreeProjectionShader,
    /retainPressureHysteresis = fineSummaryFactor == 1u[\s\S]*currentEvidence = retainPressureHysteresis/,
    "fine factor-4\/8 runs must not publish tile-wide hysteresis");
  const fine = octreeProjectionShader.slice(
    octreeProjectionShader.indexOf("fn leafNeedsRefinement"),
    octreeProjectionShader.indexOf("fn splitLeaf"),
  );
  assert.match(fine,
    /pressureRefinementEvidence\(origin, size\)[\s\S]*pressureRetained = pressureRetentionAt\(origin\) > 0u[\s\S]*fineSummaryFactor == 1u[\s\S]*if \(crossesBoundary\)[\s\S]*return minimumPhi <= params\.solve\.w \* params\.cellRelax\.x[\s\S]*if \(pressureRetained\)/,
    "current local evidence wins while factor-specific retention follows boundary rejection");
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
