import assert from "node:assert/strict";
import test from "node:test";

import {
  packSvoFluidCoverageFrame,
  planSvoFluidCoverageVolume,
  quantizeSvoFluidCoverage,
  reduceSvoFluidCoverage,
  svoFluidCoverageFromSignedDistance,
  svoFluidCoverageLod,
  svoFluidCoverageWGSL,
  SVO_FLUID_COVERAGE_LAYOUT,
  SVO_FLUID_COVERAGE_LIMITS,
} from "../lib/svo-fluid-coverage";
import { sampleSparseScenePrimitiveCell } from "../lib/webgpu-sparse-scene-proxies";

const domain = {
  fieldDimensions: [320, 96, 80] as const,
  worldOrigin_m: [-8, 0, -4] as const,
  cellSize_m: [0.05, 0.05, 0.05] as const,
};

test("the volume covers the solver's coarse field and nothing else", () => {
  const plan = planSvoFluidCoverageVolume({ ...domain, cellsPerTexel: 2 });
  assert.deepEqual([...plan.dimensions], [160, 48, 40], "one texel spans two coarse cells on each axis");
  assert.deepEqual([...plan.texelSize_m].map((value) => Number(value.toFixed(6))), [0.1, 0.1, 0.1]);
  // The coarse level set spans the container box exactly, which is the same
  // mapping the raster composite uses when it samples the field.
  assert.deepEqual([...plan.origin_m].map((value) => Number(value.toFixed(6))), [-8, 0, -4]);
  assert.deepEqual([...plan.extent_m].map((value) => Number(value.toFixed(6))), [16, 4.8, 4]);
  assert.deepEqual([...plan.fieldDimensions], [320, 96, 80]);
});

test("the mip chain reaches the level a room-crossing cone actually reads", () => {
  const plan = planSvoFluidCoverageVolume({ ...domain, cellsPerTexel: 2 });
  assert.equal(plan.levelCount, 8, "160 texels on the long axis needs levels 0..7");
  const coarsestTexel_m = plan.texelSize_m[0] * 2 ** (plan.levelCount - 1);
  assert.ok(coarsestTexel_m >= plan.extent_m[1], "the coarsest level must span the shortest axis in one texel");
});

test("allocation stays small enough to rebuild every frame", () => {
  const plan = planSvoFluidCoverageVolume({ ...domain, cellsPerTexel: 2 });
  assert.ok(plan.allocatedBytes < 2 * 1024 * 1024,
    `a per-frame volume must stay well under the paged atlas it complements, got ${plan.allocatedBytes} bytes`);
  const finest = planSvoFluidCoverageVolume({ ...domain, cellsPerTexel: 1 });
  assert.ok(finest.allocatedBytes > 7 * plan.allocatedBytes,
    "halving the resolution must cut allocation by roughly eight, which is why two is the default");
});

test("a field that does not divide evenly rounds up rather than dropping cells", () => {
  // The trailing texel then averages fewer real cells, treating the remainder as
  // air. It borders the domain wall, where there is no water to lose.
  const plan = planSvoFluidCoverageVolume({ ...domain, fieldDimensions: [321, 96, 80], cellsPerTexel: 2 });
  assert.deepEqual([...plan.dimensions], [161, 48, 40]);
});

test("an oversized domain is refused rather than silently truncated", () => {
  assert.throws(() => planSvoFluidCoverageVolume({
    ...domain,
    fieldDimensions: [(SVO_FLUID_COVERAGE_LIMITS.maximumTexelsPerAxis + 1) * 2, 8, 8],
    cellsPerTexel: 2,
  }), /per-axis texel limit/);
});

test("coverage uses the same interface band as solid voxelization", () => {
  const cellSize = [0.1, 0.1, 0.1] as const;
  const diagonal = Math.hypot(...cellSize);
  assert.equal(svoFluidCoverageFromSignedDistance(0, diagonal), 0.5, "the interface itself is half covered");
  assert.equal(svoFluidCoverageFromSignedDistance(-diagonal, diagonal), 1, "well inside saturates");
  assert.equal(svoFluidCoverageFromSignedDistance(diagonal, diagonal), 0, "well outside contributes nothing");

  // Parity with the solid path: an identical signed distance must produce an
  // identical fraction, so a shadow does not change character where water meets
  // stone inside one cell.
  for (const distance of [-0.09, -0.03, 0, 0.02, 0.049]) {
    // A half-space whose face sits `distance` above the sampled cell centre.
    const solid = sampleSparseScenePrimitiveCell(
      [{ kind: "box", center: [0, -distance - 1, 0], halfExtents: [10, 1, 10], materialId: 3 }],
      [0, 0, 0], [...cellSize],
    );
    assert.ok(Math.abs(solid.solidFraction - svoFluidCoverageFromSignedDistance(solid.solidSignedDistance, diagonal)) < 1e-9,
      `fluid and solid coverage must agree at distance ${distance}`);
  }
});

test("a non-finite distance reads as empty rather than poisoning the volume", () => {
  assert.equal(svoFluidCoverageFromSignedDistance(Number.NaN, 0.1), 0);
  assert.equal(svoFluidCoverageFromSignedDistance(Number.POSITIVE_INFINITY, 0.1), 0);
  assert.throws(() => svoFluidCoverageFromSignedDistance(0, 0), /must be positive/);
});

test("reduction is the mean, matching the node-mip pyramid's mean lanes", () => {
  assert.equal(reduceSvoFluidCoverage([1, 1, 1, 1, 0, 0, 0, 0]), 0.5);
  assert.equal(reduceSvoFluidCoverage([0, 0, 0, 0, 0, 0, 0, 0]), 0);
  assert.equal(reduceSvoFluidCoverage([1, 1, 1, 1, 1, 1, 1, 1]), 1);
  assert.throws(() => reduceSvoFluidCoverage([1, 1, 1, 1]), /eight children/);
  assert.throws(() => reduceSvoFluidCoverage([2, 0, 0, 0, 0, 0, 0, 0]), /\[0, 1\]/);
});

test("quantization is stable across the byte lane", () => {
  assert.equal(quantizeSvoFluidCoverage(0), 0);
  assert.equal(quantizeSvoFluidCoverage(1), 1);
  assert.equal(quantizeSvoFluidCoverage(2), 1, "out-of-range input clamps rather than wrapping");
  assert.ok(Math.abs(quantizeSvoFluidCoverage(0.5) - 128 / 255) < 1e-12);
});

test("lod resolves against this volume's texel, not the finest simulation cell", () => {
  assert.equal(svoFluidCoverageLod(0.1, 0.1), 0);
  assert.equal(svoFluidCoverageLod(0.05, 0.1), 0, "a cone narrower than a texel still reads level zero");
  assert.equal(svoFluidCoverageLod(0.4, 0.1), 2);
  // A downsampled volume must report the level that covers the footprint, which
  // is one lower than the same cone would read against a finest-cell pyramid.
  assert.equal(svoFluidCoverageLod(0.4, 0.05) - svoFluidCoverageLod(0.4, 0.1), 1);
});

test("the packed frame is the ABI the shader declares", () => {
  const plan = planSvoFluidCoverageVolume({ ...domain, cellsPerTexel: 2 });
  const packed = packSvoFluidCoverageFrame({
    origin_m: plan.origin_m, texelSize_m: plan.texelSize_m, dimensions: plan.dimensions,
    levelCount: plan.levelCount, valid: true, generation: 7,
  });
  assert.equal(packed.byteLength, SVO_FLUID_COVERAGE_LAYOUT.frameWords * 4);
  const floats = new Float32Array(packed), words = new Uint32Array(packed);
  assert.ok(Math.abs(floats[0] - plan.origin_m[0]) < 1e-6);
  assert.equal(words[3], plan.levelCount);
  assert.ok(Math.abs(floats[4] - plan.texelSize_m[0]) < 1e-6);
  assert.equal(words[7], 1);
  assert.deepEqual([words[8], words[9], words[10]], [...plan.dimensions]);
  assert.equal(words[11], 7);

  const invalid = new Uint32Array(packSvoFluidCoverageFrame({
    origin_m: plan.origin_m, texelSize_m: plan.texelSize_m, dimensions: plan.dimensions,
    levelCount: plan.levelCount, valid: false, generation: 0,
  }));
  assert.equal(invalid[7], 0, "an invalid frame must be expressible so the shader can skip every fetch");
});

test("the sampler helper fails closed outside the solver box", () => {
  // Clamp-to-edge addressing would otherwise project the boundary texel along
  // every axis and shadow the entire scene downwind of one full edge cell.
  assert.match(svoFluidCoverageWGSL, /if\(any\(uvw<vec3f\(0\.0\)\)\|\|any\(uvw>vec3f\(1\.0\)\)\)\{return 0\.0;\}/);
  assert.match(svoFluidCoverageWGSL, /if\(!svoFluidCoverageReady\(frame\)\)\{return 0\.0;\}/);
  assert.match(svoFluidCoverageWGSL, /textureSampleLevel\(volume,volumeSampler,uvw,level\)/);
  assert.doesNotMatch(svoFluidCoverageWGSL, /@group|@binding/,
    "the sampling library must stay binding-free so the dry shader can place it in its own group");
});
